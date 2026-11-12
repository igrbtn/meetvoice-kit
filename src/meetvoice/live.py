"""Live captions with translation from growing audio tracks: draft + final.

Every POLL_SEC each track's new audio is appended to its open buffer (audio since its last
final line). Two kinds of output:
  - draft: the open buffer of a call track decoded once without word timestamps, shown
    gray and replaced wholesale every step (asr.draft_window). Skipped when the GPU is busy.
  - final: when the speaker pauses (audio.pause_cut) or the buffer reaches MAX_BUF_SEC, the
    buffer up to the pause is decoded with the glossary prompt (asr.transcribe_window) and
    appended to `entries`; only finals are translated, on their own thread.
Finals are never edited and drafts never stitched, so no words are doubled or lost between
passes (stitching partial hypotheses without word timestamps doubled words and looped).
The mic gets finals only by default (mic_drafts=False): the user knows what they said.

Routing of a final:
  1. second-language finder (Uzbek) first: if the audio is not Russian/English by ear and
     decodes as Uzbek, those lines win; after Uzbek was heard on a track the finder is
     forced for UZ_STICKY_SEC (short Uzbek windows full of loanwords pass for Russian), and
     the draft shows "..." (Whisper drafts of Uzbek are noise);
  2. otherwise Whisper; its detected language is kept per line;
  3. foreign lines (anything but `native`) are translated to `foreign_to` always, other
     lines only when `translate_to` is set.
Trust: a line is `unsure` on weak Whisper signals (asr.unsure_reasons), on a language nobody
in the meeting speaks, or when it is foreign and Whisper's p(lang) < LANG_SURE ("Are you
naked?" over Uzbek noise had 0.20). Unsure lines are shown marked and NOT translated:
translated garbage reads as fact. Fillers ("угу" comes out as "Mm-hmm", "Gracias") are shown
dim and not translated. Dropped: glossary read back, "Субтитры...", lines without letters or
of a couple of letters repeated, a loop with another signal, the same line within REPEAT_SEC.
Mic track: the call coming back through speakers is cut by envelope correlation with the
call track (audio.drop_crosstalk); Uzbek on the mic is people in the room and is dropped;
on the mic only MIC_LANGS survive (the rest is echo garbage).
"""

import json
import logging
import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .asr import norm
from .audio import SR, WavTail, crosstalk_lag, drop_crosstalk, pause_cut, resample, speech_regions

log = logging.getLogger("meetvoice.live")

POLL_SEC = 1.0            # one step: read new audio, close finals, one draft
MIN_FINAL_SEC = 2.0       # an utterance shorter than this waits for more audio
MAX_BUF_SEC = 8.0         # no pause by then: cut at the quietest point
SILENCE_RMS = 0.004       # whole final below this = nothing said
UZ_STICKY_SEC = 90.0
CONTEXT_LINES = 4         # earlier lines given to the translator
HIST_SEC = 60.0           # audio kept per track for the echo check
ECHO_MIN_SEC = 5.0        # enough of both tracks to find the echo lag
REPEAT_SEC = 15.0         # the same final again within this: Whisper repeating itself
LANG_SURE = 0.5           # a foreign line below this p(lang) is unsure
MIC_LANGS = {"ru", "en", ""}
EXPECTED_LANGS = {"ru", "en", "uz"}

SUBTITLES_RE = re.compile(r"^\W*(субтитр\w*\W*)+$", re.I)
FILLERS = set("""mm hmm mmhmm mhm uh huh uhhuh uhum um hm hmm ah oh okay ok yeah yep yes no gracias
угу ага да нет ну м мм э а ох хм ок окей""".split())


def is_filler(text: str) -> bool:
    words = norm(text.replace("-", " ")).split()
    return bool(words) and all(w in FILLERS for w in words)


def is_noise_text(text: str) -> bool:
    """"O-O-O-O-O..." and the like: a long line made of a couple of characters."""
    letters = [c for c in text.lower() if c.isalpha()]
    return len(letters) >= 12 and len(set(letters)) <= 3


def glossary_readback(text: str, glossary: str) -> bool:
    """On noise Whisper reads the prompt back: a line made mostly of glossary terms."""
    terms = {norm(t) for t in re.split(r"[,;\n]", glossary) if norm(t)}
    words = {norm(w) for w in re.split(r"[,;]", text) if norm(w)}
    return len(words) >= 3 and sum(w in terms for w in words) / len(words) >= 0.5


@dataclass
class Track:
    path: Path | None     # a WAV file that keeps growing (see audio.WavTail), or None with `source`
    speaker: str          # label shown in captions
    mic: bool = False     # the local user's microphone (vs the call / system audio)
    source: object = None  # instead of a file: anything with .rate and .read_new() -> mono float32


class PushSource:
    """Source for audio that arrives as callbacks (a capture SDK, a WebRTC sink, a socket):
    call push(samples) from any thread; LiveTranscriber pulls."""

    def __init__(self, rate: int):
        self.rate = rate
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    def push(self, samples) -> None:
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        with self._lock:
            self._chunks.append(x)

    def read_new(self) -> np.ndarray:
        with self._lock:
            chunks, self._chunks = self._chunks, []
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


class _State:
    def __init__(self, track: Track):
        self.track = track
        self.tail = track.source if track.source is not None else WavTail(track.path)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.buffer_start = 0.0   # seconds from the start of the recording
        self.last_uz = -1e9
        self.hist = np.zeros(0, dtype=np.float32)
        self.hist_start = 0.0
        self.draft_text: str | None = None
        self.recent: list[tuple[float, str]] = []   # (end, normalized text) of the last finals

    def pull(self) -> None:
        new = self.tail.read_new()
        if new.size:
            new = resample(new, self.tail.rate)
            self.buffer = np.concatenate([self.buffer, new])
            self.hist = np.concatenate([self.hist, new])
            extra = self.hist.size - int(HIST_SEC * SR)
            if extra > 0:
                self.hist = self.hist[extra:]
                self.hist_start += extra / SR

    @property
    def hist_end(self) -> float:
        return self.hist_start + self.hist.size / SR

    def span(self, a: float, b: float) -> np.ndarray:
        i = max(0, int((a - self.hist_start) * SR))
        return self.hist[i:max(i, int((b - self.hist_start) * SR))]

    def take(self, cut: int) -> tuple[float, np.ndarray]:
        start, window = self.buffer_start, self.buffer[:cut]
        self.buffer = self.buffer[cut:]
        self.buffer_start += cut / SR
        return start, window


class LiveTranscriber:
    """
    transcriber(audio, draft=None) -> [{start, end, text, lang, unsure}]
                                     e.g. partial(asr.transcribe_window, prompt=asr.glossary_prompt(g))
    drafter(audio) -> str | None     e.g. partial(asr.draft_window, prompt=...); None = no drafts
    lang_probs(audio) -> {lang: p}   e.g. asr.lang_probs; None = no lang_weak check
    uz_finder(audio, force) -> (foreign: bool, [segments])   e.g. partial(uzbek.live_find, path); or None
    translator(text, target, context=, source_lang=) -> str  e.g. translate.Translator(..., glossary=g)
    glossary: the same terms as in the prompt, to drop lines that are the prompt read back
    """

    def __init__(self, tracks: list[Track], transcriber, translator=None, uz_finder=None,
                 translate_to: str = "", foreign_to: str = "ru", native: str = "ru",
                 out_path: Path | None = None, drafter=None, lang_probs=None, glossary: str = "",
                 mic_drafts: bool = False):
        self.transcriber, self.translator, self.uz_finder = transcriber, translator, uz_finder
        self.drafter, self.lang_probs, self.glossary = drafter, lang_probs, glossary
        self.translate_to, self.foreign_to, self.native = translate_to, foreign_to, native
        self.mic_drafts = mic_drafts
        self.out_path = Path(out_path) if out_path else None
        self.entries: list[dict] = []
        self.drafts: dict[str, dict] = {}   # speaker -> {"text", "start", "end"} of the open utterance
        self.error: str | None = None
        self._states = [_State(t) for t in tracks]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._tq: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._flush = False

    # -- public -------------------------------------------------------------------------

    def start(self) -> None:
        for target, name in ((self._run, "meetvoice-live"), (self._translate_loop, "meetvoice-tr")):
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)

    def stop(self, flush: bool = False, timeout: float = 5.0) -> None:
        """flush=True also recognizes what is left in the buffers."""
        self._flush = flush
        self._stop.set()
        for t in self._threads:
            t.join(timeout)

    def since(self, index: int) -> list[dict]:
        """Entries from index on, each with its index `i` and translation `target` ("" = none).
        Poll this; re-poll an entry until its translation arrives."""
        with self._lock:
            return [dict(e, i=i, target=self.target(e)) for i, e in enumerate(self.entries) if i >= index]

    def current_drafts(self) -> list[dict]:
        """Open utterances: [{speaker, text, start, end}]; replace the previous ones wholesale."""
        with self._lock:
            return [dict(d, speaker=k) for k, d in self.drafts.items() if d.get("text")]

    def target(self, entry: dict) -> str:
        if entry.get("unsure") or entry.get("filler"):
            return ""
        lang = entry.get("lang") or ""
        if lang and lang != self.native:
            to = self.translate_to or self.foreign_to
            return "" if to == lang else to
        return self.translate_to

    # -- workers ------------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            stopping = self._stop.wait(POLL_SEC)
            for st in self._states:
                st.pull()
                while (cut := pause_cut(st.buffer, MIN_FINAL_SEC, MAX_BUF_SEC)) is not None:
                    self._final(st, *st.take(cut))
                if stopping and self._flush and st.buffer.size >= 0.5 * SR:
                    self._final(st, *st.take(st.buffer.size))
            if stopping:
                break
            # one draft per step: the most recent buffer first; finals above always ran
            for st in sorted(self._states, key=lambda s: -s.buffer_start):
                if (st.track.mic and not self.mic_drafts) or not st.buffer.size or not speech_regions(st.buffer):
                    continue
                if self._draft(st):
                    break
        with self._lock:
            self.drafts.clear()
        self._save()
        self._tq.put(None)

    def _set_draft(self, st: _State, text: str) -> None:
        with self._lock:
            self.drafts[st.track.speaker] = {"text": text, "start": round(st.buffer_start, 2),
                                             "end": round(st.buffer_start + st.buffer.size / SR, 2)}

    def _draft(self, st: _State) -> bool:
        if self.uz_finder and st.buffer_start - st.last_uz < UZ_STICKY_SEC:
            self._set_draft(st, "...")  # Whisper drafts of Uzbek are noise: show that someone talks
            return False
        if self.drafter is None:
            return False
        try:
            text = self.drafter(st.buffer)
        except Exception:
            log.warning("draft failed", exc_info=True)
            return False
        if text:
            st.draft_text = text
            self._set_draft(st, text)
        return text is not None

    def _echo_free(self, st: _State, start: float, audio: np.ndarray) -> np.ndarray | None:
        call = next((s for s in self._states if not s.track.mic), None)
        if not st.track.mic or call is None:
            return audio
        a, b = max(st.hist_start, call.hist_start), min(st.hist_end, call.hist_end)
        if b - a < ECHO_MIN_SEC:
            return audio
        lag = crosstalk_lag(st.span(a, b), call.span(a, b), min_sec=ECHO_MIN_SEC)
        if lag is None:
            return audio  # headset: the mic does not hear the call
        regions = speech_regions(audio)
        kept = drop_crosstalk(regions, audio, call.span(start, start + audio.size / SR), lag)
        if not kept:
            return None
        if kept == regions:
            return audio
        clean = np.zeros_like(audio)
        for x, y in kept:
            clean[int(x * SR):int(y * SR)] = audio[int(x * SR):int(y * SR)]
        return clean

    def _final(self, st: _State, start: float, audio: np.ndarray) -> None:
        draft, st.draft_text = st.draft_text, None
        with self._lock:
            self.drafts.pop(st.track.speaker, None)
        if audio.size == 0 or float(np.sqrt(np.mean(audio ** 2))) < SILENCE_RMS:
            return
        try:
            audio = self._echo_free(st, start, audio)
        except Exception:
            log.warning("echo check failed", exc_info=True)
        if audio is None:
            return
        sticky = start - st.last_uz < UZ_STICKY_SEC
        try:
            foreign, segments = self.uz_finder(audio, force=sticky) if self.uz_finder else (False, [])
            if segments:
                st.last_uz = start + audio.size / SR
            if foreign and segments and st.track.mic:
                return  # Uzbek on the mic: the room behind the user, not the meeting
            if not segments:
                segments = [s for s in self.transcriber(audio, draft=draft)
                            if not glossary_readback(s["text"], self.glossary) and not SUBTITLES_RE.match(s["text"])]
                self._check_language(audio, segments)
        except Exception as e:
            log.exception("recognition failed")
            self.error = str(e)
            return
        for seg in segments:
            text = seg["text"].strip()
            lang = seg.get("lang", "")
            if not text or (st.track.mic and lang not in MIC_LANGS | {"uz"}):
                continue
            if sticky and lang not in MIC_LANGS | {"uz"}:
                continue  # Uzbek the finder missed, which Whisper "heard" as Polish or Turkish
            unsure = list(seg.get("unsure") or [])
            key, filler = norm(text), is_filler(text)
            if (not any(c.isalpha() for c in text) or is_noise_text(text)
                    or ("loop" in unsure and len(unsure) > 1)
                    or (not filler and any(k == key and start + seg["start"] - e < REPEAT_SEC
                                           for e, k in st.recent))):
                continue
            st.recent = (st.recent + [(start + seg["end"], key)])[-5:]
            entry = {"start": round(start + seg["start"], 2), "end": round(start + seg["end"], 2),
                     "speaker": st.track.speaker, "text": text, "lang": lang}
            if unsure:
                entry["unsure"] = unsure
            if filler:
                entry["filler"] = True
            with self._lock:
                self.entries.append(entry)
                index = len(self.entries) - 1
            if self.translator and self.target(entry):
                self._tq.put(index)
        self._save()

    def _check_language(self, audio: np.ndarray, segments: list[dict]) -> None:
        foreign = [s for s in segments if s.get("lang", "") not in (self.native, "")]
        if not foreign:
            return
        probs = self.lang_probs(audio[:30 * SR]) if self.lang_probs else {}
        for s in foreign:
            why = s.setdefault("unsure", [])
            if s["lang"] not in EXPECTED_LANGS and "language" not in why:
                why.append("language")
            if self.lang_probs and probs.get(s["lang"], 0.0) < LANG_SURE:
                why.append("lang_weak")

    def _translate_loop(self) -> None:
        while (index := self._tq.get()) is not None:
            with self._lock:
                entry = dict(self.entries[index])
                context = "\n".join(e["text"] for e in self.entries[max(0, index - CONTEXT_LINES):index])
            target = self.target(entry)
            if not target:
                continue
            source = entry["lang"] if entry.get("lang") not in ("", self.native) else ""
            try:
                tr = self.translator(entry["text"], target, context=context, source_lang=source)
            except Exception as e:
                log.warning("translation failed: %s", e)
                self.error = f"translation: {e}"
                continue
            with self._lock:
                self.entries[index]["translation"] = tr
                self.entries[index]["translated_to"] = target
            self._save()
        self._save()

    def _save(self) -> None:
        if not self.out_path:
            return
        with self._lock:
            lines = [json.dumps(e, ensure_ascii=False) for e in self.entries]
        try:
            self.out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        except OSError:
            pass

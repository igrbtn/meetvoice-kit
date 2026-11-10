"""Live transcription and translation of growing WAV tracks.

Every POLL_SEC each track's new audio is cut at the quietest point near the end of the
window (words are not split), recognized and appended to `entries`. Translation runs on
its own thread so a slow LLM never delays the text.

Routing of a window:
  1. second-language finder (Uzbek) first: if the window is not Russian/English by ear and
     decodes as Uzbek, those lines win; after Uzbek was heard on a track the finder is
     forced for UZ_STICKY_SEC (short Uzbek windows full of loanwords pass for Russian);
  2. otherwise Whisper; its detected language is kept per line;
  3. foreign lines (anything but `native`) are translated to `foreign_to` always, other
     lines only when `translate_to` is set.
Mic track specifics: the call coming back through speakers is cut by envelope correlation
with the call track (audio.drop_crosstalk); Uzbek on the mic is people in the room and is
dropped; on the mic only MIC_LANGS survive (the rest is echo garbage).
"""

import json
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import SR, WavTail, crosstalk_lag, drop_crosstalk, quiet_cut, resample, speech_regions

log = logging.getLogger("meetvoice.live")

MIN_WINDOW_SEC = 6.0      # do not recognize shorter pieces (except on flush)
MAX_WINDOW_SEC = 20.0     # cut even without a pause
MIN_CUT_SEC = 3.0
SILENCE_RMS = 0.004       # whole window below this = nothing said
POLL_SEC = 1.0
UZ_STICKY_SEC = 90.0
CONTEXT_LINES = 4         # earlier lines given to the translator
HIST_SEC = 60.0           # audio kept per track for the echo check
ECHO_MIN_SEC = 5.0        # enough of both tracks to find the echo lag
MIC_LANGS = {"ru", "en", ""}


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

    def take_window(self, flush: bool = False) -> tuple[float, np.ndarray] | None:
        dur = self.buffer.size / SR
        if dur < (0.5 if flush else MIN_WINDOW_SEC):
            return None
        if flush:
            cut = self.buffer.size
        elif dur >= MAX_WINDOW_SEC:
            cut = quiet_cut(self.buffer[:int(MAX_WINDOW_SEC * SR)], MIN_CUT_SEC)
        else:
            cut = quiet_cut(self.buffer, MIN_CUT_SEC)
            if cut < MIN_CUT_SEC * SR:
                return None
        start, window = self.buffer_start, self.buffer[:cut]
        self.buffer = self.buffer[cut:]
        self.buffer_start += cut / SR
        return start, window


class LiveTranscriber:
    """
    transcriber(audio) -> [{start, end, text, lang}]           e.g. asr.transcribe_window
    uz_finder(audio, force) -> (foreign: bool, [segments])     e.g. partial(uzbek.live_find, path); or None
    translator(text, target, context=, source_lang=) -> str    e.g. translate.Translator(...)
    """

    def __init__(self, tracks: list[Track], transcriber, translator=None, uz_finder=None,
                 translate_to: str = "", foreign_to: str = "ru", native: str = "ru",
                 out_path: Path | None = None):
        self.transcriber, self.translator, self.uz_finder = transcriber, translator, uz_finder
        self.translate_to, self.foreign_to, self.native = translate_to, foreign_to, native
        self.out_path = Path(out_path) if out_path else None
        self.entries: list[dict] = []
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

    def target(self, entry: dict) -> str:
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
                while (item := st.take_window(flush=stopping and self._flush)) is not None:
                    self._recognize(st, *item)
            if stopping:
                break
        self._save()
        self._tq.put(None)

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

    def _recognize(self, st: _State, start: float, audio: np.ndarray) -> None:
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
                segments = self.transcriber(audio)
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
            entry = {"start": round(start + seg["start"], 2), "end": round(start + seg["end"], 2),
                     "speaker": st.track.speaker, "text": text, "lang": lang}
            with self._lock:
                self.entries.append(entry)
                index = len(self.entries) - 1
            if self.translator and self.target(entry):
                self._tq.put(index)
        self._save()

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

"""Prototype: streaming Whisper with a stable prefix (LocalAgreement-2), glossary prompt and
phrase-level translation. Replays a recording in real time and measures what a viewer
would see.

    python proto/stream_proto.py call.flac --start 300 --dur 90 --glossary "IPsec, ADFS" \
        --out log.json [--baseline]

Per pass the whole uncommitted buffer (<= MAX_BUF s) is re-decoded with word timestamps.
Words on which two consecutive passes agree are committed (never change again); the rest
is a draft. A phrase is closed on sentence punctuation or a pause and goes to translation
right away. --baseline also runs the current window engine (meetvoice.live) on the same
audio for comparison.
"""

import argparse
import json
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

import mlx.core as mx
import mlx_whisper
import numpy as np

from meetvoice import asr
from meetvoice.audio import SR, speech_regions
from meetvoice.translate import Translator, system_prompt

MODEL = asr.DEFAULT_MODEL
STEP = 1.0           # seconds of new audio between passes
MAX_BUF = 15.0       # longer buffers cost ~1 s a pass on an M5 Pro
PROMPT_CHARS = 200   # committed text carried into the prompt
PAUSE_END = 0.8      # gap after a committed word that closes a phrase
TAIL_SILENT = 1.2    # no speech for this long after the last committed word: flush the draft
DEBUG = False
WORD_TS = False      # Whisper word timestamps double the pass cost; segment times are enough


def load(path, start, dur):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(dur), "-i", str(path),
                          "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").copy()


def norm(w):
    return re.sub(r"[^\w']", "", w.lower())


class StreamingASR:
    def __init__(self, glossary=""):
        self.glossary = glossary
        self.buf = np.zeros(0, np.float32)
        self.buf_start = 0.0
        self.committed = []          # [{w, start, end, p, lang}]
        self.prev = []               # last pass's words after the committed point
        self.passes = []             # decode seconds
        self.lang = None

    def insert(self, x):
        self.buf = np.concatenate([self.buf, x])

    def _prompt(self):
        ctx = " ".join(w["w"] for w in self.committed)[-PROMPT_CHARS:]
        parts = []
        if self.glossary:
            parts.append("Термины: " + self.glossary + ".")
        if ctx:
            parts.append(ctx)
        return " ".join(parts) or None

    def process(self):
        """One pass. Returns (newly committed words, draft words)."""
        if self.buf.size < SR or not speech_regions(self.buf):
            return [], []
        t0 = time.time()
        with asr.LOCK:
            res = mlx_whisper.transcribe(
                self.buf, path_or_hf_repo=MODEL, word_timestamps=WORD_TS, initial_prompt=self._prompt(),
                condition_on_previous_text=False, temperature=0.0, verbose=None, language=None,
                no_speech_threshold=0.5, logprob_threshold=-1.0, compression_ratio_threshold=2.2)
        self.passes.append(time.time() - t0)
        if DEBUG:
            print(f"pass {len(self.passes):3d} buf {self.buf.size / SR:5.1f}s took {self.passes[-1]:.2f}s "
                  f"committed {len(self.committed)} draft {len(self.prev)}", flush=True)
        lang = res.get("language") or ""
        words = []
        for seg in res.get("segments", []):
            text = seg.get("text", "")
            if asr.HALLUCINATION_RE.search(text) or asr.FAR_SCRIPT_RE.search(text):
                continue
            if WORD_TS:
                ws = [(w["word"], w["start"], w["end"]) for w in seg.get("words", [])]
            else:  # spread the segment's time over its words by length
                toks = text.split()
                total = sum(len(t) for t in toks) or 1
                t, ws = seg["start"], []
                for tok in toks:
                    d = (seg["end"] - seg["start"]) * len(tok) / total
                    ws.append((tok, t, t + d))
                    t += d
            for i, (w, a, b) in enumerate(ws):
                if norm(w):
                    words.append({"w": w.strip(), "start": self.buf_start + a, "end": self.buf_start + b,
                                  "lang": lang, "seg_last": i == len(ws) - 1})
        # drop what is already committed inside the buffer: by count, checked word by word
        done = [norm(w["w"]) for w in self.committed if w["end"] > self.buf_start + 0.05]
        k = 0
        while k < min(len(done), len(words)) and norm(words[k]["w"]) == done[k]:
            k += 1
        hyp = words[len(done):] if k < len(done) and len(words) >= len(done) else words[k:]
        # LocalAgreement-2: the longest common prefix of this and the previous hypothesis
        n = 0
        while n < min(len(hyp), len(self.prev)) and norm(hyp[n]["w"]) == norm(self.prev[n]["w"]):
            n += 1
        new = hyp[:n]
        self.committed += new
        self.prev = hyp[n:]
        self._trim()
        return new, self.prev

    def flush(self):
        """End of speech: the draft becomes final."""
        new, self.prev = self.prev, []
        self.committed += new
        self._trim(force=True)
        return new

    def _trim(self, force=False):
        """Drop audio up to the end of the last committed sentence (or everything committed
        when the buffer is long), so passes stay short."""
        if not self.committed:
            return
        cut = None
        for a, b in zip(reversed(self.committed[:-1]), reversed(self.committed[1:]), strict=True):
            if a["end"] <= self.buf_start:
                break
            if a.get("seg_last") or re.search(r"[.?!]$", a["w"]) or b["start"] - a["end"] > 0.5:
                cut = a["end"]
                break
        if force or (cut is None and self.buf.size / SR > MAX_BUF - 5):
            cut = self.committed[-1]["end"]
        if cut is not None and cut > self.buf_start:
            k = int((cut - self.buf_start) * SR)
            self.buf, self.buf_start = self.buf[k:], cut
        if self.buf.size > MAX_BUF * SR:
            # nothing agreed for a whole buffer: take the draft as final and start over
            self.committed += self.prev
            self.prev = []
            k = self.buf.size - int((MAX_BUF - 5) * SR)
            self.buf, self.buf_start = self.buf[k:], self.buf_start + k / SR


def run_stream(audio, glossary, translator, foreign_to="ru", native="ru"):
    sa = StreamingASR(glossary)
    events = []                      # (wall, kind, payload)
    tq = queue.Queue()
    t0 = time.time()

    def wall():
        return time.time() - t0

    def tr_loop():
        context = []
        while (item := tq.get()) is not None:
            text, lang, end = item
            if lang in ("", native):
                continue
            prompt_extra = f" Термины (не переводить, писать так): {glossary}." if glossary else ""
            try:
                t = translator._post("/api/chat", {
                    "model": translator.model, "stream": False, "think": False, "options": {"temperature": 0.2},
                    "messages": [{"role": "system", "content": system_prompt(foreign_to, "\n".join(context[-4:]),
                                                                              lang) + prompt_extra},
                                 {"role": "user", "content": text}]})["message"]["content"].strip()
            except Exception as e:
                t = f"<error {e}>"
            context.append(text)
            events.append((wall(), "tr", {"text": text, "tr": t, "end": end}))

    th = threading.Thread(target=tr_loop, daemon=True)
    th.start()
    phrase = []

    def close_phrase():
        nonlocal phrase
        if phrase:
            text = " ".join(w["w"] for w in phrase)
            events.append((wall(), "phrase", {"text": text, "end": phrase[-1]["end"], "lang": phrase[-1]["lang"]}))
            tq.put((text, phrase[-1]["lang"], phrase[-1]["end"]))
            phrase = []

    fed = 0
    total = audio.size
    while fed < total:
        target = min(total, int(wall() * SR))
        step = max(STEP, sa.passes[-1] if sa.passes else 0)
        if target - fed < step * SR and target < total:
            time.sleep(0.02)
            continue
        sa.insert(audio[fed:target])
        fed = target
        new, draft = sa.process()
        now = wall()
        events.append((now, "draft", {"words": [w["w"] for w in draft], "start": draft[0]["start"] if draft else None,
                                      "ends": [w["end"] for w in draft],
                                      "display": [norm(w["w"]) for w in sa.committed + draft]}))
        for w in new:
            events.append((now, "commit", w))
            if phrase and w["start"] - phrase[-1]["end"] > PAUSE_END:
                close_phrase()
            phrase.append(w)
            if re.search(r"[.?!]$", w["w"]):
                close_phrase()
        # silence after the last committed word: the draft is final
        last = sa.committed[-1]["end"] if sa.committed else 0.0
        tail = audio[int(last * SR):fed]
        if draft and tail.size > TAIL_SILENT * SR and not speech_regions(tail[-int(TAIL_SILENT * SR):]):
            for w in sa.flush():
                events.append((wall(), "commit", w))
                phrase.append(w)
            close_phrase()
    for w in sa.flush():
        events.append((wall(), "commit", w))
        phrase.append(w)
    close_phrase()
    tq.put(None)
    th.join(90)
    return events, sa


def run_baseline(audio, translator, tmp: Path):
    """The current engine: 6-20 s windows cut at pauses (meetvoice.live)."""
    import struct

    from meetvoice.live import LiveTranscriber, Track

    p = tmp / "call.wav"
    fmt = struct.pack("<IHHIIHH", 16, 3, 1, SR, SR * 4, 4, 32)
    p.write_bytes(b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " + fmt + b"data" + struct.pack("<I", 0xFFFFFFFF))
    live = LiveTranscriber([Track(p, "call")], transcriber=asr.transcribe_window, translator=translator)
    t0 = time.time()
    seen, events = set(), []
    live.start()
    fed = 0
    while fed < audio.size or any(e["target"] and not e.get("translation") for e in live.since(0)):
        target = min(audio.size, int((time.time() - t0) * SR))
        if target > fed:
            with open(p, "ab") as f:
                f.write(audio[fed:target].astype("<f4").tobytes())
            fed = target
        for e in live.since(0):
            for kind in ("line", "tr"):
                if (e["i"], kind) not in seen and (kind == "line" or e.get("translation")):
                    seen.add((e["i"], kind))
                    events.append((time.time() - t0, kind, {"text": e["text"], "end": e["end"], "lang": e["lang"],
                                                            "tr": e.get("translation", "")}))
        if time.time() - t0 > audio.size / SR + 60:
            break
        time.sleep(0.1)
    live.stop(flush=True)
    return events


def summarize(events, sa, audio_sec):
    commits = [(t, p) for t, k, p in events if k == "commit"]
    lat = [t - p["end"] for t, p in commits]
    # first time each committed word showed up anywhere on screen (draft or commit)
    first = []
    drafts = [(t, p) for t, k, p in events if k == "draft"]
    for t, w in commits:
        seen = [dt for dt, d in drafts if any(abs(e - w["end"]) < 0.25 for e in d["ends"])]
        first.append(min(seen + [t]) - w["end"])
    # flicker: words already on screen that a later pass replaced or removed
    flick, prev = 0, []
    for _, d in drafts:
        cur = d["display"]
        n = 0
        while n < min(len(prev), len(cur)) and prev[n] == cur[n]:
            n += 1
        flick += len(prev) - n
        prev = cur
    trs = [(t, p) for t, k, p in events if k == "tr"]
    tr_lat = [t - p["end"] for t, p in trs]

    def q(v, p):
        return float(np.percentile(v, p)) if v else float("nan")

    return {
        "audio_sec": round(audio_sec, 1), "passes": len(sa.passes),
        "decode_mean_s": round(float(np.mean(sa.passes)), 3) if sa.passes else None,
        "decode_p95_s": round(q(sa.passes, 95), 3), "gpu_busy_share": round(sum(sa.passes) / audio_sec, 2),
        "words": len(commits),
        "first_seen_median_s": round(q(first, 50), 2), "first_seen_p90_s": round(q(first, 90), 2),
        "commit_median_s": round(q(lat, 50), 2), "commit_p90_s": round(q(lat, 90), 2),
        "flicker_per_100_words": round(100 * flick / max(1, len(commits)), 1),
        "translations": len(trs), "tr_median_s": round(q(tr_lat, 50), 2), "tr_p90_s": round(q(tr_lat, 90), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=90)
    ap.add_argument("--glossary", default="")
    ap.add_argument("--llm-model", default="gemma4:12b-mxfp8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()

    audio = load(args.audio, args.start, args.dur)
    translator = Translator(model=args.llm_model)
    translator("salom", "ru")                          # warm the LLM
    with asr.LOCK:                                     # warm Whisper
        mlx_whisper.transcribe(np.zeros(SR, np.float32), path_or_hf_repo=MODEL, verbose=None)
    mx.clear_cache()
    out = {}
    events, sa = run_stream(audio, args.glossary, translator)
    out["stream"] = summarize(events, sa, audio.size / SR)
    out["stream_text"] = [p["text"] for t, k, p in events if k == "phrase"]
    out["stream_tr"] = [[round(t - p["end"], 1), p["text"], p["tr"]] for t, k, p in events if k == "tr"]
    if args.baseline:
        tmp = Path(args.out).parent / "baseline"
        tmp.mkdir(parents=True, exist_ok=True)
        b = run_baseline(audio, translator, tmp)
        lines = [(t, p) for t, k, p in b if k == "line"]
        trs = [(t, p) for t, k, p in b if k == "tr"]
        out["baseline"] = {
            "lines": len(lines),
            "line_median_s": round(float(np.median([t - p["end"] for t, p in lines])), 2) if lines else None,
            "line_p90_s": round(float(np.percentile([t - p["end"] for t, p in lines], 90)), 2) if lines else None,
            "tr_median_s": round(float(np.median([t - p["end"] for t, p in trs])), 2) if trs else None,
        }
        out["baseline_text"] = [p["text"] for t, p in lines]
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k in ("stream", "baseline")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

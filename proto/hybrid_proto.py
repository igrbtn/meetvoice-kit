"""Prototype B: draft + final, no stitching.

Every STEP s the open buffer (audio since the last final) is decoded once without word
timestamps and shown as a gray draft that is replaced wholesale. When the speaker pauses
(PAUSE s of silence after >= MIN_FINAL s of audio) or the buffer reaches MAX_BUF, the buffer
up to the pause is decoded by the regular window engine (asr.transcribe_window, filters
included) as the final line and sent to translation. Finals are never edited, drafts are
never stitched, so there are no duplicated or lost words between passes.

    python proto/hybrid_proto.py call.flac --start 600 --dur 90 --glossary "..." --out log.json
"""

import argparse
import json
import queue
import re
import threading
import time
from pathlib import Path

import mlx_whisper
import numpy as np

from meetvoice import asr, uzbek
from meetvoice.audio import SR, speech_regions
from meetvoice.translate import Translator, system_prompt

STEP = 1.0
PAUSE = 0.6          # silence that ends an utterance
MIN_FINAL = 2.0
MAX_BUF = 8.0
UZ_STICKY_SEC = 90.0
EXPECTED_LANGS = {"ru", "en", "uz"}
SILENCE_RMS = 0.004


def load(path, start, dur):
    import subprocess
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(dur), "-i", str(path), "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").copy()


def pause_cut(buf) -> int | None:
    """Where to close an utterance: a trailing pause, shorter the longer the buffer runs
    (fluent speakers rarely pause 0.6 s), or the quietest point once it reaches MAX_BUF."""
    dur = buf.size / SR
    if dur < MIN_FINAL:
        return None
    frame = SR // 50                                   # 20 ms
    n = buf.size // frame
    db = 20 * np.log10(np.sqrt(np.mean(buf[:n * frame].reshape(n, frame) ** 2, axis=1)) + 1e-9)
    quiet = db < max(np.percentile(db, 15) + 6, -55)
    need = 0.6 if dur < 5 else 0.35 if dur < 8 else 0.25
    run = 0
    for i in range(n - 1, -1, -1):                     # trailing silence length
        if not quiet[i]:
            break
        run += 1
    if run * 0.02 >= need and (n - run) * 0.02 >= MIN_FINAL:
        return (n - run) * frame + int(0.1 * SR)
    if dur >= MAX_BUF:
        from meetvoice.audio import quiet_cut
        return quiet_cut(buf, min_cut_sec=MAX_BUF * 0.5)
    return None


def draft_pass(buf, prompt):
    """None when the GPU is busy with a final or a batch job: a draft is never worth waiting for."""
    if not asr.LOCK.acquire(blocking=False):
        return None
    try:
        res = mlx_whisper.transcribe(buf, path_or_hf_repo=asr.DEFAULT_MODEL, initial_prompt=prompt, verbose=None,
                                     condition_on_previous_text=False, temperature=0.0, word_timestamps=False,
                                     no_speech_threshold=0.5, logprob_threshold=-1.0, compression_ratio_threshold=2.2)
    finally:
        asr.LOCK.release()
    text = " ".join(s["text"].strip() for s in res.get("segments", [])
                    if not asr.HALLUCINATION_RE.search(s["text"]) and not asr.FAR_SCRIPT_RE.search(s["text"]))
    return asr.collapse_repeats(text)


# a final line is "unsure" (shown marked, translated with a caveat) on any of these
UNSURE_LOGPROB = -0.8
UNSURE_NOSPEECH = 0.4
UNSURE_COMPRESSION = 2.0
MAX_WORDS_PER_SEC = 5.0
DRAFT_DISAGREE = 0.5   # share of the final's words missing from the last draft of that audio


def _unsure_reasons(seg, text, draft) -> list[str]:
    why = []
    if seg.get("avg_logprob", 0) < UNSURE_LOGPROB:
        why.append("logprob")
    if seg.get("no_speech_prob", 0) > UNSURE_NOSPEECH:
        why.append("no_speech")
    if seg.get("compression_ratio", 0) > UNSURE_COMPRESSION:
        why.append("loop")
    dur = max(seg["end"] - seg["start"], 0.3)
    if len(text.split()) / dur > MAX_WORDS_PER_SEC:
        why.append("too_fast")
    if why and draft is not None:  # alone it fires on good lines too: only reinforces a Whisper signal
        words = [asr.norm(w) for w in text.split() if asr.norm(w)]
        seen = set(asr.norm(draft).split())
        if len(words) >= 4 and sum(w not in seen for w in words) / len(words) > DRAFT_DISAGREE:
            why.append("draft_disagrees")
    return why


def glossary_readback(text: str, glossary: str) -> bool:
    """On noise Whisper reads the prompt back: a line made mostly of glossary terms."""
    terms = {asr.norm(t) for t in re.split(r"[,;\n]", glossary) if asr.norm(t)}
    words = {asr.norm(w) for w in re.split(r"[,;]", text) if asr.norm(w)}
    return len(words) >= 3 and sum(w in terms for w in words) / len(words) >= 0.5


def final_pass(buf, prompt, draft=None):
    """The window engine with the glossary prompt; each line carries `unsure` reasons."""
    with asr.LOCK:
        res = mlx_whisper.transcribe(buf, path_or_hf_repo=asr.DEFAULT_MODEL, initial_prompt=prompt, verbose=None,
                                     condition_on_previous_text=False, temperature=0.0,
                                     compression_ratio_threshold=2.2, logprob_threshold=-1.0, no_speech_threshold=0.5)
    out = []
    for seg in res.get("segments", []):
        text = asr.collapse_repeats(seg.get("text", "").strip())
        if (not text or asr.HALLUCINATION_RE.search(text) or asr.CAPTION_RE.match(text)
                or asr.norm(text) in asr.LONE_STOCK or asr.FAR_SCRIPT_RE.search(text)
                or seg["start"] >= buf.size / SR):
            continue
        out.append({"start": seg["start"], "end": min(seg["end"], buf.size / SR), "text": text,
                    "lang": res.get("language") or "", "unsure": _unsure_reasons(seg, text, draft)})
    return asr.clean_segments(out)


def run(audio, glossary, translator, native="ru", foreign_to="ru", uz_path=None):
    prompt = ("Термины: " + glossary + ".") if glossary else None
    events, tq = [], queue.Queue()
    t0 = time.time()

    def wall():
        return time.time() - t0

    def tr_loop():
        ctx = []
        while (item := tq.get()) is not None:
            text, lang, end = item
            if lang in ("", native):
                continue
            extra = f" Термины (не переводить, писать так): {glossary}." if glossary else ""
            try:
                tr = translator._post("/api/chat", {
                    "model": translator.model, "stream": False, "think": False,
                    "options": {"temperature": 0.2, "num_ctx": 4096},
                    "messages": [{"role": "system", "content": system_prompt(foreign_to, "\n".join(ctx[-4:]), lang)
                                  + extra}, {"role": "user", "content": text}]})["message"]["content"].strip()
            except Exception as e:
                tr = f"<error {e}>"
            ctx.append(text)
            events.append((wall(), "tr", {"text": text, "tr": tr, "end": end}))

    th = threading.Thread(target=tr_loop, daemon=True)
    th.start()
    buf, buf_start, fed = np.zeros(0, np.float32), 0.0, 0
    passes = {"draft": [], "final": []}
    last_uz = -1e9
    draft_text = None
    while True:
        target = min(audio.size, int(wall() * SR))
        last = target >= audio.size
        if target - fed < STEP * SR and not last:
            time.sleep(0.02)
            continue
        buf = np.concatenate([buf, audio[fed:target]])
        fed = target
        cut = pause_cut(buf)
        if last:
            cut = buf.size
        sticky = buf_start - last_uz < UZ_STICKY_SEC
        if cut:
            piece = buf[:cut]
            if float(np.sqrt(np.mean(piece ** 2))) >= SILENCE_RMS:
                t = time.time()
                lines = []
                if uz_path is not None:
                    _, lines = uzbek.live_find(uz_path, piece, force=sticky)
                    for ln in lines:
                        ln["unsure"] = []
                    if lines:
                        last_uz = buf_start + piece.size / SR
                if not lines:
                    lines = [ln for ln in final_pass(piece, prompt, draft_text)
                             if not glossary_readback(ln["text"], glossary)]
                    for ln in lines:  # a language nobody in the meeting speaks: garbage until proven otherwise
                        if ln["lang"] not in EXPECTED_LANGS and "language" not in ln["unsure"]:
                            ln["unsure"].append("language")
                    if sticky:  # Uzbek the finder missed comes out as Polish/Turkish noise
                        lines = [ln for ln in lines if ln["lang"] in ("ru", "en", "")]
                passes["final"].append(time.time() - t)
                for ln in lines:
                    end = buf_start + ln["end"]
                    events.append((wall(), "final", {"text": ln["text"], "start": buf_start + ln["start"], "end": end,
                                                     "lang": ln["lang"], "unsure": ln["unsure"]}))
                    tq.put((ln["text"], ln["lang"], end))
            buf, buf_start = buf[cut:], buf_start + cut / SR
            draft_text = None
            events.append((wall(), "draft", {"text": "", "upto": buf_start}))
        elif speech_regions(buf):
            if sticky:
                # Whisper drafts of Uzbek are noise: show that someone is talking instead
                events.append((wall(), "draft", {"text": "...", "upto": buf_start + buf.size / SR, "from": buf_start}))
            else:
                t = time.time()
                text = draft_pass(buf, prompt)
                if text is None:
                    passes["skipped"] = passes.get("skipped", 0) + 1
                else:
                    passes["draft"].append(time.time() - t)
                    draft_text = text
                    events.append((wall(), "draft", {"text": text, "upto": buf_start + buf.size / SR,
                                                     "from": buf_start}))
        if last:
            break
    tq.put(None)
    th.join(120)
    return events, passes


def summarize(events, passes, audio_sec):
    finals = [(t, p) for t, k, p in events if k == "final"]
    drafts = [(t, p) for t, k, p in events if k == "draft" and p["text"]]
    trs = [(t, p) for t, k, p in events if k == "tr"]
    # something on screen for a line: the first draft covering its first second, or the final itself
    shown = []
    for t, f in finals:
        covering = [dt for dt, d in drafts if d["from"] <= f["start"] + 0.5 and d["upto"] >= f["start"] + 1.0]
        shown.append(min(covering + [t]) - min(f["start"] + 1.0, f["end"]))

    def q(v, p):
        return round(float(np.percentile(v, p)), 2) if v else None

    allp = passes["draft"] + passes["final"]
    return {
        "audio_sec": round(audio_sec, 1), "draft_passes": len(passes["draft"]), "final_passes": len(passes["final"]),
        "draft_pass_mean_s": q(passes["draft"], 50), "final_pass_mean_s": q(passes["final"], 50),
        "gpu_busy_share": round(sum(allp) / audio_sec, 2), "finals": len(finals),
        "first_seen_median_s": q(shown, 50), "first_seen_p90_s": q(shown, 90),
        "final_median_s": q([t - f["end"] for t, f in finals], 50),
        "final_p90_s": q([t - f["end"] for t, f in finals], 90),
        "unsure": sum(1 for _, f in finals if f["unsure"]), "skipped_drafts": passes.get("skipped", 0),
        "langs": {lang: sum(1 for _, f in finals if f["lang"] == lang) for lang in {f["lang"] for _, f in finals}},
        "translations": len(trs), "tr_median_s": q([t - p["end"] for t, p in trs], 50),
        "tr_p90_s": q([t - p["end"] for t, p in trs], 90),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=90)
    ap.add_argument("--glossary", default="")
    ap.add_argument("--native", default="ru")
    ap.add_argument("--uz-models", default="", help="models dir with the converted Uzbek model; empty = off")
    ap.add_argument("--llm-model", default="gemma4:12b-mxfp8")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    audio = load(args.audio, args.start, args.dur)
    translator = Translator(model=args.llm_model)
    translator("salom", "ru")
    with asr.LOCK:
        mlx_whisper.transcribe(np.zeros(SR, np.float32), path_or_hf_repo=asr.DEFAULT_MODEL, verbose=None)
    uz_path = uzbek.model_path(Path(args.uz_models)) if args.uz_models else None
    if uz_path is not None:  # load the weights now, not inside the first final
        uzbek._decode(uz_path, np.zeros(SR, np.float32), [0.0, 1.0], 0.0)
    events, passes = run(audio, args.glossary, translator, native=args.native, uz_path=uz_path)
    out = {"hybrid": summarize(events, passes, audio.size / SR),
           "final_text": [p["text"] for t, k, p in events if k == "final"],
           "unsure_lines": [[p["unsure"], p["lang"], p["text"]] for t, k, p in events if k == "final" and p["unsure"]],
           "translations": [[round(t - p["end"], 1), p["text"], p["tr"]] for t, k, p in events if k == "tr"]}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["hybrid"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

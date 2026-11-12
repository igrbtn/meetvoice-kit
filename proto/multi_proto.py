"""Prototype C: draft + final on two tracks at once (call + mic), as in a real meeting.

One loop serves both tracks on one GPU. Finals always run; drafts are rationed to one per
step, given to the track that spoke most recently and was not drafted last time, so two
active tracks do not double the GPU load. Mic finals lose the call's echo first
(audio.crosstalk_lag + drop_crosstalk over the last 30 s of both tracks); Uzbek on the mic is
the room and is dropped. Every 15 s the MLX memory in use is sampled to see drift.

    python proto/multi_proto.py --call system.flac --mic mic.flac --start 1620 --dur 300 \
        --glossary "..." --uz-models ~/models --llm-model gemma4:12b-mlx --out log.json
"""

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import mlx.core as mx
import mlx_whisper
import numpy as np
from hybrid_proto import (
    EXPECTED_LANGS,
    SILENCE_RMS,
    STEP,
    UZ_STICKY_SEC,
    draft_pass,
    final_pass,
    glossary_readback,
    load,
    pause_cut,
)

from meetvoice import asr, uzbek
from meetvoice.audio import SR, crosstalk_lag, drop_crosstalk, speech_regions
from meetvoice.translate import Translator, system_prompt

ECHO_SEC = 30.0
LANG_SURE = 0.5
SUBTITLES_RE = __import__("re").compile(r"^\W*(субтитр\w*\W*)+$", __import__("re").I)
MIC_DRAFTS = False   # the user knows what they said: the call gets all the drafts
# "угу" comes out of Whisper as "Mm-hmm"/"Uh-huh"/"Gracias": shown dim, never translated
FILLERS = set("""mm hmm mmhmm mhm uh huh uhhuh uhum um hm hmm ah oh okay ok yeah yep yes no gracias
угу ага да нет ну м мм э а ох хм ок окей""".split())


def is_noise_text(text: str) -> bool:
    """"O-O-O-O-O..." and the like: a long line made of a couple of characters."""
    letters = [c for c in text.lower() if c.isalpha()]
    return len(letters) >= 12 and len(set(letters)) <= 3


def is_filler(text: str) -> bool:
    words = asr.norm(text.replace("-", " ")).split()
    return bool(words) and all(w in FILLERS for w in words)


class Lane:
    REPEAT_SEC = 15.0   # the same final again within this: Whisper repeating itself, not the speaker

    def __init__(self, name, audio, mic):
        self.name, self.audio, self.mic = name, audio, mic
        self.buf, self.buf_start = np.zeros(0, np.float32), 0.0
        self.last_uz, self.draft_text, self.last_speech = -1e9, None, -1.0
        self.drafted_last = False
        self.recent = []    # (end, normalized text) of the last finals


def echo_free(piece, start, mic_audio, call_audio, fed):
    """The piece of mic audio without the call coming back through the speakers."""
    a = max(0, fed - int(ECHO_SEC * SR))
    lag = crosstalk_lag(mic_audio[a:fed], call_audio[a:fed], min_sec=5.0)
    if lag is None:
        return piece, None
    s = int(start * SR)
    regions = speech_regions(piece)
    kept = drop_crosstalk(regions, piece, call_audio[s:s + piece.size], lag)
    if not kept:
        return None, lag
    clean = np.zeros_like(piece)
    for x, y in kept:
        clean[int(x * SR):int(y * SR)] = piece[int(x * SR):int(y * SR)]
    return clean, lag


def run(call, mic, glossary, translator, uz_path, native="ru", foreign_to="ru"):
    prompt = ("Термины: " + glossary + ".") if glossary else None
    lanes = [Lane("call", call, False)] + ([Lane("mic", mic, True)] if mic is not None else [])
    total = max(len(ln.audio) for ln in lanes)
    events, tq, passes, mem = [], queue.Queue(), {"draft": [], "final": [], "skipped": 0}, []
    t0 = time.time()

    def wall():
        return time.time() - t0

    def tr_loop():
        ctx = []
        while (item := tq.get()) is not None:
            text, lang, end, lane = item
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
            events.append((wall(), "tr", {"text": text, "tr": tr, "end": end, "lane": lane}))

    th = threading.Thread(target=tr_loop, daemon=True)
    th.start()
    fed, next_mem = 0, 0.0
    while True:
        target = min(total, int(wall() * SR))
        last = target >= total
        if target - fed < STEP * SR and not last:
            time.sleep(0.02)
            continue
        for ln in lanes:
            ln.buf = np.concatenate([ln.buf, ln.audio[fed:target]])
            if speech_regions(ln.buf[-int(STEP * SR):]) if ln.buf.size >= STEP * SR else False:
                ln.last_speech = target / SR
        fed = target
        if wall() >= next_mem:
            mem.append((round(wall()), round(mx.get_active_memory() / 1e9, 2)))
            next_mem += 15
        want_draft = []
        for ln in lanes:
            cut = ln.buf.size if last else pause_cut(ln.buf)
            sticky = ln.buf_start - ln.last_uz < UZ_STICKY_SEC
            if cut:
                piece, start = ln.buf[:cut], ln.buf_start
                ln.buf, ln.buf_start, ln.draft_text = ln.buf[cut:], ln.buf_start + cut / SR, None
                events.append((wall(), "draft", {"lane": ln.name, "text": "", "upto": ln.buf_start}))
                if float(np.sqrt(np.mean(piece ** 2))) < SILENCE_RMS:
                    continue
                if ln.mic:
                    piece, lag = echo_free(piece, start, mic, call, fed)
                    if piece is None:
                        events.append((wall(), "echo_dropped", {"start": start}))
                        continue
                t = time.time()
                lines = []
                if uz_path is not None:
                    _, lines = uzbek.live_find(uz_path, piece, force=sticky)
                    for x in lines:
                        x["unsure"] = []
                    if lines:
                        ln.last_uz = start + piece.size / SR
                        if ln.mic:  # Uzbek on the mic: the room behind the user
                            events.append((wall(), "room_dropped", {"text": " ".join(x["text"] for x in lines)}))
                            lines = []
                            passes["final"].append(time.time() - t)
                            continue
                if not lines:
                    lines = [x for x in final_pass(piece, prompt, None)
                             if not glossary_readback(x["text"], glossary) and not SUBTITLES_RE.match(x["text"])]
                    probs = asr.lang_probs(piece) if any(x["lang"] not in (native, "") for x in lines) else {}
                    for x in lines:
                        if x["lang"] not in EXPECTED_LANGS and "language" not in x["unsure"]:
                            x["unsure"].append("language")
                        # a foreign line Whisper is not sure is foreign: "Are you naked?" over Uzbek noise
                        if x["lang"] not in (native, "") and probs.get(x["lang"], 0.0) < LANG_SURE:
                            x["unsure"].append("lang_weak")
                    if sticky or ln.mic:
                        lines = [x for x in lines if x["lang"] in ("ru", "en", "")]
                passes["final"].append(time.time() - t)
                kept = []
                for x in lines:
                    key = asr.norm(x["text"])
                    garbage = (not any(c.isalpha() for c in x["text"])
                               or ("loop" in x["unsure"] and len(x["unsure"]) > 1))
                    if garbage or is_noise_text(x["text"]) or any(k == key and start + x["start"] - e < Lane.REPEAT_SEC
                                                       for e, k in ln.recent if not is_filler(x["text"])):
                        events.append((wall(), "repeat_dropped", {"lane": ln.name, "text": x["text"]}))
                        continue
                    ln.recent = (ln.recent + [(start + x["end"], key)])[-5:]
                    kept.append(x)
                for x in kept:
                    end = start + x["end"]
                    filler = is_filler(x["text"])
                    events.append((wall(), "final", {"lane": ln.name, "text": x["text"], "start": start + x["start"],
                                                     "end": end, "lang": x["lang"], "unsure": x["unsure"],
                                                     "filler": filler}))
                    if not filler and not x["unsure"]:  # garbage translated reads as fact: show it, do not translate
                        tq.put((x["text"], x["lang"], end, ln.name))
            elif speech_regions(ln.buf):
                if sticky:
                    events.append((wall(), "draft", {"lane": ln.name, "text": "...", "from": ln.buf_start,
                                                     "upto": ln.buf_start + ln.buf.size / SR}))
                elif MIC_DRAFTS or not ln.mic:
                    want_draft.append(ln)
        # one draft per step: the most recent speaker, alternating when both talk
        want_draft.sort(key=lambda ln: (ln.drafted_last, -ln.last_speech))
        for i, ln in enumerate(want_draft):
            ln.drafted_last = False
            if i:
                passes["skipped"] += 1
                continue
            t = time.time()
            text = draft_pass(ln.buf, prompt)
            if text is None:
                passes["skipped"] += 1
                continue
            passes["draft"].append(time.time() - t)
            ln.draft_text, ln.drafted_last = text, True
            events.append((wall(), "draft", {"lane": ln.name, "text": text, "from": ln.buf_start,
                                             "upto": ln.buf_start + ln.buf.size / SR}))
        if last:
            break
    tq.put(None)
    th.join(120)
    return events, passes, mem


def summarize(events, passes, audio_sec, lane):
    finals = [(t, p) for t, k, p in events if k == "final" and p["lane"] == lane]
    drafts = [(t, p) for t, k, p in events if k == "draft" and p["lane"] == lane and p["text"]]
    trs = [(t, p) for t, k, p in events if k == "tr" and p["lane"] == lane]
    shown = []
    for t, f in finals:
        cov = [dt for dt, d in drafts if d["from"] <= f["start"] + 0.5 and d["upto"] >= f["start"] + 1.0]
        shown.append(min(cov + [t]) - min(f["start"] + 1.0, f["end"]))

    def q(v, p):
        return round(float(np.percentile(v, p)), 2) if v else None

    return {"finals": len(finals), "fillers": sum(1 for _, f in finals if f.get("filler")),
            "unsure": sum(1 for _, f in finals if f["unsure"] and not f.get("filler")),
            "langs": {g: sum(1 for _, f in finals if f["lang"] == g) for g in {f["lang"] for _, f in finals}},
            "first_seen_median_s": q(shown, 50), "first_seen_p90_s": q(shown, 90),
            "final_median_s": q([t - f["end"] for t, f in finals], 50),
            "final_p90_s": q([t - f["end"] for t, f in finals], 90),
            "translations": len(trs), "tr_median_s": q([t - p["end"] for t, p in trs], 50),
            "tr_p90_s": q([t - p["end"] for t, p in trs], 90)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", required=True)
    ap.add_argument("--mic")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=300)
    ap.add_argument("--glossary", default="")
    ap.add_argument("--uz-models", default="")
    ap.add_argument("--llm-model", default="gemma4:12b-mlx")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    call = load(args.call, args.start, args.dur)
    mic = load(args.mic, args.start, args.dur) if args.mic else None
    if mic is not None:
        n = max(call.size, mic.size)
        call, mic = np.pad(call, (0, n - call.size)), np.pad(mic, (0, n - mic.size))
    translator = Translator(model=args.llm_model)
    translator("salom", "ru")
    with asr.LOCK:
        mlx_whisper.transcribe(np.zeros(SR, np.float32), path_or_hf_repo=asr.DEFAULT_MODEL, verbose=None)
    uz_path = uzbek.model_path(Path(args.uz_models)) if args.uz_models else None
    if uz_path is not None:
        uzbek._decode(uz_path, np.zeros(SR, np.float32), [0.0, 1.0], 0.0)
    events, passes, mem = run(call, mic, args.glossary, translator, uz_path)
    sec = call.size / SR
    allp = passes["draft"] + passes["final"]
    out = {"audio_sec": round(sec), "gpu_busy_share": round(sum(allp) / sec, 2),
           "draft_pass_mean_s": round(float(np.mean(passes["draft"])), 2) if passes["draft"] else None,
           "final_pass_mean_s": round(float(np.mean(passes["final"])), 2) if passes["final"] else None,
           "final_pass_p95_s": round(float(np.percentile(passes["final"], 95)), 2) if passes["final"] else None,
           "skipped_drafts": passes["skipped"], "mlx_mem_gb": mem,
           "echo_dropped": sum(1 for _, k, _ in events if k == "echo_dropped"),
           "room_dropped": [p["text"][:80] for _, k, p in events if k == "room_dropped"],
           "repeat_dropped": [p["text"][:60] for _, k, p in events if k == "repeat_dropped"],
           "lanes": {lane: summarize(events, passes, sec, lane) for lane in ("call", "mic")}}
    full = {**out, "finals": [[p["lane"], round(p["start"], 1), p["lang"], p["unsure"], p["text"]]
                              for t, k, p in events if k == "final"],
            "translations": [[p["lane"], round(t - p["end"], 1), p["text"], p["tr"]]
                             for t, k, p in events if k == "tr"]}
    Path(args.out).write_text(json.dumps(full, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""Replay recorded audio as if it were a live call and watch captions + translations.

    python examples/live_from_wav.py call.wav [--mic mic.wav] [--panel] [--speed 1]

The files are copied into growing WAVs in a temp dir in real time (the same way a capture
tool writes them), so the timing you see is the latency you will get. Open
http://127.0.0.1:8765/ for the overlay page, or pass --panel on macOS for the floating
window. Needs: extras "mlx" (and "macos" for --panel), a local OpenAI-compatible LLM
(Ollama by default) for translation.
"""

import argparse
import struct
import subprocess
import tempfile
import threading
import time
from functools import partial
from pathlib import Path

from meetvoice import asr, server, uzbek
from meetvoice.live import LiveTranscriber, Track
from meetvoice.translate import Translator

RATE = 48000


def pcm16(path: str) -> bytes:
    return subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"],
                          capture_output=True, check=True).stdout


def header() -> bytes:
    # sizes 0xFFFFFFFF: a capture tool finalizes them only on stop
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16) + b"data" + struct.pack("<I", 0xFFFFFFFF))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("call")
    ap.add_argument("--mic")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--llm-url", default="http://127.0.0.1:11434")
    ap.add_argument("--llm-kind", default="ollama", choices=["ollama", "openai"])
    ap.add_argument("--llm-model", default="gemma3:12b")
    ap.add_argument("--models-dir", default=str(Path.home() / ".cache" / "meetvoice"))
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="meetvoice-"))
    sources = {"call.wav": pcm16(args.call)}
    if args.mic:
        sources["mic.wav"] = pcm16(args.mic)
    for name in sources:
        (work / name).write_bytes(header())
    tracks = [Track(work / "call.wav", "Собеседник")]
    if args.mic:
        tracks.append(Track(work / "mic.wav", "Я", mic=True))

    uz_path = uzbek.model_path(Path(args.models_dir))
    translator = Translator(args.llm_url, args.llm_model, kind=args.llm_kind)
    print("warming up the LLM (a cold start would delay the first translations by ~30 s)...")
    translator("salom", "ru")
    live = LiveTranscriber(tracks, transcriber=asr.transcribe_window,
                           uz_finder=partial(uzbek.live_find, uz_path),
                           translator=translator, out_path=work / "live.jsonl")
    live.start()
    server.serve(lambda: live)
    print(f"overlay: http://127.0.0.1:8765/   ({work})")

    def feed():
        step = 0.5
        total = max(len(b) for b in sources.values()) / (RATE * 2)
        for i in range(int(total / step) + 1):
            for name, raw in sources.items():
                with open(work / name, "ab") as f:
                    f.write(raw[int(i * step * RATE * 2):int((i + 1) * step * RATE * 2)])
            time.sleep(step / args.speed)

    def show():
        seen, t0 = set(), time.time()
        while True:
            for e in live.since(0):
                key = (e["i"], bool(e.get("translation")))
                if key not in seen:
                    seen.add(key)
                    tr = f"\n      -> {e['translation']}" if key[1] else ""
                    print(f"[{time.time() - t0:5.1f}s] {e['speaker']} {e['lang']:<2} {e['text']}{tr}", flush=True)
            time.sleep(0.3)

    threading.Thread(target=show, daemon=True).start()
    if args.panel:
        threading.Thread(target=feed, daemon=True).start()
        from meetvoice.macos_overlay import run_standalone
        run_standalone("http://127.0.0.1:8765/")
    else:
        feed()
        deadline = time.time() + 60
        time.sleep(3)
        while time.time() < deadline and any(e["target"] and not e.get("translation") for e in live.since(0)):
            time.sleep(0.5)  # let the last translations land
        time.sleep(1)
        live.stop()


if __name__ == "__main__":
    main()

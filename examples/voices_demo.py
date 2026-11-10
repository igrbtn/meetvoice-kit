"""Voice library round trip: learn named speakers from one meeting, recognize them in another.

    python examples/voices_demo.py learn  m1 --track mic=m1_mic.wav --track call=m1_call.wav --lines m1.json
    python examples/voices_demo.py match  m2 --track mic=m2_mic.wav --track call=m2_call.wav --lines m2.json

lines JSON: [{"start": 1.2, "end": 4.8, "speaker": "Anna"}, ...] - your diarized transcript.
In "learn" give real names; in "match" leave diarization labels ("Speaker 1", "Me") and see
which ones the library names. Library file: ./voices.json. Needs extra "voices" + ffmpeg.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from meetvoice.voices import VoiceLibrary, embedder, speaker_prints


def load(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["learn", "match"])
    ap.add_argument("meeting_id")
    ap.add_argument("--track", action="append", required=True, help="name=path")
    ap.add_argument("--lines", required=True)
    ap.add_argument("--library", default="voices.json")
    ap.add_argument("--models-dir", default=str(Path.home() / ".cache" / "meetvoice"))
    args = ap.parse_args()

    audio = {k: load(v) for k, v in (t.split("=", 1) for t in args.track)}
    segments = json.loads(Path(args.lines).read_text())
    lib = VoiceLibrary(Path(args.library))
    embed = embedder(Path(args.models_dir))
    prints = speaker_prints(audio, segments, embed)
    print("prints:", {k: f"{v[1]:.0f}s" for k, v in prints.items()})
    if args.mode == "learn":
        print("learned:", lib.remember(args.meeting_id, prints))
    else:
        cents = lib.centroids()
        for label, (emb, _) in prints.items():
            print(f"  {label:<14}", {n: round(float(emb @ c), 2) for n, c in cents.items()})
        print("named:", lib.match(prints))


if __name__ == "__main__":
    main()

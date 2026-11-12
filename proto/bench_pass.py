"""Cost of one streaming pass: model variant x word timestamps x buffer length."""
import subprocess
import sys
import time

import mlx_whisper
import numpy as np

SR = 16000
raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "600", "-t", "20", "-i", sys.argv[1], "-ac", "1", "-ar", str(SR),
                      "-f", "f32le", "-"], capture_output=True, check=True).stdout
audio = np.frombuffer(raw, dtype="<f4").copy()
MODELS = ["mlx-community/whisper-large-v3-turbo", "mlx-community/whisper-large-v3-turbo-q4"]
for model in MODELS:
    mlx_whisper.transcribe(audio[:SR], path_or_hf_repo=model, verbose=None)  # load
    for words in (True, False):
        for sec in (5, 10, 15):
            times = []
            for _ in range(3):
                t = time.time()
                r = mlx_whisper.transcribe(audio[:sec * SR], path_or_hf_repo=model, verbose=None, word_timestamps=words,
                                           condition_on_previous_text=False, temperature=0.0, language="ru")
                times.append(time.time() - t)
            print(f"{model.split('/')[-1]:<32} words={words!s:<5} {sec:>2}s  {min(times):.2f}s  "
                  f"{r['text'][:60]!r}", flush=True)

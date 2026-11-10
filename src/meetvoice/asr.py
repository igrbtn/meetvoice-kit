"""Whisper (mlx-whisper, Apple Silicon) for short live windows, with the filters that make
it usable on meeting audio.

mlx-whisper caches exactly one model (ModelHolder). Live mode alternates the main model
and a language-specific one every few seconds, so `use_model` swaps a model we hold
ourselves into that cache for one call instead of reloading weights each time.
All model calls go through LOCK: one GPU, one decoder at a time.
"""

import contextlib
import importlib
import re
import threading

import numpy as np

LOCK = threading.Lock()
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

# Whisper was trained on subtitled video: on silence, noise or a language it does not
# know it "hears" credits. Extend for your languages.
HALLUCINATION_RE = re.compile(
    r"(субтитр\w*\s+(сделал|создавал|подготовил|делал)|редактор субтитров|продолжение следует|"
    r"спасибо за (просмотр|внимание)|подписывайтесь|ставьте лайк|thank(s| you) for watching|"
    r"amara\.org|go to the next (video|one)\b)", re.I)
CAPTION_RE = re.compile(r"^\W*(\w+\s+)?(музыка|аплодисменты|смех|music|applause|laughter)\W*$", re.I)
FAR_SCRIPT_RE = re.compile(r"[֐-׿぀-ヿ㐀-鿿가-힯؀-ۿ฀-๿]")
LONE_STOCK = {"thank you", "thanks"}


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def collapse_repeats(text: str, max_phrase: int = 8, keep: int = 2) -> str:
    """Cut decoder loops: a phrase of 1..max_phrase words repeated more than `keep` times."""
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        for size in range(1, max_phrase + 1):
            chunk = words[i:i + size]
            reps = 1
            while words[i + reps * size:i + (reps + 1) * size] == chunk:
                reps += 1
            if reps > keep:
                out += chunk * keep
                i += reps * size
                break
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def clean_segments(segments: list[dict], max_same: int = 2) -> list[dict]:
    """Drop credit hallucinations and runs of identical lines (a loop spread over windows)."""
    out: list[dict] = []
    run = 0
    for seg in segments:
        if HALLUCINATION_RE.search(seg["text"]) or CAPTION_RE.match(seg["text"]):
            continue
        same = out and norm(seg["text"]) == norm(out[-1]["text"])
        run = run + 1 if same else 1
        if run > max_same:
            continue
        out.append(seg)
    return out


def _holder():
    return importlib.import_module("mlx_whisper.transcribe").ModelHolder


@contextlib.contextmanager
def use_model(path: str, model):
    """Put an already loaded model into mlx-whisper's cache for the duration of a call.
    Caller holds LOCK."""
    holder = _holder()
    saved = holder.model, holder.model_path
    holder.model, holder.model_path = model, path
    try:
        yield
    finally:
        holder.model, holder.model_path = saved


def lang_probs(clip, model_name: str = DEFAULT_MODEL) -> dict:
    """Whisper's language guess for a clip (only the first 30 s are looked at)."""
    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim

    with LOCK:
        model = _holder().get_model(model_name, mx.float16)
        mel = log_mel_spectrogram(np.asarray(clip, np.float32), n_mels=model.dims.n_mels)
        _, probs = model.detect_language(pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16))
    return probs


def transcribe_window(audio: np.ndarray, model_name: str = DEFAULT_MODEL,
                      language: str | None = None) -> list[dict]:
    """A short live window (a few to 20 s, 16 kHz mono) -> [{start, end, text, lang}]."""
    import mlx_whisper

    with LOCK:
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=model_name, language=language, verbose=None,
            condition_on_previous_text=False,   # on: one hallucination poisons the rest
            compression_ratio_threshold=2.2, logprob_threshold=-1.0, no_speech_threshold=0.5,
            temperature=0.0,
        )
    dur = len(audio) / 16000
    out = []
    for seg in result.get("segments", []):
        text = collapse_repeats(seg.get("text", "").strip())
        if (not text or HALLUCINATION_RE.search(text) or CAPTION_RE.match(text)
                or norm(text) in LONE_STOCK or FAR_SCRIPT_RE.search(text)):
            continue
        if seg["start"] >= dur:
            continue  # a loop on a short window runs its timestamps past the audio
        out.append({"start": round(seg["start"], 2), "end": round(min(seg["end"], dur), 2),
                    "text": text, "lang": result.get("language") or ""})
    return clean_segments(out)

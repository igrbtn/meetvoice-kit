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


# a final line is "unsure" (shown marked, not translated) on any of these Whisper signals
UNSURE_LOGPROB = -0.8
UNSURE_NOSPEECH = 0.4
UNSURE_COMPRESSION = 2.0
MAX_WORDS_PER_SEC = 5.0
DRAFT_DISAGREE = 0.5   # share of the final's words missing from the last draft of that audio


def unsure_reasons(seg: dict, text: str, draft: str | None = None) -> list[str]:
    why = []
    if seg.get("avg_logprob", 0) < UNSURE_LOGPROB:
        why.append("logprob")
    if seg.get("no_speech_prob", 0) > UNSURE_NOSPEECH:
        why.append("no_speech")
    if seg.get("compression_ratio", 0) > UNSURE_COMPRESSION:
        why.append("loop")
    if len(text.split()) / max(seg["end"] - seg["start"], 0.3) > MAX_WORDS_PER_SEC:
        why.append("too_fast")
    if why and draft is not None:  # alone it fires on good lines too: only reinforces a Whisper signal
        words = [w for w in (norm(x) for x in text.split()) if w]
        seen = set(norm(draft).split())
        if len(words) >= 4 and sum(w not in seen for w in words) / len(words) > DRAFT_DISAGREE:
            why.append("draft_disagrees")
    return why


def glossary_prompt(glossary: str, max_chars: int = 480) -> str | None:
    """Whisper takes the prompt as "previous text": a plain sentence works best. Only ~223
    tokens fit and the tail is kept, so the list is cut at a term boundary."""
    terms = [t.strip() for t in re.split(r"[,\n;]", glossary) if t.strip()]
    if not terms:
        return None
    text = "Термины: " + terms[0]
    for term in terms[1:]:
        if len(text) + len(term) + 2 > max_chars:
            break
        text += ", " + term
    return text + "."


def transcribe_window(audio: np.ndarray, model_name: str = DEFAULT_MODEL, language: str | None = None,
                      prompt: str | None = None, draft: str | None = None) -> list[dict]:
    """A final pass over a closed utterance (16 kHz mono) -> [{start, end, text, lang, unsure}].
    prompt: glossary_prompt(...); draft: the last draft of the same audio (reinforces `unsure`)."""
    import mlx_whisper

    with LOCK:
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=model_name, language=language, initial_prompt=prompt, verbose=None,
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
                    "text": text, "lang": result.get("language") or "",
                    "unsure": unsure_reasons(seg, text, draft)})
    return clean_segments(out)


def draft_window(audio: np.ndarray, model_name: str = DEFAULT_MODEL, prompt: str | None = None) -> str | None:
    """Quick pass over the open buffer, shown gray until the final replaces it. No word
    timestamps (they double the cost: 0.77 s vs 0.42 s on 15 s). None when the GPU is busy
    with a final or another job: a draft is never worth waiting for."""
    import mlx_whisper

    if not LOCK.acquire(blocking=False):
        return None
    try:
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=model_name, initial_prompt=prompt, verbose=None,
            condition_on_previous_text=False, temperature=0.0, word_timestamps=False,
            no_speech_threshold=0.5, logprob_threshold=-1.0, compression_ratio_threshold=2.2)
    finally:
        LOCK.release()
    text = " ".join(s["text"].strip() for s in result.get("segments", [])
                    if not HALLUCINATION_RE.search(s["text"]) and not FAR_SCRIPT_RE.search(s["text"]))
    return collapse_repeats(text)

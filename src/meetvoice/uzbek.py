"""Uzbek speech: find it in a stream that Whisper takes for something else, decode it with a
fine-tuned Whisper, verify the result is Uzbek.

Why a separate model: Whisper large-v3-turbo never outputs "uz" for Uzbek (it hears en, fa
or ru), loops on it and answers in Persian script. A fine-tuned medium model is published
in the Hugging Face transformers layout; `convert` turns it into the mlx-whisper layout
once, no torch needed.

The same pattern works for any language Whisper handles badly: swap MODEL_REPO, the
language code passed to the decoder and `is_uzbek`.
"""

import json
import logging
import math
import re
import threading
from pathlib import Path

import numpy as np

from . import asr
from .audio import SR, group_regions, speech_regions

log = logging.getLogger(__name__)

MODEL_REPO = "islomov/rubaistt_v2_medium"   # best of 3 compared on a live call, see docs/CALIBRATION.md
MIN_GROUP_SEC = 3.0
PIECE_SEC = 12.0      # the fine-tuned model gives no timestamps inside a window: decode in pieces
RU_MAX = 0.5          # Whisper never says "uz", so the test is "not Russian"...
EN_SURE = 0.6         # ...and not clearly English (the Uzbek model writes English out as Uzbek-looking words)
UZ_MIN_SHARE = 0.2
UZ_MIN_WORDS = 4      # "mhm. xo'p." on keyboard noise is not a talk
UZ_MIN_LOGPROB = -0.9
_CAPTION_RE = re.compile(r"\b(?:musiqa\w*|qarsak\w*|kulgi|olqish\w*)\b|,?\s*music by .*", re.I)
_FILLERS = {"mhm", "ah", "uh", "aha", "ha", "hm", "qah", "e", "a"}
_UZ_WORDS = set("""va bu u bor yo'q yoq kerak edi ekan bilan uchun emas bo'ladi bo'lsa bo'ldi qilib
qiladi qilish menimcha biz bizda sizlar sizdan ularda ularga nima qanday mana xo'p ha albatta
yaxshi rahmat assalomu alaykum endi hozir keyin lekin chunki agar shunaqa o'sha mumkin kerakmi
aka opa tushundim aytib aytdim olib berib""".split())
# o'/g' and plural/genitive endings; -da/-ga/-ni also end Russian words written in Latin, so not used
_UZ_MARK_RE = re.compile(r"(?:o'|g'|oʻ|gʻ|\w(?:lar|larni|larga|larda|ning|dagi|imiz|ingiz)\b)")

_lock = threading.Lock()
_model = None  # (path, loaded mlx model)


# -- model preparation --------------------------------------------------------------

def _hf_to_mlx_name(name: str) -> str | None:
    name = name.removeprefix("model.")
    if name.startswith(("proj_out", "encoder.embed_positions")):
        return None  # tied to the token embedding / sinusoidal in mlx-whisper
    rules = [
        (r"^encoder\.layer_norm\.", "encoder.ln_post."), (r"^decoder\.layer_norm\.", "decoder.ln."),
        (r"^decoder\.embed_tokens\.", "decoder.token_embedding."),
        (r"^decoder\.embed_positions\.weight$", "decoder.positional_embedding"),
        (r"\.layers\.", ".blocks."), (r"\.self_attn_layer_norm\.", ".attn_ln."),
        (r"\.encoder_attn_layer_norm\.", ".cross_attn_ln."), (r"\.final_layer_norm\.", ".mlp_ln."),
        (r"\.self_attn\.", ".attn."), (r"\.encoder_attn\.", ".cross_attn."),
        (r"\.q_proj\.", ".query."), (r"\.k_proj\.", ".key."), (r"\.v_proj\.", ".value."),
        (r"\.out_proj\.", ".out."), (r"\.fc1\.", ".mlp1."), (r"\.fc2\.", ".mlp2."),
    ]
    for pattern, repl in rules:
        name = re.sub(pattern, repl, name)
    return name


def convert(hf_dir: Path, out_dir: Path) -> Path:
    """transformers Whisper (config.json + model*.safetensors) -> mlx-whisper (float16)."""
    import mlx.core as mx

    cfg = json.loads((hf_dir / "config.json").read_text())
    dims = {
        "n_mels": cfg["num_mel_bins"], "n_audio_ctx": cfg["max_source_positions"],
        "n_audio_state": cfg["d_model"], "n_audio_head": cfg["encoder_attention_heads"],
        "n_audio_layer": cfg["encoder_layers"], "n_vocab": cfg["vocab_size"],
        "n_text_ctx": cfg["max_target_positions"], "n_text_state": cfg["d_model"],
        "n_text_head": cfg["decoder_attention_heads"], "n_text_layer": cfg["decoder_layers"],
    }
    weights = {}
    for f in sorted(hf_dir.glob("model*.safetensors")):
        for name, w in mx.load(str(f)).items():
            new = _hf_to_mlx_name(name)
            if new is None:
                continue
            if new.endswith(("conv1.weight", "conv2.weight")):
                w = w.transpose(0, 2, 1)  # torch (out, in, k) -> mlx (out, k, in)
            weights[new] = w.astype(mx.float16)
    if not weights:
        raise RuntimeError(f"{hf_dir}: no model*.safetensors to convert")
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "weights.safetensors"), weights)
    (out_dir / "config.json").write_text(json.dumps({**dims, "model_type": "whisper"}))
    return out_dir


def model_path(models_dir: Path, repo: str = MODEL_REPO) -> Path:
    """mlx-whisper directory of the model, downloaded (config + weights only) and converted once."""
    out = Path(models_dir) / ("uz-" + repo.replace("/", "--"))
    if (out / "weights.safetensors").is_file():
        return out
    with _lock:
        if not (out / "weights.safetensors").is_file():
            from huggingface_hub import snapshot_download

            log.info("preparing %s", repo)
            convert(Path(snapshot_download(repo, allow_patterns=["config.json", "model*.safetensors"])), out)
    return out


# -- text checks -----------------------------------------------------------------------

def tidy(text: str) -> str:
    """Drop caption words, fillers, repeated and cut-off phrases ("xo'p. xo'p. xo'")."""
    out = []
    for part in re.split(r"(?<=[.!?])\s+|\.\s*", _CAPTION_RE.sub("", text)):
        part = part.strip(" .,!?")
        low = part.lower()
        if not part or low in _FILLERS:
            continue
        if out and (low == out[-1].lower() or out[-1].lower().startswith(low)):
            continue
        out.append(part)
    return ". ".join(out)


def is_uzbek(text: str) -> bool:
    """Uzbek (Latin) text rather than Russian or English heard by the Uzbek model."""
    words = re.findall(r"[\w'ʻ]+", text.lower())
    if len(words) < 2:
        return False
    hits = sum(1 for w in words if w in _UZ_WORDS or _UZ_MARK_RE.search(w))
    return hits / len(words) >= UZ_MIN_SHARE


def _looped(text: str, sec: float) -> bool:
    """Few distinct words for the time spoken: the decoder went round in circles."""
    return len(set(re.findall(r"[\w'ʻ]+", text.lower()))) < sec / 3


# -- decoding -------------------------------------------------------------------------

def _pieces(group) -> list[list[tuple[float, float]]]:
    """<= PIECE_SEC parts; a long single region is cut too (past ~30 s the model loops)."""
    cut = []
    for a, b in group:
        n = max(1, math.ceil((b - a) / PIECE_SEC))
        step = (b - a) / n
        cut += [(a + i * step, a + (i + 1) * step) for i in range(n)]
    return group_regions(cut, PIECE_SEC)


def _decode(path: Path, clip, stamps, temperature) -> str:
    import mlx.core as mx
    import mlx_whisper
    from mlx_whisper.load_models import load_model

    global _model
    with asr.LOCK:
        if _model is None or _model[0] != str(path):
            _model = (str(path), load_model(str(path), dtype=mx.float16))
        with asr.use_model(*_model):
            result = mlx_whisper.transcribe(
                clip, path_or_hf_repo=str(path), language="uz", verbose=None, clip_timestamps=stamps,
                condition_on_previous_text=False, temperature=temperature)
    parts = []
    for seg in result.get("segments", []):
        lp = seg.get("avg_logprob", 0.0)
        if lp != lp or lp < UZ_MIN_LOGPROB:  # NaN or unsure
            continue
        text = tidy(asr.collapse_repeats(seg.get("text", "")))
        if text and text not in parts:
            parts.append(text)
    return ". ".join(parts).strip()


def release() -> None:
    """Free the held model (before loading a big local LLM, for example)."""
    global _model
    _model = None


def transcribe_groups(path: Path, audio, groups, sr: int = SR):
    """Decode groups of speech regions; keep those that came out Uzbek.
    Returns (segments with lang="uz", the groups kept)."""
    segs, kept = [], []
    for group in groups:
        found = []
        for piece in _pieces(group):
            base, end = piece[0][0], piece[-1][1]
            clip = np.asarray(audio[int(base * sr):int(end * sr) + 1], np.float32)
            stamps = [t - base for region in piece for t in region]
            text = _decode(path, clip, stamps, 0.0)   # greedy first: temperature only loops longer on noise
            speech = sum(b - a for a, b in piece)
            if _looped(text, speech):
                retry = _decode(path, clip, stamps, (0.2, 0.4))
                if len(retry) > len(text) and not _looped(retry, speech):
                    text = retry
            if text:
                found.append({"start": round(base, 2), "end": round(end, 2), "text": text, "lang": "uz"})
        words = {w for f in found for w in re.findall(r"[\w'ʻ]+", f["text"].lower())} - _FILLERS
        if len(words) >= UZ_MIN_WORDS and is_uzbek(" ".join(f["text"] for f in found)):
            segs += found
            kept.append(group)
    return asr.clean_segments(segs), kept


def live_find(path: Path, audio, force: bool = False, whisper_model: str = asr.DEFAULT_MODEL,
              sr: int = SR) -> tuple[bool, list[dict]]:
    """A live window -> (not Russian/English by ear, its Uzbek lines).

    Lines are empty when the window is not Uzbek after all: give it to Whisper then.
    force skips the "not Russian" test (in a short window Uzbek full of Russian loanwords
    often passes for Russian): use it for a while after Uzbek was heard on the track.
    Clear English is never sent to the Uzbek model, forced or not."""
    if len(audio) < MIN_GROUP_SEC * sr:
        return False, []
    regions = speech_regions(audio)
    if not regions:
        return False, []
    probs = asr.lang_probs(np.asarray(audio[int(regions[0][0] * sr):int(regions[-1][1] * sr)])[:30 * sr],
                           whisper_model)
    if probs.get("en", 0.0) >= EN_SURE or (not force and probs.get("ru", 0.0) >= RU_MAX):
        return False, []
    segs, _ = transcribe_groups(path, audio, [regions])
    return True, segs

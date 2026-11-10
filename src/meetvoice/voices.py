"""Voice library: people named in one meeting are recognized by voice in later ones.

A voice print is the titanet embedding (NeMo titanet_small via sherpa-onnx, CPU) averaged
over a speaker's longest *clean* lines. Prints are stored per meeting, so renaming a
speaker or reprocessing a meeting replaces that meeting's contribution.

Input everywhere: `audio` = {track_name: 16 kHz mono float32}, `segments` = transcript
lines [{start, end, speaker}]. Track names are free-form; a two-track setup (mic + call)
is what the clean-line filter is for, a single mixed track works too.
"""

import json
import logging
import re
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from .audio import SR, level_db

log = logging.getLogger(__name__)

EMBED_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/nemo_en_titanet_small.onnx")   # sic: "recongition" in the release name
EMBED_MAX_SEC = 10.0    # the model fails on minutes-long input; the middle 10 s is enough
MIN_LINE_SEC = 1.5
LINES_PER_PRINT = 12
MIN_PRINT_SEC = 8.0     # less clean speech than this makes a print that matches anyone
LOUD_DB = 10.0          # a print line is within this of its track's talking level...
OTHER_QUIET_DB = 12.0   # ...while every other track is this far below its own
MATCH = 0.62            # same person across meetings 0.64-0.88, different people <= 0.52
MARGIN = 0.08           # over the runner-up: two similar voices name nobody
GENERIC_RE = re.compile(r"^(Я|Me|Собеседник|Спикер|Speaker|Запись|Unknown)(\s+\d+)?$", re.I)


def is_generic(label: str) -> bool:
    """Diarization placeholders that the library may rename."""
    return not label or bool(GENERIC_RE.match(label))


def embedder(models_dir: Path):
    """embed(audio, a, b) -> unit vector; downloads the ~40 MB model on first use."""
    import sherpa_onnx

    path = Path(models_dir) / "nemo_en_titanet_small.onnx"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(EMBED_URL, tmp)
        tmp.rename(path)
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path)))

    def embed(audio, a: float, b: float, sr: int = SR):
        if b - a > EMBED_MAX_SEC:
            mid = (a + b) / 2
            a, b = mid - EMBED_MAX_SEC / 2, mid + EMBED_MAX_SEC / 2
        st = ext.create_stream()
        st.accept_waveform(sr, np.asarray(audio[int(a * sr):int(b * sr)], dtype=np.float32))
        st.input_finished()
        v = np.array(ext.compute(st))
        return v / (np.linalg.norm(v) + 1e-9)

    return embed


def speaker_prints(audio: dict[str, np.ndarray], segments: list[dict], embed) -> dict[str, tuple]:
    """{label: (unit embedding, seconds used)} for labels with enough clean speech.

    A label lives on one track - the one its lines stand out on. Only lines where every
    other track is quiet go in: on double talk the mic also hears the call from the
    speakers and the print becomes a blend of two people (measured: two different people
    scored 0.69 with blended prints, 0.2 with clean ones)."""
    lines = [s for s in segments if s["end"] - s["start"] >= MIN_LINE_SEC]
    if not audio or not lines:
        return {}
    lv = {t: np.array([level_db(x, s["start"], s["end"]) for s in lines]) for t, x in audio.items()}
    speech = {t: float(np.percentile(v, 75)) for t, v in lv.items()}  # the track's talking level
    by_label: dict[str, list[int]] = {}
    for i, seg in enumerate(lines):
        by_label.setdefault(seg.get("speaker") or "", []).append(i)
    out = {}
    for label, idx in by_label.items():
        votes = {t: sum(1 for i in idx if max(lv, key=lambda u: lv[u][i] - speech[u]) == t) for t in audio}
        track = max(votes, key=votes.get)
        clean = [i for i in idx if lv[track][i] >= speech[track] - LOUD_DB
                 and all(lv[t][i] < speech[t] - OTHER_QUIET_DB for t in audio if t != track)]
        clean = sorted(clean, key=lambda i: lines[i]["start"] - lines[i]["end"])[:LINES_PER_PRINT]
        vecs, secs = [], []
        for i in clean:
            a, b = lines[i]["start"], min(lines[i]["end"], len(audio[track]) / SR)
            if b - a < 1.0:
                continue  # a line past the end of a shorter track
            vecs.append(embed(audio[track], a, b))
            secs.append(min(b - a, EMBED_MAX_SEC))
        if sum(secs) < MIN_PRINT_SEC:
            continue
        v = np.sum([v * w for v, w in zip(vecs, secs, strict=True)], axis=0)
        out[label] = (v / (np.linalg.norm(v) + 1e-9), float(sum(secs)))
    return out


class VoiceLibrary:
    """JSON store {"voices": {name: {meeting_id: {"emb", "sec", "ts"}}}}. Thread-safe."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text()).get("voices", {})
        except (OSError, ValueError):
            return {}

    def _save(self, lib: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"voices": lib}, ensure_ascii=False))
        tmp.replace(self.path)

    def centroids(self, lib: dict | None = None) -> dict[str, np.ndarray]:
        out = {}
        for name, prints in (self.load() if lib is None else lib).items():
            if prints:
                v = np.sum([np.array(p["emb"]) * p["sec"] for p in prints.values()], axis=0)
                out[name] = v / (np.linalg.norm(v) + 1e-9)
        return out

    def remember(self, meeting_id: str, prints: dict) -> list[str]:
        """Store the meeting's prints of named speakers (replacing its earlier ones)."""
        named = {label: p for label, p in prints.items() if not is_generic(label)}
        with self._lock:
            lib = self.load()
            for name in list(lib):
                lib[name].pop(meeting_id, None)
                if not lib[name]:
                    del lib[name]
            for label, (emb, sec) in named.items():
                lib.setdefault(label, {})[meeting_id] = {"emb": [round(float(x), 5) for x in emb],
                                                         "sec": round(sec, 1), "ts": time.time()}
            self._save(lib)
        return sorted(named)

    def match(self, prints: dict, taken: set[str] = frozenset(), lib: dict | None = None
              ) -> dict[str, tuple[str, float]]:
        """{generic label: (name, score)}: best voice >= MATCH and MARGIN clear of the
        runner-up; each name goes to one label (the most similar); names in `taken`
        (already in the meeting, e.g. from the screen) are not given out again."""
        cents = self.centroids(lib)
        cands = []
        for label, (emb, _) in prints.items():
            if not is_generic(label):
                continue
            scores = sorted(((float(emb @ c), n) for n, c in cents.items() if n not in taken), reverse=True)
            if not scores or scores[0][0] < MATCH:
                continue
            if len(scores) > 1 and scores[0][0] - scores[1][0] < MARGIN:
                continue
            cands.append((scores[0][0], label, scores[0][1]))
        out, used = {}, set()
        for score, label, name in sorted(cands, reverse=True):
            if name not in used and label not in out:
                out[label] = (name, round(score, 2))
                used.add(name)
        return out

    def apply(self, meeting_id: str, audio: dict, segments: list[dict], embed) -> dict[str, tuple[str, float]]:
        """Name the generic labels the library recognizes (segments renamed in place), then
        learn the meeting's named voices. Returns {old label: (name, score)}."""
        prints = speaker_prints(audio, segments, embed)
        taken = {s.get("speaker") for s in segments if not is_generic(s.get("speaker") or "")}
        found = self.match(prints, taken)
        for label, (name, _) in found.items():
            prints[name] = prints.pop(label)
        for seg in segments:
            if seg.get("speaker") in found:
                seg["speaker"] = found[seg["speaker"]][0]
        self.remember(meeting_id, prints)
        return found

    def forget(self, name: str) -> bool:
        with self._lock:
            lib = self.load()
            if lib.pop(name, None) is None:
                return False
            self._save(lib)
        return True

    def rename(self, old: str, new: str) -> bool:
        new = " ".join(new.split())[:60]
        with self._lock:
            lib = self.load()
            if old not in lib or not new or is_generic(new):
                return False
            lib.setdefault(new, {}).update(lib.pop(old))
            self._save(lib)
        return True

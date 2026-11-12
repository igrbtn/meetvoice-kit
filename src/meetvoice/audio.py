"""Audio primitives: reading a WAV that is still being written, VAD, loudness, echo.

Everything here is numpy only and works on 16 kHz mono float32 unless stated otherwise.
"""

import struct
from pathlib import Path

import numpy as np

SR = 16000


class WavTail:
    """Incremental reader for a WAV file that is still being written.

    Capture tools usually finalize the RIFF/data sizes only on stop, so the data chunk is
    located once and new frames are read by file size. Supports PCM16 and float32,
    WAVE_FORMAT_EXTENSIBLE, any channel count (downmixed to mono).
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data_offset: int | None = None
        self.channels = 1
        self.rate = 48000
        self.fmt_tag = 1
        self.bits = 16
        self.read_frames = 0

    def _parse_header(self) -> bool:
        try:
            with open(self.path, "rb") as f:
                head = f.read(65536)
        except OSError:
            return False
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return False
        pos, fmt_seen = 12, False
        while pos + 8 <= len(head):
            cid, size = head[pos:pos + 4], struct.unpack("<I", head[pos + 4:pos + 8])[0]
            body = pos + 8
            if cid == b"fmt " and body + 16 <= len(head):
                self.fmt_tag, self.channels, self.rate = struct.unpack("<HHI", head[body:body + 8])
                self.bits = struct.unpack("<H", head[body + 14:body + 16])[0]
                if self.fmt_tag == 0xFFFE and body + 26 <= len(head):  # WAVE_FORMAT_EXTENSIBLE
                    self.fmt_tag = struct.unpack("<H", head[body + 24:body + 26])[0]
                fmt_seen = True
            elif cid == b"data":
                if not fmt_seen:
                    return False
                self.data_offset = body
                return True
            pos = body + size + (size & 1)
        return False

    def read_new(self) -> np.ndarray:
        """New samples since the last call, mono float32 at the file's own rate."""
        if self.data_offset is None and not self._parse_header():
            return np.zeros(0, dtype=np.float32)
        frame_bytes = self.bits // 8 * self.channels
        try:
            size = self.path.stat().st_size
        except OSError:
            return np.zeros(0, dtype=np.float32)
        n = (size - self.data_offset) // frame_bytes - self.read_frames
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        with open(self.path, "rb") as f:
            f.seek(self.data_offset + self.read_frames * frame_bytes)
            raw = f.read(n * frame_bytes)
        n = len(raw) // frame_bytes
        self.read_frames += n
        raw = raw[:n * frame_bytes]
        if self.fmt_tag == 3 and self.bits == 32:
            samples = np.frombuffer(raw, dtype="<f4")
        elif self.bits == 16:
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        else:
            return np.zeros(0, dtype=np.float32)
        if self.channels > 1:
            samples = samples.reshape(-1, self.channels).mean(axis=1)
        return samples.astype(np.float32)


def resample(samples: np.ndarray, rate: int, target: int = SR) -> np.ndarray:
    """Linear resampling: good enough for speech models, no scipy needed."""
    if rate == target or samples.size == 0:
        return samples
    n_out = int(samples.size * target / rate)
    return np.interp(np.linspace(0, samples.size - 1, n_out),
                     np.arange(samples.size, dtype=np.float64), samples).astype(np.float32)


def quiet_cut(samples: np.ndarray, min_cut_sec: float = 3.0, rate: int = SR) -> int:
    """Index to cut a live window at: the quietest 100 ms after min_cut_sec, so words
    are not split. Searches the whole buffer: after a lag several seconds arrive at once."""
    frame = rate // 10
    start = min(int(min_cut_sec * rate), max(0, samples.size - frame))
    best, best_rms = samples.size, None
    for i in range(start, samples.size - frame + 1, frame // 2):
        rms = float(np.sqrt(np.mean(samples[i:i + frame] ** 2)))
        if best_rms is None or rms < best_rms:
            best, best_rms = i + frame // 2, rms
    return best


def pause_cut(buf: np.ndarray, min_final_sec: float = 2.0, max_buf_sec: float = 8.0, rate: int = SR) -> int | None:
    """Where to close an utterance in the open buffer, or None to keep listening: a trailing
    pause, shorter the longer the buffer runs (0.6 s under 5 s, 0.35 s under max_buf_sec, then
    0.25 s - fluent speakers rarely pause 0.6 s), or the quietest point at max_buf_sec."""
    dur = buf.size / rate
    if dur < min_final_sec:
        return None
    frame = rate // 50                                   # 20 ms
    n = buf.size // frame
    db = 20 * np.log10(np.sqrt(np.mean(buf[:n * frame].reshape(n, frame) ** 2, axis=1)) + 1e-9)
    quiet = db < max(np.percentile(db, 15) + 6, -55)
    need = 0.6 if dur < 5 else 0.35 if dur < max_buf_sec else 0.25
    run = 0
    for i in range(n - 1, -1, -1):                       # trailing silence length
        if not quiet[i]:
            break
        run += 1
    if run * 0.02 >= need and (n - run) * 0.02 >= min_final_sec:
        return min(buf.size, (n - run) * frame + int(0.1 * rate))
    if dur >= max_buf_sec:
        return quiet_cut(buf, min_cut_sec=max_buf_sec * 0.5, rate=rate)
    return None


def speech_regions(audio, sr: int = SR, frame_sec: float = 0.03, gap_sec: float = 0.8,
                   min_sec: float = 0.25, pad_sec: float = 0.4) -> list[tuple[float, float]]:
    """Energy VAD relative to the track's own noise floor. Whisper fed only these spans
    cannot hallucinate on the silent stretches of a mic track while the other side talks."""
    audio = np.asarray(audio, dtype=np.float32)
    n = int(sr * frame_sec)
    frames = len(audio) // n
    if frames == 0:
        return []
    rms = np.sqrt(np.mean(audio[:frames * n].reshape(frames, n) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    voiced = db > max(np.percentile(db, 10) + 12, -50)
    regions: list[list[float]] = []
    for i in np.flatnonzero(voiced):
        t = i * frame_sec
        if regions and t - regions[-1][1] <= gap_sec:
            regions[-1][1] = t + frame_sec
        else:
            regions.append([t, t + frame_sec])
    total = frames * frame_sec
    out: list[tuple[float, float]] = []
    for a, b in regions:
        if b - a < min_sec:
            continue
        a, b = max(a - pad_sec, 0.0), min(b + pad_sec, total)
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return [(round(a, 2), round(b, 2)) for a, b in out]


def group_regions(regions, max_sec: float) -> list[list[tuple[float, float]]]:
    """Consecutive regions packed into groups spanning at most max_sec."""
    groups: list[list[tuple[float, float]]] = []
    for a, b in regions:
        if groups and b - groups[-1][0][0] <= max_sec:
            groups[-1].append((a, b))
        else:
            groups.append([(a, b)])
    return groups


def level_db(audio, a: float, b: float, sr: int = SR) -> float:
    """Loudness of a span: 90th percentile of 30 ms RMS, in dB."""
    x = np.asarray(audio[int(a * sr):int(b * sr)], dtype=np.float32)
    n = len(x) // 480 * 480
    if n == 0:
        return -120.0
    rms = np.sqrt((x[:n].reshape(-1, 480) ** 2).mean(axis=1))
    return float(20 * np.log10(np.percentile(rms, 90) + 1e-9))


# -- echo of the call on the mic (speakers instead of a headset) -----------------------

def _log_envelope(audio, hop: int = 160):
    n = len(audio) // hop * hop
    env = np.log(np.add.reduceat(np.asarray(audio[:n], np.float64) ** 2, np.arange(0, n, hop)) + 1e-9)
    return env - env.mean()


def _corr(a, b) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / den if den else 0.0


def crosstalk_lag(mic, ref, max_lag_sec: float = 1.0, min_corr: float = 0.15,
                  min_sec: float = 30.0) -> int | None:
    """Lag in 10 ms frames at which the call track shows up on the mic (negative = the mic
    hears it first: people in the same room). None when unrelated (headset)."""
    em, er = _log_envelope(np.asarray(mic)), _log_envelope(np.asarray(ref))
    n = min(len(em), len(er))
    if n < min_sec * 100:
        return None
    em, er = em[:n], er[:n]
    size = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(em, size) * np.conj(np.fft.rfft(er, size)), size)
    k = int(max_lag_sec * 100)
    cc = np.concatenate([cc[-k:], cc[:k + 1]]) / (np.linalg.norm(em) * np.linalg.norm(er) + 1e-12)
    best = int(np.argmax(cc))
    return best - k if cc[best] >= min_corr else None


def drop_crosstalk(regions, mic, ref, lag: int, win_sec: float = 2.0,
                   threshold: float = 0.45) -> list[tuple[float, float]]:
    """Cut the parts of mic speech regions whose envelope follows the call track. The user
    talking over the call decorrelates the envelopes, so double talk is kept."""
    em, er = _log_envelope(np.asarray(mic)), _log_envelope(np.asarray(ref))
    step = int(win_sec * 100)
    out: list[tuple[float, float]] = []
    for a, b in regions:
        t, keep_from = int(a * 100), None
        while t < int(b * 100):
            u = min(t + step, int(b * 100))
            s0, s1 = t - lag, u - lag
            other = 0 <= s0 and s1 <= len(er) and u - t >= 30 and _corr(em[t:u], er[s0:s1]) >= threshold
            if not other and keep_from is None:
                keep_from = t
            if other and keep_from is not None:
                out.append((keep_from / 100, t / 100))
                keep_from = None
            t = u
        if keep_from is not None:
            out.append((keep_from / 100, b))
    return [(round(x, 2), round(y, 2)) for x, y in out if y - x >= 0.3]

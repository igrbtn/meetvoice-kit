"""Unit tests: no models, no network. Recognition and embeddings are replaced by fakes."""

import json
import struct
import time

import numpy as np
import pytest

SR = 16000


def _wav_header(channels=2, rate=48000, bits=16, fmt_tag=1) -> bytes:
    block = channels * bits // 8
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, fmt_tag, channels, rate, rate * block, block, bits)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


def _tone(seconds, rate, amp=0.3, channels=2):
    t = np.arange(int(seconds * rate)) / rate
    x = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return np.repeat(x[:, None], channels, axis=1) if channels > 1 else x[:, None]


def _pcm16(frames) -> bytes:
    return (np.clip(frames, -1, 1) * 32767).astype("<i2").tobytes()


# -- audio ---------------------------------------------------------------------------

def test_wav_tail_reads_growing_file(tmp_path):
    from meetvoice.audio import WavTail
    p = tmp_path / "a.wav"
    p.write_bytes(_wav_header())
    tail = WavTail(p)
    assert tail.read_new().size == 0
    with open(p, "ab") as f:
        f.write(_pcm16(_tone(1, 48000)))
    assert tail.read_new().size == 48000 and tail.rate == 48000
    with open(p, "ab") as f:
        f.write(_pcm16(_tone(0.5, 48000)))
    assert tail.read_new().size == 24000


def test_resample_and_quiet_cut():
    from meetvoice.audio import quiet_cut, resample
    assert resample(np.zeros(48000, np.float32), 48000).size == 16000
    audio = np.concatenate([_tone(4, SR, channels=1)[:, 0], np.zeros(SR // 2, np.float32),
                            _tone(3, SR, channels=1)[:, 0]])
    assert 4.0 <= quiet_cut(audio) / SR <= 4.5


def test_speech_regions_and_crosstalk():
    from meetvoice.audio import crosstalk_lag, drop_crosstalk, speech_regions
    rng = np.random.default_rng(0)
    # bursts of noise as "speech" on the call; the mic hears them 200 ms later, quieter
    call = np.zeros(40 * SR, np.float32)
    for k in range(0, 36, 3):
        call[k * SR:k * SR + SR + SR // 2] = rng.normal(0, 0.2, SR + SR // 2)
    mic = np.zeros_like(call)
    mic[int(0.2 * SR):] = 0.3 * call[:-int(0.2 * SR)]
    regions = speech_regions(mic)
    assert regions
    lag = crosstalk_lag(mic, call)
    assert lag is not None and abs(lag - 20) <= 2
    kept = drop_crosstalk(regions, mic, call, lag)
    # only stubs too short to judge (< 0.3 s windows, the very start) survive
    assert sum(b - a for a, b in kept) < 0.25 * sum(b - a for a, b in regions)


# -- asr filters ---------------------------------------------------------------------

def test_collapse_and_clean():
    from meetvoice.asr import clean_segments, collapse_repeats
    assert collapse_repeats("да да да да да нет") == "да да нет"
    segs = [{"text": t} for t in ("Субтитры сделал DimaTorzok", "Привет", "Привет", "Привет", "Пока")]
    assert [s["text"] for s in clean_segments(segs)] == ["Привет", "Привет", "Пока"]


# -- uzbek helpers -------------------------------------------------------------------

def test_uzbek_text_checks():
    from meetvoice import uzbek
    assert uzbek.is_uzbek("bu yaxshi, menimcha ularga aytib bo'ladi")
    assert not uzbek.is_uzbek("и тогда больше вопрос нету")
    assert not uzbek.is_uzbek("we still need the final list")
    assert uzbek.tidy("xo'p. xo'p. xo'. musiqa. yaxshi") == "xo'p. yaxshi"
    name = "model.decoder.layers.3.self_attn.q_proj.weight"
    assert uzbek._hf_to_mlx_name(name) == "decoder.blocks.3.attn.query.weight"
    assert uzbek._hf_to_mlx_name("proj_out.weight") is None
    assert [len(p) for p in uzbek._pieces([(0.0, 30.0)])] == [1, 1, 1]


# -- live ----------------------------------------------------------------------------

def _tracks(tmp_path):
    from meetvoice.live import Track
    (tmp_path / "call.wav").write_bytes(_wav_header())
    (tmp_path / "mic.wav").write_bytes(_wav_header(channels=1, rate=24000))
    return [Track(tmp_path / "call.wav", "Собеседник"), Track(tmp_path / "mic.wav", "Я", mic=True)]


def _write(path, frames):
    with open(path, "ab") as f:
        f.write(_pcm16(frames))


def test_live_windows_and_translation(tmp_path, monkeypatch):
    from meetvoice import live
    monkeypatch.setattr(live, "POLL_SEC", 0.05)
    calls = []

    def stt(audio):
        calls.append(audio.size / SR)
        return [{"start": 0.5, "end": 1.5, "text": f"phrase {len(calls)}", "lang": "en"}]

    lt = live.LiveTranscriber(_tracks(tmp_path), transcriber=stt, out_path=tmp_path / "live.jsonl",
                              translator=lambda text, target, **kw: f"[{target}|{kw['source_lang']}] {text}")
    lt.start()
    _write(tmp_path / "call.wav", _tone(7, 48000))
    _write(tmp_path / "call.wav", np.zeros((24000, 2), np.float32))
    _write(tmp_path / "call.wav", _tone(3, 48000))
    time.sleep(0.5)
    lt.stop(flush=True)
    entries = lt.since(0)
    assert len(entries) == 2 and entries[0]["start"] == 0.5 and 7.0 <= entries[1]["start"] <= 8.2
    # English is foreign to a Russian-speaking user: translated without live translation on
    assert all(e["translation"] == f"[ru|en] {e['text']}" and e["target"] == "ru" for e in entries)
    assert len((tmp_path / "live.jsonl").read_text().splitlines()) == 2


def test_live_uzbek_routing(tmp_path, monkeypatch):
    from meetvoice import live
    monkeypatch.setattr(live, "POLL_SEC", 0.05)
    forced, stt = [], []

    def uz_finder(audio, force=False):
        forced.append(force)
        return True, [{"start": 0.2, "end": 2.0, "text": "bu yaxshi", "lang": "uz"}]

    lt = live.LiveTranscriber(_tracks(tmp_path), transcriber=lambda a: stt.append(a) or [],
                              translator=lambda t, target, **kw: f"[{target}] {t}", uz_finder=uz_finder)
    lt.start()
    _write(tmp_path / "call.wav", _tone(7, 48000))
    _write(tmp_path / "call.wav", np.zeros((24000, 2), np.float32))
    _write(tmp_path / "call.wav", _tone(3, 48000))
    _write(tmp_path / "mic.wav", _tone(7, 24000, channels=1))
    time.sleep(0.5)
    lt.stop(flush=True)
    entries = lt.since(0)
    # the call's Uzbek is translated; the mic's Uzbek is the room and dropped; Whisper never ran
    assert entries and all(e["speaker"] == "Собеседник" and e["translation"] == "[ru] bu yaxshi" for e in entries)
    assert stt == [] and True in forced
    assert lt.target({"lang": "ru"}) == "" and lt.target({"lang": "uz"}) == "ru"
    lt.translate_to = "en"
    assert lt.target({"lang": "en"}) == "" and lt.target({"lang": "ru"}) == "en"


def test_live_push_source(monkeypatch):
    from meetvoice import live
    monkeypatch.setattr(live, "POLL_SEC", 0.05)
    src = live.PushSource(rate=48000)
    got = []
    lt = live.LiveTranscriber([live.Track(None, "Guest", source=src)],
                              transcriber=lambda a: got.append(a.size) or [{"start": 0, "end": 1, "text": "hi",
                                                                            "lang": "ru"}])
    lt.start()
    for _ in range(14):
        src.push(_tone(0.5, 48000))   # stereo chunks from a callback
    time.sleep(0.4)
    lt.stop(flush=True)
    assert sum(got) == 7 * SR and lt.since(0)[0]["speaker"] == "Guest"


def test_live_silence_is_skipped(tmp_path, monkeypatch):
    from meetvoice import live
    monkeypatch.setattr(live, "POLL_SEC", 0.05)
    stt = []
    lt = live.LiveTranscriber(_tracks(tmp_path), transcriber=lambda a: stt.append(a) or [])
    lt.start()
    _write(tmp_path / "mic.wav", np.zeros((24000 * 8, 1), np.float32))
    time.sleep(0.3)
    lt.stop(flush=True)
    assert stt == []


# -- translate -----------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_translator_request(monkeypatch, kind):
    from meetvoice import translate
    sent = {}
    reply = ({"message": {"content": "\"Привет\""}} if kind == "ollama"
             else {"choices": [{"message": {"content": "\"Привет\""}}]})

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(reply).encode()

    def urlopen(req, timeout):
        sent["url"], sent["body"] = req.full_url, json.loads(req.data)
        return Resp()

    monkeypatch.setattr(translate.urllib.request, "urlopen", urlopen)
    tr = translate.Translator("http://127.0.0.1:1", "m", kind=kind)
    assert tr("salom", "ru", context="prev line", source_lang="uz") == "Привет"
    system = sent["body"]["messages"][0]["content"]
    assert "узбекский" in system and "prev line" in system and "как есть" not in system
    if kind == "ollama":
        assert sent["url"].endswith("/api/chat") and sent["body"]["think"] is False
    else:
        assert sent["url"].endswith("/v1/chat/completions")


# -- voices --------------------------------------------------------------------------

def _meeting(names):
    """Mic lines 0-30 s, call lines 40-70 s, each side silent while the other talks."""
    me = [(i, i + 4) for i in range(0, 30, 5)]
    them = [(i, i + 4) for i in range(40, 70, 5)]
    audio = {"mic": np.full(80 * SR, 1e-4, np.float32), "call": np.full(80 * SR, 1e-4, np.float32)}
    for track, spans in (("mic", me), ("call", them)):
        for a, b in spans:
            audio[track][a * SR:b * SR] = _tone(b - a, SR, channels=1)[:, 0]
    segs = [{"start": a, "end": b, "speaker": names[0]} for a, b in me]
    segs += [{"start": a, "end": b, "speaker": names[1]} for a, b in them]
    return audio, segs


def _embed(audio, a, b):
    v = [1.0, 0.1, 0.0] if a < 35 else [0.1, 1.0, 0.0]
    return np.array(v) / np.linalg.norm(v)


def test_voice_library_round_trip(tmp_path):
    from meetvoice.voices import VoiceLibrary, is_generic, speaker_prints
    lib = VoiceLibrary(tmp_path / "voices.json")
    audio, segs = _meeting(["Anna", "Boris"])
    assert lib.remember("m1", speaker_prints(audio, segs, _embed)) == ["Anna", "Boris"]

    audio2, segs2 = _meeting(["Я", "Speaker 1"])
    found = lib.apply("m2", audio2, segs2, _embed)
    assert {k: v[0] for k, v in found.items()} == {"Я": "Anna", "Speaker 1": "Boris"}
    assert {s["speaker"] for s in segs2} == {"Anna", "Boris"}
    assert sorted(lib.load()["Boris"]) == ["m1", "m2"]
    # a name already in the meeting is not handed out by voice
    p = speaker_prints(*_meeting(["Я", "Speaker 1"]), _embed)
    assert "Speaker 1" not in lib.match(p, taken={"Boris"})
    assert is_generic("Speaker 3") and not is_generic("Anna")
    assert lib.rename("Boris", "Boris K.") and lib.forget("Anna")
    assert sorted(lib.load()) == ["Boris K."]


def test_voice_prints_skip_double_talk(tmp_path):
    from meetvoice.voices import speaker_prints
    spans = [(i, i + 4) for i in range(0, 40, 5)]
    audio = {"mic": np.full(45 * SR, 1e-4, np.float32), "call": np.full(45 * SR, 1e-4, np.float32)}
    for a, b in spans:
        audio["mic"][a * SR:b * SR] = audio["call"][a * SR:b * SR] = _tone(b - a, SR, channels=1)[:, 0]
    segs = [{"start": a, "end": b, "speaker": "Я"} for a, b in spans]
    segs += [{"start": a, "end": b, "speaker": "Speaker 1"} for a, b in spans]
    assert speaker_prints(audio, segs, _embed) == {}


@pytest.mark.parametrize("second", [0.99, 0.5])
def test_voice_match_margin(tmp_path, second):
    from meetvoice.voices import VoiceLibrary
    lib = {"A": {"m": {"emb": [1, 0], "sec": 30, "ts": 0}},
           "B": {"m": {"emb": [second, (1 - second ** 2) ** 0.5], "sec": 30, "ts": 0}}}
    v = np.array([1.0, 0.05]) / np.linalg.norm([1.0, 0.05])
    got = VoiceLibrary(tmp_path / "v.json").match({"Speaker 1": (v, 30.0)}, lib=lib)
    assert got == ({} if second == 0.99 else {"Speaker 1": ("A", 1.0)})


def test_overlay_server(tmp_path):
    import urllib.request

    from meetvoice import server

    class Live:
        error = None

        def since(self, after):
            return [{"i": 0, "speaker": "S", "text": "salom", "lang": "uz", "translation": "привет", "target": "ru"}]

    httpd = server.serve(lambda: Live(), port=0)
    port = httpd.server_address[1]
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        data = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/live?after=0").read())
    finally:
        httpd.shutdown()
    assert "live?after=" in page and data["active"] and data["entries"][0]["translation"] == "привет"

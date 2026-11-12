# meetvoice-kit

Живые субтитры встречи с переводом (с отдельным путём для узбекской речи, которую Whisper не
знает), окно-оверлей поверх экрана и библиотека голосов, которая называет спикеров по голосу в
следующих встречах. Локально: mlx-whisper на Apple Silicon, перевод локальной LLM (Ollama и
OpenAI-совместимые серверы), эмбеддинги голоса sherpa-onnx на CPU.

Модуль выделен из рабочего приложения записи встреч и рассчитан на встраивание в чужую систему.
**Начните с [docs/AGENT.md](docs/AGENT.md)** - пошаговая инструкция для AI-агента (Claude Opus),
который будет подключать модуль: что выяснить о системе, как подключить звук, перевод, окно и
голоса, что проверить на приёмке.

| Документ | О чём |
|---|---|
| [docs/AGENT.md](docs/AGENT.md) | интеграция по шагам, контракт `/live`, чек-лист приёмки |
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | откуда взялись пороги, измерения, скрипт для своих данных |
| [docs/PITFALLS.md](docs/PITFALLS.md) | найденные на живых звонках поломки и их решения |

## Как это работает

```
 mic.wav / call.wav (растут)        или PushSource (колбэки захвата)
            |
     WavTail / resample 16 kHz
            |
   открытый буфер --каждую 1 с--> черновик (без меток слов, серым)  --> drafts[]
            |
   пауза (0.6 -> 0.25 с по длине буфера) или 8 с --> финал (со словарём)
            |
   mic: вырезать эхо звонка (crosstalk_lag + drop_crosstalk)
            |
   не русский/английский на слух? --да--> узбекская модель --узбекский текст?--да--> lang=uz
            | нет                                             | нет
            v                                                 v
        Whisper turbo  <------------------------------------- +
            |
   фильтры галлюцинаций, петель, чужих алфавитов, повторов; unsure / filler
            |
   entries[]  --->  поток перевода (LLM, контекст 4 реплики,   --->  entries[i].translation
                    только строки без unsure / filler)
            |
   /live?after=N  --->  overlay.html  (браузер / встроенный UI / NSPanel на macOS)

 после встречи:  реплики с диаризацией + дорожки  --->  VoiceLibrary.apply
                 ("Speaker 1" -> "Anna", 0.81)            voices.json (отпечатки по встречам)
```

## Быстрый старт (Apple Silicon)

```bash
python3.12 -m venv .venv
.venv/bin/pip install ".[dev,mlx,voices]"
.venv/bin/pytest -q
ollama pull gemma3:12b
.venv/bin/python examples/live_from_wav.py some_call.m4a --llm-model gemma3:12b
# открыть http://127.0.0.1:8765/   или добавить --panel для окна поверх экрана (нужен ".[macos]")
```

Библиотека голосов на двух своих встречах:

```bash
.venv/bin/python examples/voices_demo.py learn m1 --track mic=m1_mic.wav --track call=m1_call.wav --lines m1.json
.venv/bin/python examples/voices_demo.py match m2 --track mic=m2_mic.wav --track call=m2_call.wav --lines m2.json
```

`lines.json` - реплики с диаризацией `[{"start", "end", "speaker"}]`; в первой встрече имена
настоящие, во второй метки "Speaker 1", "Я".

## Структура

| Модуль | Что внутри | Зависимости |
|---|---|---|
| `audio.py` | чтение растущего WAV, ресемплинг, VAD, громкость, поиск и вырезание эха | numpy |
| `asr.py` | Whisper для коротких окон, фильтры галлюцинаций, общий LOCK, подмена модели в кэше | mlx-whisper |
| `uzbek.py` | конвертация HF -> mlx, поиск и проверка узбекской речи, декодирование кусками | mlx-whisper, huggingface_hub |
| `translate.py` | перевод реплики с контекстом: Ollama native или OpenAI-совместимый | stdlib |
| `live.py` | `LiveTranscriber`, `Track`, `PushSource` | numpy |
| `voices.py` | отпечатки голосов, `VoiceLibrary` (match/apply/rename/forget) | sherpa-onnx |
| `server.py` + `static/overlay.html` | `/live` и страница субтитров, только 127.0.0.1 | stdlib |
| `macos_overlay.py` | полупрозрачная панель поверх всех окон, не видна в демонстрации экрана | pyobjc WebKit |

## Ограничения

- Проверено только на macOS / Apple Silicon. Для CUDA нужен адаптер распознавания
  (набросок в docs/AGENT.md, шаг 2.3, не проверен).
- Живые субтитры - предварительный текст. Итоговую расшифровку встречи лучше делать по полным дорожкам после
  остановки записи.
- Отпечаток голоса - биометрические данные: храните локально, с согласия участников и с
  возможностью удалить.

## Модели

- Whisper: [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo)
- Узбекский: [islomov/rubaistt_v2_medium](https://huggingface.co/islomov/rubaistt_v2_medium), конвертируется в mlx при первом запуске
- Голос: NeMo titanet_small из [релизов sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)

Лицензии моделей - у их авторов. Код репозитория - MIT.

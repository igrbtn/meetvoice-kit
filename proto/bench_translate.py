"""Live-translation models: speed, loaded size, output on real Uzbek/English caption lines."""
import json
import sys
import time
import urllib.request

from meetvoice.translate import Translator

LINES = list(json.load(open(sys.argv[1]))["final_text"])
CTX = 4096


def ps():
    with urllib.request.urlopen("http://127.0.0.1:11434/api/ps") as r:
        return {m["name"]: (round(m["size"] / 1e9, 1), m.get("context_length")) for m in json.loads(r.read())["models"]}


def unload(model):
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                                 data=json.dumps({"model": model, "keep_alive": 0}).encode())
    urllib.request.urlopen(req).read()


for model in sys.argv[2:]:
    for m in ps():
        unload(m)
    tr = Translator(model=model, extra={"options": {"temperature": 0.2, "num_ctx": CTX}})
    t = time.time()
    tr("salom", "ru", source_lang="uz")
    load = time.time() - t
    print(f"=== {model}: load+first {load:.1f}s, loaded {ps().get(model)}")
    ctx, times = [], []
    for line in LINES:
        lang = ("en" if line.isascii() and " the " in f" {line.lower()} "
                else "uz" if "'" in line or "bo" in line else "ru")
        if lang == "ru":
            continue
        t = time.time()
        out = tr(line, "ru", context="\n".join(ctx[-4:]), source_lang=lang)
        times.append(time.time() - t)
        ctx.append(line)
        print(f"  {times[-1]:.2f}s  {line[:70]}\n         => {out[:110]}")
    print(f"  mean {sum(times) / len(times):.2f}s")

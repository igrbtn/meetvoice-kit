"""Line translation through a local or remote LLM. Stdlib only.

Two wire formats:
  - "ollama": Ollama's native /api/chat with think=false and a small context (num_ctx).
    A caption needs ~4k tokens; Ollama's default 32k context turned a 12B model into a
    21 GB resident and pushed a 24 GB Mac into swap. The context is fixed when the model
    is loaded: a model already loaded with a bigger one ignores num_ctx until unloaded.
    Use this for Ollama: through its OpenAI-compatible endpoint a reasoning model (gemma4,
    qwen3...) thinks before every caption, 10-30 s instead of 1-2 s.
  - "openai": any OpenAI-compatible /v1/chat/completions (LM Studio, vLLM, a gateway);
    `extra` is merged into the request body for server-specific switches.

Live captions are fragments with recognition errors, so the prompt carries the previous
lines as context and, for foreign speech, says that the text is ASR output.
"""

import json
import urllib.request

LANG_NAMES = {"ru": "русский", "en": "English", "uz": "o'zbekcha", "de": "Deutsch", "tr": "Türkçe"}
SOURCE_NAMES = {"uz": "узбекский", "en": "английский", "de": "немецкий", "fr": "французский",
                "es": "испанский", "tr": "турецкий", "kk": "казахский", "zh": "китайский"}


def system_prompt(target: str, context: str = "", source: str = "", glossary: str = "") -> str:
    prompt = (f"Переведи реплику из живой расшифровки разговора на язык: {LANG_NAMES.get(target, target)}. "
              "Верни только перевод, без пояснений и кавычек. Имена, названия продуктов и команды не переводи.")
    if source:
        # no "return it as is if already translated": a model unsure of the language did just that
        prompt += (f" Исходный язык: {source}; текст распознан автоматически и может содержать ошибки - "
                   "переводи по смыслу, исходный текст не повторяй.")
    else:
        prompt += " Если реплика уже на этом языке - верни её как есть."
    if glossary.strip():
        terms = ", ".join(t.strip() for t in glossary.replace("\n", ",").replace(";", ",").split(",") if t.strip())
        prompt += f" Термины (не переводить, писать так): {terms[:480]}."
    if context:
        prompt += f" Предыдущие реплики (для понимания, их не переводи):\n{context}"
    return prompt


class Translator:
    """translator(text, target, context="", source_lang="") -> str"""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "gemma3:12b",
                 kind: str = "ollama", api_key: str = "", timeout: float = 60.0, extra: dict | None = None,
                 num_ctx: int = 4096, glossary: str = ""):
        self.base_url, self.model, self.kind, self.num_ctx = base_url.rstrip("/"), model, kind, num_ctx
        self.api_key, self.timeout, self.extra = api_key, timeout, extra or {}
        self.glossary = glossary

    def _post(self, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def __call__(self, text: str, target: str, context: str = "", source_lang: str = "") -> str:
        system = system_prompt(target, context, SOURCE_NAMES.get(source_lang, ""), self.glossary)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
        if self.kind == "ollama":
            data = self._post("/api/chat", {"model": self.model, "messages": messages, "stream": False,
                                            "think": False, **self.extra,
                                            "options": {"temperature": 0.2, "num_ctx": self.num_ctx,
                                                        **self.extra.get("options", {})}})
            out = data["message"]["content"]
        else:
            data = self._post("/v1/chat/completions", {"model": self.model, "messages": messages,
                                                       "temperature": 0.2, "stream": False, **self.extra})
            out = data["choices"][0]["message"]["content"]
        return out.strip().strip('"')

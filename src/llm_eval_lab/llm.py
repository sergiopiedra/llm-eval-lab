"""LLM clients. `OllamaClient` talks to a local Ollama server; tests use `MockLLM`."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Protocol

from llm_eval_lab.text import strip_thinking


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    model: str

    def chat(self, messages: list[dict], *, schema: dict | None = None) -> str:
        """Return the assistant message content for a chat conversation."""
        ...


class OllamaClient:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: int = 42,
        num_ctx: int = 8192,
        timeout_s: float = 180.0,
        retries: int = 2,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}
        self.timeout_s = timeout_s
        self.retries = retries

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read())

    def check_ready(self) -> None:
        """Fail fast with an actionable message if Ollama is down or the model is missing."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as resp:
                tags = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.host} ({exc}). Start it with `ollama serve`."
            ) from exc
        names = {m["name"] for m in tags.get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names:
            raise LLMError(f"Model {self.model!r} is not pulled. Run `ollama pull {self.model}`.")

    def chat(self, messages: list[dict], *, schema: dict | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self.options,
            # Qwen3 is a hybrid reasoning model; disable thinking for short, cheap answers.
            "think": False,
        }
        if schema is not None:
            payload["format"] = schema  # Ollama structured outputs (JSON schema)
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                data = self._post("/api/chat", payload)
                return strip_thinking(data["message"]["content"])
            except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as exc:
                last_exc = exc
                time.sleep(2**attempt)
        raise LLMError(f"Ollama chat failed after {self.retries + 1} attempts: {last_exc}")


def make_client(backend: str, model: str, config, cases=None, behavior: str = "good") -> LLMClient:
    if backend == "ollama":
        return OllamaClient(
            model,
            host=config.host,
            temperature=config.temperature,
            seed=config.seed,
            num_ctx=config.num_ctx,
            timeout_s=config.timeout_s,
        )
    if backend == "mock":
        from llm_eval_lab.mock import MockLLM

        return MockLLM(cases or [], behavior=behavior)
    raise ValueError(f"unknown backend {backend!r} (expected 'ollama' or 'mock')")

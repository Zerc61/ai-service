"""Abstraksi LLM + implementasi Groq, dengan mode streaming.

- `BaseLLM`: interface (generate + stream).
- `GroqLLM`: memakai api-key dari settings; model env-configurable.
- `FakeLLM`: deterministik untuk test (tanpa jaringan).
"""
from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Sequence

from ..config import settings


class BaseLLM:
    model: str = ""

    def generate(self, messages: Sequence[Dict[str, str]]) -> str:
        raise NotImplementedError

    def stream(self, messages: Sequence[Dict[str, str]]) -> Iterator[str]:
        raise NotImplementedError


class GroqLLM(BaseLLM):
    def __init__(self, api_key: str = "", model: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.7, max_tokens: int = 1024) -> None:
        import groq
        key = api_key or settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError("GROQ_API_KEY belum di-set di environment/.env")
        self._client = groq.Groq(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: Sequence[Dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            messages=list(messages), model=self.model,
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: Sequence[Dict[str, str]]) -> Iterator[str]:
        resp = self._client.chat.completions.create(
            messages=list(messages), model=self.model,
            temperature=self.temperature, max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


class FakeLLM(BaseLLM):
    """Deterministik, tanpa API key/bantuan apa pun — untuk testing."""

    def __init__(self, model: str = "fake") -> None:
        self.model = model

    def generate(self, messages: Sequence[Dict[str, str]]) -> str:
        last_user = next((m["content"] for m in reversed(list(messages))
                          if m.get("role") == "user"), "")
        return f"[FakeLLM:{self.model}] Balasan untuk: {last_user[:40]}"

    def stream(self, messages: Sequence[Dict[str, str]]) -> Iterator[str]:
        parts = self.generate(messages).split(" ")
        for i, word in enumerate(parts):
            yield word + (" " if i < len(parts) - 1 else "")


def build_llm(provider: str = "groq", model: str = "llama-3.3-70b-versatile",
              api_key: str = "", temperature: float = 0.7,
              max_tokens: int = 1024) -> BaseLLM:
    if provider == "fake":
        return FakeLLM(model=model or "fake")
    if provider == "groq":
        return GroqLLM(api_key=api_key, model=model, temperature=temperature,
                       max_tokens=max_tokens)
    raise ValueError(f"LLM tidak dikenal: {provider}")

"""Orkestrasi chat RAG: embed query -> retrieve konteks -> build prompt -> stream.

Mendukung peran (role) sehingga system prompt menyesuaikan persona
(KAVI/RAKA/MAJA) sesuai token Sanctum yang diteruskan Laravel.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

from .embedder import BaseEmbedder
from .llm import BaseLLM
from .prompts import build_rag_prompt, get_system_prompt
from .retrieval import retrieve
from .vectorstore import BaseVectorStore


class ChatOrchestrator:
    def __init__(self, store: BaseVectorStore, embedder: BaseEmbedder, llm: BaseLLM,
                 top_k: int = 5, score_threshold: float = 0.0) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.score_threshold = score_threshold

    def _retrieve_context(self, message: str) -> str:
        result = retrieve(self.store, self.embedder, message,
                          top_k=self.top_k, score_threshold=self.score_threshold)
        return result.context

    def generate(self, message: str, role: Optional[str] = None,
                 history: Optional[Sequence[Dict[str, str]]] = None) -> str:
        context = self._retrieve_context(message)
        messages = self._build_messages(message, role, history, context)
        return self.llm.generate(messages)

    def stream(self, message: str, role: Optional[str] = None,
               history: Optional[Sequence[Dict[str, str]]] = None) -> Iterator[str]:
        context = self._retrieve_context(message)
        messages = self._build_messages(message, role, history, context)
        yield from self.llm.stream(messages)

    def _build_messages(self, message: str, role: Optional[str],
                        history: Optional[Sequence[Dict[str, str]]],
                        context: str) -> List[Dict[str, str]]:
        base = get_system_prompt(role)
        if context.strip():
            system_content = (f"{base}\n\nKonteks (gunakan sebagai sumber fakta):\n{context}")
        else:
            system_content = (f"{base}\n\nAnda tidak memiliki konteks. "
                              "Jangan mengarang fakta; akui keterbatasan data.")
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
        for m in (history or []):
            mrole = m.get("role")
            content = m.get("content")
            if mrole in ("user", "assistant") and content:
                messages.append({"role": mrole, "content": content})
        messages.append({"role": "user", "content": message})
        return messages

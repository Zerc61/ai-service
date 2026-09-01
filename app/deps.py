"""Dependency injection FastAPI — ekspos komponen terakit dari factory."""
from __future__ import annotations

from typing import Generator

from .factory import (get_embedder, get_llm, get_orchestrator, get_vector_store)
from .rag.chat import ChatOrchestrator
from .rag.embedder import BaseEmbedder
from .rag.llm import BaseLLM
from .rag.vectorstore import BaseVectorStore


def get_store() -> Generator[BaseVectorStore, None, None]:
    yield get_vector_store()


def get_embedder_dep() -> Generator[BaseEmbedder, None, None]:
    yield get_embedder()


def get_llm_dep() -> Generator[BaseLLM, None, None]:
    yield get_llm()


def get_chat() -> Generator[ChatOrchestrator, None, None]:
    yield get_orchestrator()

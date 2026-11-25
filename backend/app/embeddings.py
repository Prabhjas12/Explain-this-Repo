"""Step 3: turn text into vectors.

An embedding model maps a piece of text to a list of numbers (here 384 of them) so
that texts with similar meaning land close together. "how are redirects handled?"
and a function called `resolve_redirects` share almost no words, but their vectors
point in similar directions, which is what lets vector search find it.

Backends:
- fastembed (default): BAAI/bge-small-en-v1.5 run locally with ONNX Runtime. Free,
  private, no PyTorch. The first run downloads the model (~70 MB) from Hugging Face.
- openai: text-embedding-3-small through the API (needs OPENAI_API_KEY).
- wordllama: a 16 MB model that ships inside its pip package. Lower quality; useful
  when you can't download anything.
- hash: not a real model. Hashes words into a vector so tests run with no downloads.

All vectors are L2-normalized, so cosine similarity is a plain dot product.
"""
from __future__ import annotations

import hashlib
import os
import threading
from functools import lru_cache

import numpy as np

from . import config

DEFAULT_MODELS = {
    "fastembed": "BAAI/bge-small-en-v1.5",
    "openai": "text-embedding-3-small",
    "wordllama": "l2_supercat_256",
    "hash": "hash-512",
}


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        return m / (np.linalg.norm(m) or 1.0)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class Embedder:
    backend = "base"

    def __init__(self, model: str):
        self.model = model
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"{self.backend}:{self.model}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        with self._lock:
            return _normalize(self._embed_documents(texts))

    def embed_query(self, text: str) -> np.ndarray:
        with self._lock:
            return _normalize(self._embed_query(text))

    def _embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def _embed_query(self, text: str) -> np.ndarray:
        return self._embed_documents([text])[0]


class FastEmbedEmbedder(Embedder):
    backend = "fastembed"

    def __init__(self, model: str):
        super().__init__(model)
        from fastembed import TextEmbedding  # imported lazily: it's the slow part of startup
        self._m = TextEmbedding(model_name=model)

    def _embed_documents(self, texts):
        return np.array(list(self._m.passage_embed(texts, batch_size=32)))

    def _embed_query(self, text):
        return np.array(next(iter(self._m.query_embed(text))))


class OpenAIEmbedder(Embedder):
    backend = "openai"

    def __init__(self, model: str):
        super().__init__(model)
        self._key = os.environ.get("OPENAI_API_KEY", "")
        if not self._key:
            raise RuntimeError("ETR_EMBEDDINGS=openai needs OPENAI_API_KEY.")

    def _embed_documents(self, texts):
        import httpx
        out = []
        for i in range(0, len(texts), 128):
            r = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._key}"},
                json={"model": self.model, "input": [t[:24000] for t in texts[i:i + 128]]},
                timeout=120,
            )
            r.raise_for_status()
            out.extend(d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"]))
        return np.array(out)


class WordLlamaEmbedder(Embedder):
    backend = "wordllama"

    def __init__(self, model: str):
        super().__init__(model)
        from wordllama import WordLlama
        self._m = WordLlama.load()

    def _embed_documents(self, texts):
        return np.asarray(self._m.embed(texts, norm=True))


class HashEmbedder(Embedder):
    """Feature hashing over code-aware tokens. Fast and deterministic; not semantic."""
    backend = "hash"
    dim = 512

    def _embed_documents(self, texts):
        from .search import tokenize
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                m[row, h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return m


BACKENDS = {
    "fastembed": FastEmbedEmbedder,
    "openai": OpenAIEmbedder,
    "wordllama": WordLlamaEmbedder,
    "hash": HashEmbedder,
}

_lock = threading.Lock()


@lru_cache(maxsize=4)
def _build(backend: str, model: str) -> Embedder:
    if backend not in BACKENDS:
        raise RuntimeError(f"Unknown ETR_EMBEDDINGS={backend!r}. Use one of: {', '.join(BACKENDS)}.")
    return BACKENDS[backend](model)


def get_embedder(backend: str | None = None, model: str | None = None) -> Embedder:
    backend = (backend or config.EMBEDDINGS).lower()
    model = model or config.EMBEDDING_MODEL or DEFAULT_MODELS.get(backend, "")
    with _lock:
        return _build(backend, model)


def parse_name(name: str) -> tuple[str, str]:
    backend, _, model = name.partition(":")
    return backend, model

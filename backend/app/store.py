"""The vector store: how an index lives on disk and in memory.

Each indexed repo gets a folder in data/indexes/<repo_id>/:

    meta.json      what was indexed: source, commit, counts, timings, embedding model
    chunks.jsonl   one chunk per line (path, line range, symbol, text)
    vectors.npy    float32 matrix, one normalized row per chunk, same order as chunks.jsonl
    repo_map.json  file tree, languages and key files (used for the overview)
    overview.md    the generated overview, cached after the first time

Search is exact brute force: one matrix-vector product over every chunk. For a repo
with 20,000 chunks and 384 dimensions that's ~7.7M multiply-adds, about a
millisecond in NumPy, so an approximate index (FAISS, HNSW) would add complexity
without a visible speedup. That changes at millions of chunks.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path

import numpy as np

from . import config
from .chunking import Chunk
from .search import BM25, Hit, retrieve, search_text, tokenize


def index_path(repo_id: str) -> Path:
    if not repo_id or "/" in repo_id or "\\" in repo_id or repo_id.startswith("."):
        raise ValueError("bad repo id")
    return config.INDEX_DIR / repo_id


def content_hash(chunk: Chunk) -> str:
    return hashlib.sha1(chunk.embed_text().encode("utf-8", "replace")).hexdigest()


class RepoIndex:
    def __init__(self, repo_id: str, meta: dict, chunks: list[Chunk], vectors: np.ndarray, repo_map: dict):
        self.repo_id = repo_id
        self.meta = meta
        self.chunks = chunks
        self.vectors = vectors
        self.repo_map = repo_map
        self.bm25 = BM25([tokenize(search_text(c)) for c in chunks])
        self.paths = {c.path for c in chunks} | {f["path"] for f in repo_map.get("files", [])}

    # ---- persistence

    def save(self) -> None:
        d = index_path(self.repo_id)
        tmp = d.with_name(d.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        with open(tmp / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(tmp / "vectors.npy", self.vectors.astype(np.float32))
        (tmp / "repo_map.json").write_text(json.dumps(self.repo_map), encoding="utf-8")
        (tmp / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        # swap in atomically-ish so a crash never leaves a half-written index
        old = d.with_name(d.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if d.exists():
            d.rename(old)
        tmp.rename(d)
        shutil.rmtree(old, ignore_errors=True)

    @classmethod
    def load(cls, repo_id: str) -> "RepoIndex":
        d = index_path(repo_id)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        vectors = np.load(d / "vectors.npy")
        repo_map = json.loads((d / "repo_map.json").read_text(encoding="utf-8"))
        return cls(repo_id, meta, chunks, vectors, repo_map)

    def cached_vectors(self) -> dict[str, np.ndarray]:
        """content hash -> vector, so re-indexing only embeds chunks that changed."""
        return {content_hash(c): self.vectors[i] for i, c in enumerate(self.chunks)}

    # ---- overview cache

    def overview_file(self, provider: str) -> Path:
        return index_path(self.repo_id) / f"overview.{provider}.md"

    # ---- search

    def search(self, query: str, query_vector: np.ndarray | None, k: int = 8, mode: str = "hybrid",
               use_prior: bool = True) -> list[Hit]:
        return retrieve(self.chunks, self.vectors, self.bm25, query, query_vector, k=k, mode=mode,
                        per_file=config.MAX_CHUNKS_PER_FILE,
                        doc_weight=config.DOC_WEIGHT if use_prior else 1.0,
                        test_weight=config.TEST_WEIGHT if use_prior else 1.0)


class IndexCache:
    """Keeps loaded indexes in memory; reloads when an index on disk is rebuilt."""

    def __init__(self):
        self._lock = threading.Lock()
        self._items: dict[str, tuple[float, RepoIndex]] = {}

    def get(self, repo_id: str) -> RepoIndex | None:
        meta_file = index_path(repo_id) / "meta.json"
        if not meta_file.exists():
            return None
        mtime = meta_file.stat().st_mtime
        with self._lock:
            hit = self._items.get(repo_id)
            if hit and hit[0] == mtime:
                return hit[1]
        idx = RepoIndex.load(repo_id)
        with self._lock:
            self._items[repo_id] = (mtime, idx)
        return idx

    def put(self, idx: RepoIndex) -> None:
        mtime = (index_path(idx.repo_id) / "meta.json").stat().st_mtime
        with self._lock:
            self._items[idx.repo_id] = (mtime, idx)

    def drop(self, repo_id: str) -> None:
        with self._lock:
            self._items.pop(repo_id, None)


def list_indexes() -> list[dict]:
    out = []
    if not config.INDEX_DIR.exists():
        return out
    for d in sorted(config.INDEX_DIR.iterdir()):
        meta = d / "meta.json"
        if d.is_dir() and meta.exists() and not d.name.endswith((".tmp", ".old")):
            try:
                out.append(json.loads(meta.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    out.sort(key=lambda m: m.get("indexed_at", ""), reverse=True)
    return out

"""The indexing pipeline, end to end: clone -> scan -> chunk -> embed -> save.

`build_index` is a plain function (the eval script calls it directly). The web app
runs it in a background thread through `start_job` and the UI polls the job for
progress.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from .chunking import chunk_file
from .embeddings import get_embedder
from .ingest import IngestError, RepoSource, fetch, parse_source, scan
from .repo_map import build_repo_map
from .store import RepoIndex, content_hash, index_path

EMBED_BATCH = 64
Progress = Callable[[str, float, str], None]


def build_index(src: RepoSource, progress: Progress | None = None, embedder=None) -> RepoIndex:
    report = progress or (lambda stage, frac, msg: None)
    timings: dict[str, float] = {}
    t_start = t = time.perf_counter()

    report("cloning", 0.02, "Cloning the repository" if src.kind == "github" else "Reading the folder")
    root, commit = fetch(src)
    timings["clone_s"] = time.perf_counter() - t

    t = time.perf_counter()
    report("scanning", 0.12, "Choosing which files to read")
    scanned = scan(root)
    if not scanned.files:
        raise IngestError("No readable source files found in this repository.")
    timings["scan_s"] = time.perf_counter() - t

    t = time.perf_counter()
    chunks = []
    n_files = len(scanned.files)
    for i, f in enumerate(scanned.files):
        chunks.extend(chunk_file(f))
        if i % 50 == 0:
            report("chunking", 0.15 + 0.1 * i / n_files, f"Splitting files into chunks ({i}/{n_files})")
    if not chunks:
        raise IngestError("Found files, but none had any text to index.")
    timings["chunk_s"] = time.perf_counter() - t

    t = time.perf_counter()
    report("embedding", 0.26, "Loading the embedding model (the first run downloads it)")
    if embedder is None:
        try:
            embedder = get_embedder()
        except Exception as e:
            raise IngestError(
                f"Couldn't load the embedding model ({type(e).__name__}: {str(e)[:200]}). The first run downloads it "
                "from Hugging Face (about 70 MB), so check your internet connection. To skip the download, set "
                "ETR_EMBEDDINGS=wordllama in .env and run `pip install wordllama` in the .venv."
            ) from e
    cached: dict[str, np.ndarray] = {}
    meta_file = index_path(src.repo_id) / "meta.json"
    if meta_file.exists():
        try:
            old = RepoIndex.load(src.repo_id)
            if old.meta.get("embedding") == embedder.name:
                cached = old.cached_vectors()
        except Exception:
            cached = {}
    hashes = [content_hash(c) for c in chunks]
    todo = [i for i, h in enumerate(hashes) if h not in cached]
    rows: list[np.ndarray | None] = [cached.get(h) for h in hashes]
    for b in range(0, len(todo), EMBED_BATCH):
        batch = todo[b:b + EMBED_BATCH]
        vecs = embedder.embed_documents([chunks[i].embed_text() for i in batch])
        for i, v in zip(batch, vecs):
            rows[i] = v
        done = min(b + EMBED_BATCH, len(todo))
        report("embedding", 0.28 + 0.67 * done / max(len(todo), 1), f"Embedding chunks ({done}/{len(todo)})")
    vectors = np.vstack(rows).astype(np.float32)
    timings["embed_s"] = time.perf_counter() - t

    report("saving", 0.97, "Saving the index")
    repo_map = build_repo_map(scanned.files, chunks)
    timings["total_s"] = time.perf_counter() - t_start
    meta = {
        "repo_id": src.repo_id,
        "name": src.name,
        "kind": src.kind,
        "source": src.web_url or str(src.local_path),
        "web_url": src.web_url,
        "branch": src.branch,
        "local_path": str(src.local_path) if src.local_path else None,
        "root": str(root),
        "commit": commit,
        "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": n_files,
        "lines": sum(f.lines for f in scanned.files),
        "chunks": len(chunks),
        "skipped": scanned.skipped,
        "languages": repo_map["languages"][:8],
        "embedding": embedder.name,
        "dim": int(vectors.shape[1]),
        "reused_vectors": len(chunks) - len(todo),
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }
    idx = RepoIndex(src.repo_id, meta, chunks, vectors, repo_map)
    idx.save()
    report("ready", 1.0, "Ready")
    return idx


@dataclass
class Job:
    repo_id: str
    name: str
    source: str = ""
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Waiting to start"
    error: str | None = None
    failed_at: str | None = None  # the stage that was running when it failed
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def running(self) -> bool:
        return self.stage not in ("ready", "error")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["running"] = self.running
        return d


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def get_job(repo_id: str) -> Job | None:
    with _jobs_lock:
        return _jobs.get(repo_id)


def all_jobs() -> list[Job]:
    with _jobs_lock:
        return list(_jobs.values())


def start_job(source_text: str, on_ready: Callable[[RepoIndex], None] | None = None) -> Job:
    src = parse_source(source_text)
    with _jobs_lock:
        existing = _jobs.get(src.repo_id)
        if existing and existing.running:
            return existing
        job = Job(src.repo_id, src.name, source_text.strip())
        _jobs[src.repo_id] = job

    def progress(stage: str, frac: float, msg: str) -> None:
        job.stage, job.progress, job.message = stage, round(frac, 3), msg

    def run() -> None:
        try:
            idx = build_index(src, progress)
            if on_ready:
                on_ready(idx)
        except Exception as e:  # shown in the UI
            if not isinstance(e, IngestError):
                traceback.print_exc()
            job.failed_at = job.stage
            job.stage, job.error, job.message = "error", str(e) or type(e).__name__, "Indexing failed"
        finally:
            job.finished_at = time.time()

    threading.Thread(target=run, name=f"index-{src.repo_id}", daemon=True).start()
    return job

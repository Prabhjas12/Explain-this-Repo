"""The web API (FastAPI). The React frontend talks to these endpoints; in production the
same server also serves the built frontend, so there's one process and one port.

    GET    /api/health                      LLM providers available, embedding model
    GET    /api/repos                       indexed repos (+ any indexing in progress)
    POST   /api/repos        {source}       start indexing a GitHub URL, owner/repo or folder
    GET    /api/repos/{id}                  metadata, languages, file tree
    GET    /api/repos/{id}/job              indexing progress
    DELETE /api/repos/{id}                  remove the index (and the clone)
    POST   /api/repos/{id}/ask              question -> cited answer + sources
    POST   /api/repos/{id}/search           raw retrieval results (for inspecting search)
    GET    /api/repos/{id}/overview         "explain this repo" summary
    GET    /api/repos/{id}/file?path=...    one file's text, for the code viewer
"""
from __future__ import annotations

import json
import shutil
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, llm, rag
from .embeddings import DEFAULT_MODELS
from .indexer import all_jobs, get_job, start_job
from .ingest import IngestError
from .store import IndexCache, index_path, list_indexes

config.ensure_dirs()
app = FastAPI(title="Explain This Repo", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
cache = IndexCache()

Mode = Literal["hybrid", "vector", "keyword"]


class IndexRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=500)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=20000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    provider: str | None = None
    mode: Mode = "hybrid"
    k: int = Field(config.TOP_K, ge=1, le=20)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    mode: Mode = "hybrid"
    k: int = Field(10, ge=1, le=30)


def _index(repo_id: str):
    try:
        idx = cache.get(repo_id)
    except ValueError:
        idx = None
    if idx is None:
        raise HTTPException(404, "That repo isn't indexed yet.")
    return idx


@app.get("/api/health")
def health():
    embed_model = config.EMBEDDING_MODEL or DEFAULT_MODELS.get(config.EMBEDDINGS, "")
    try:
        default = llm.resolve(None)
    except llm.LLMError:
        default = "none"
    return {
        "ok": True,
        "embedding": f"{config.EMBEDDINGS}:{embed_model}",
        "llm": {"configured": config.LLM_PROVIDER, "default": default, "providers": llm.status()},
    }


@app.get("/api/repos")
def repos():
    indexed = list_indexes()
    ids = {m["repo_id"] for m in indexed}
    jobs = {j.repo_id: j.to_dict() for j in all_jobs()}
    out = [dict(m, job=jobs.get(m["repo_id"])) for m in indexed]
    pending = [{"repo_id": rid, "name": j["name"], "job": j} for rid, j in jobs.items() if rid not in ids]
    return pending + out


@app.post("/api/repos")
def index_repo(req: IndexRequest):
    try:
        job = start_job(req.source, on_ready=cache.put)
    except IngestError as e:
        raise HTTPException(400, str(e))
    return {"repo_id": job.repo_id, "job": job.to_dict()}


@app.get("/api/repos/{repo_id}/job")
def job_status(repo_id: str):
    job = get_job(repo_id)
    if job is None:
        if (index_path(repo_id) / "meta.json").exists():
            return {"repo_id": repo_id, "stage": "ready", "progress": 1.0, "running": False, "error": None, "message": "Ready"}
        raise HTTPException(404, "No indexing job for that repo.")
    return job.to_dict()


@app.get("/api/repos/{repo_id}")
def repo_detail(repo_id: str):
    idx = _index(repo_id)
    m = idx.repo_map
    job = get_job(repo_id)
    return {
        "meta": idx.meta,
        "job": job.to_dict() if job else None,
        "languages": m.get("languages", []),
        "tree": m.get("tree", ""),
        "key_files": m.get("key_files", []),
        "entry_points": m.get("entry_points", []),
        "files": [f["path"] for f in m.get("files", [])],
    }


@app.delete("/api/repos/{repo_id}")
def delete_repo(repo_id: str):
    idx = _index(repo_id)
    cache.drop(repo_id)
    shutil.rmtree(index_path(repo_id), ignore_errors=True)
    clone = config.REPOS_DIR / repo_id
    if idx.meta.get("kind") == "github" and clone.resolve().parent == config.REPOS_DIR.resolve():
        shutil.rmtree(clone, ignore_errors=True)
    return {"deleted": repo_id}


@app.post("/api/repos/{repo_id}/ask")
def ask(repo_id: str, req: AskRequest, stream: bool = False):
    """With ?stream=true the reply is two JSON lines: sources as soon as retrieval finishes, then the answer."""
    idx = _index(repo_id)
    history = [t.model_dump() for t in req.history]
    if not stream:
        return rag.ask(idx, req.question, history, req.provider, req.mode, req.k)
    events = rag.ask_stream(idx, req.question, history, req.provider, req.mode, req.k)
    return StreamingResponse((json.dumps(e) + "\n" for e in events), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/repos/{repo_id}/search")
def search(repo_id: str, req: SearchRequest):
    return rag.search(_index(repo_id), req.query, req.mode, req.k)


@app.get("/api/repos/{repo_id}/overview")
def overview(repo_id: str, provider: str | None = None, refresh: bool = False):
    return rag.overview(_index(repo_id), provider, refresh)


@app.get("/api/repos/{repo_id}/file")
def file(repo_id: str, path: str = Query(..., max_length=1000)):
    try:
        return rag.read_file(_index(repo_id), path)
    except FileNotFoundError:
        raise HTTPException(404, "File not found in this index.")


# ---- the built React app (frontend/dist), served from the same port
if (config.FRONTEND_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404)
        candidate = (config.FRONTEND_DIST / full_path).resolve()
        if full_path and candidate.is_file() and config.FRONTEND_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(config.FRONTEND_DIST / "index.html")

"""Ask a question, get a cited answer: retrieve -> build prompt -> generate."""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

from . import llm
from .embeddings import get_embedder, parse_name
from .ingest import notebook_to_text
from .prompts import ANSWER_SYSTEM, OVERVIEW_SYSTEM, answer_prompt, fallback_overview, overview_prompt
from .search import Hit
from .store import RepoIndex

CITATION = re.compile(r"\[(\d{1,2})\]")


def source_url(idx: RepoIndex, path: str, start: int | None = None, end: int | None = None) -> str | None:
    web = idx.meta.get("web_url")
    if not web:
        return None
    url = f"{web}/blob/{idx.meta.get('commit') or 'HEAD'}/{path}"
    if start and not path.endswith(".ipynb"):
        url += f"#L{start}" + (f"-L{end}" if end and end != start else "")
    return url


def source_dict(idx: RepoIndex, n: int, h: Hit) -> dict:
    c = h.chunk
    return {
        "n": n, "path": c.path, "start_line": c.start_line, "end_line": c.end_line,
        "symbol": c.symbol, "context": c.context, "kind": c.kind, "language": c.language,
        "text": c.text, "url": source_url(idx, c.path, c.start_line, c.end_line),
        "score": round(h.score, 5), "vector_rank": h.vector_rank, "keyword_rank": h.keyword_rank,
        "vector_score": None if h.vector_score is None else round(h.vector_score, 4),
    }


def query_vector(idx: RepoIndex, query: str, mode: str):
    if mode == "keyword":
        return None
    backend, model = parse_name(idx.meta["embedding"])  # always query with the model that built the index
    return get_embedder(backend, model).embed_query(query)


def retrieval_query(question: str, history: list[dict]) -> str:
    """Short follow-ups ("where is it called?") get the previous question added so retrieval has a subject."""
    if history and len(question.split()) < 8:
        prev = next((t.get("content", "") for t in reversed(history) if t.get("role") == "user"), "")
        if prev:
            return f"{prev} {question}"
    return question


def search(idx: RepoIndex, query: str, mode: str = "hybrid", k: int = 8) -> dict:
    t0 = time.perf_counter()
    hits = idx.search(query, query_vector(idx, query, mode), k=k, mode=mode)
    ms = (time.perf_counter() - t0) * 1000
    return {"results": [source_dict(idx, n, h) for n, h in enumerate(hits, 1)], "mode": mode, "retrieve_ms": round(ms, 1)}


def ask_stream(idx: RepoIndex, question: str, history: list[dict], provider: str | None = None,
               mode: str = "hybrid", k: int = 8):
    """Yields two events: the retrieved sources (fast), then the answer (slow, the LLM call)."""
    query = retrieval_query(question, history)
    t0 = time.perf_counter()
    hits = idx.search(query, query_vector(idx, query, mode), k=k, mode=mode)
    retrieve_ms = (time.perf_counter() - t0) * 1000
    sources = [source_dict(idx, n, h) for n, h in enumerate(hits, 1)]
    try:
        name = llm.resolve(provider)
    except llm.LLMError as e:
        name, resolve_error = None, str(e)
    else:
        resolve_error = None
    yield {"type": "sources", "question": question, "query": query, "mode": mode, "sources": sources,
           "provider": name, "timings": {"retrieve_ms": round(retrieve_ms, 1)}}

    answer = {"type": "answer", "answer": None, "error": None, "provider": name, "cited": [], "llm_s": None}
    if resolve_error:
        answer["error"] = resolve_error
    elif name == "none":
        answer["error"] = "No LLM is connected, so here are the most relevant parts of the code."
    else:
        prompt = answer_prompt(idx.meta["name"], idx.meta.get("commit"), idx.repo_map, hits, question, history)
        try:
            res = llm.complete(ANSWER_SYSTEM, prompt, name)
        except llm.LLMError as e:
            answer["error"] = str(e)
        else:
            answer["answer"] = res.text
            answer["llm_s"] = round(res.seconds, 2)
            valid = set(range(1, len(sources) + 1))
            answer["cited"] = sorted({int(m) for m in CITATION.findall(res.text)} & valid)
    yield answer


def ask(idx: RepoIndex, question: str, history: list[dict], provider: str | None = None,
        mode: str = "hybrid", k: int = 8) -> dict:
    first, second = list(ask_stream(idx, question, history, provider, mode, k))
    out = {k: v for k, v in first.items() if k != "type"}
    out.update({k: v for k, v in second.items() if k not in ("type", "llm_s")})
    out["timings"]["llm_s"] = second["llm_s"]
    return out


def overview(idx: RepoIndex, provider: str | None = None, refresh: bool = False) -> dict:
    try:
        name = llm.resolve(provider)
    except llm.LLMError as e:
        return {"markdown": fallback_overview(idx.meta["name"], idx.repo_map), "provider": "none", "error": str(e)}
    if name == "none":
        return {"markdown": fallback_overview(idx.meta["name"], idx.repo_map), "provider": "none", "error": None}
    cache = idx.overview_file(name)
    if cache.exists() and not refresh:
        return {"markdown": cache.read_text(encoding="utf-8"), "provider": name, "cached": True, "error": None}
    prompt = overview_prompt(idx.meta["name"], idx.repo_map, idx.repo_map.get("manifests", {}))
    try:
        res = llm.complete(OVERVIEW_SYSTEM, prompt, name)
    except llm.LLMError as e:
        return {"markdown": fallback_overview(idx.meta["name"], idx.repo_map), "provider": name, "error": str(e)}
    cache.write_text(res.text, encoding="utf-8")
    return {"markdown": res.text, "provider": name, "cached": False, "error": None, "llm_s": round(res.seconds, 2),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def read_file(idx: RepoIndex, path: str) -> dict:
    """Full text of one indexed file, for the code viewer. Only files that were indexed can be read."""
    if path not in idx.paths:
        raise FileNotFoundError(path)
    root = Path(idx.meta["root"]).resolve()
    full = (root / path).resolve()
    if root not in full.parents or not full.is_file():
        raise FileNotFoundError(path)
    text = full.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    if path.endswith(".ipynb"):
        text = notebook_to_text(text)
    lang = next((c.language for c in idx.chunks if c.path == path), "text")
    return {"path": path, "text": text, "language": lang, "url": source_url(idx, path)}

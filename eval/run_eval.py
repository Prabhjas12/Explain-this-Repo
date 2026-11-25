"""Measure how often retrieval finds the right file.

For each question in a dataset we know which file(s) hold the answer. We run the
question through keyword search, vector search and hybrid search, and record the
rank of the first result that comes from a correct file.

    hit@k  share of questions with a correct file in the top k results
    MRR    mean of 1/rank of the first correct result (0 if not in the top 10)

Usage (from the project root):
    .venv/bin/python eval/run_eval.py                     # all datasets, default embedder
    .venv/bin/python eval/run_eval.py eval/datasets/flask.json --embeddings wordllama

Retrieval only: no LLM is called, so this is free and takes a minute or two.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config  # noqa: E402
from app.embeddings import get_embedder  # noqa: E402
from app.indexer import build_index  # noqa: E402
from app.ingest import parse_source  # noqa: E402

# "hybrid, no prior" is plain Reciprocal Rank Fusion; "hybrid" adds the source-code preference
# (ETR_DOC_WEIGHT / ETR_TEST_WEIGHT), which is what the app uses.
MODES = ("keyword", "vector", "hybrid, no prior", "hybrid")


def matches(path: str, expected: list[str]) -> bool:
    return any(path.startswith(e) if e.endswith("/") else path == e for e in expected)


def evaluate(dataset: Path, embeddings: str, k_max: int = 10) -> dict:
    data = json.loads(dataset.read_text())
    embedder = get_embedder(embeddings)
    t0 = time.perf_counter()
    idx = build_index(parse_source(data["repo"]), embedder=embedder)
    index_s = time.perf_counter() - t0

    results = {}
    for mode in MODES:
        ranks, latencies, misses = [], [], []
        for item in data["questions"]:
            t = time.perf_counter()
            qv = None if mode == "keyword" else embedder.embed_query(item["q"])
            hits = idx.search(item["q"], qv, k=k_max, mode=mode.split(",")[0], use_prior=(mode == "hybrid"))
            latencies.append((time.perf_counter() - t) * 1000)
            rank = next((i for i, h in enumerate(hits, 1) if matches(h.chunk.path, item["expected"])), None)
            ranks.append(rank)
            if rank is None or rank > 5:
                misses.append({"q": item["q"], "got": [h.chunk.path for h in hits[:3]]})
        n = len(ranks)
        results[mode] = {
            "hit@1": sum(1 for r in ranks if r and r <= 1) / n,
            "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
            "hit@5": sum(1 for r in ranks if r and r <= 5) / n,
            "mrr": sum(1 / r for r in ranks if r) / n,
            "median_ms": statistics.median(latencies),
            "misses_top5": misses,
        }
    return {
        "dataset": dataset.name, "repo": data["repo"], "commit": idx.meta.get("commit"),
        "embedding": embedder.name, "questions": len(data["questions"]),
        "files": idx.meta["files"], "chunks": idx.meta["chunks"],
        "index_s": round(index_s, 2), "timings": idx.meta["timings"], "modes": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", type=Path)
    ap.add_argument("--embeddings", default=config.EMBEDDINGS, help="fastembed (default), openai, wordllama or hash")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    args = ap.parse_args()
    datasets = args.datasets or sorted((ROOT / "eval" / "datasets").glob("*.json"))
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"\nEmbeddings: {args.embeddings}\n")
    print(f"| Repo | Method | hit@1 | hit@3 | hit@5 | MRR | median query |")
    print(f"|---|---|---|---|---|---|---|")
    totals = {m: [] for m in MODES}
    for ds in datasets:
        r = evaluate(ds, args.embeddings)
        name = r["repo"].rsplit("github.com/", 1)[-1]
        for m in MODES:
            s = r["modes"][m]
            totals[m].append((s, r["questions"]))
            print(f"| {name} | {m} | {s['hit@1']:.0%} | {s['hit@3']:.0%} | {s['hit@5']:.0%} | {s['mrr']:.2f} | {s['median_ms']:.1f} ms |")
        (args.out / f"{ds.stem}.{args.embeddings}.json").write_text(json.dumps(r, indent=2))
        print(f"|  | indexed {r['files']} files into {r['chunks']} chunks in {r['index_s']} s |  |  |  |  |  |")
    if len(datasets) > 1:
        for m in MODES:
            n = sum(q for _, q in totals[m])
            avg = {key: sum(s[key] * q for s, q in totals[m]) / n for key in ("hit@1", "hit@3", "hit@5", "mrr")}
            print(f"| **all ({n} questions)** | {m} | {avg['hit@1']:.0%} | {avg['hit@3']:.0%} | {avg['hit@5']:.0%} | {avg['mrr']:.2f} |  |")
    print(f"\nDetails, including the questions each method missed: {args.out}/")


if __name__ == "__main__":
    main()

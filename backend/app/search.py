"""Step 4: find the chunks that answer a question.

Two retrievers, then a merge:

1. Vector search. Embed the question, take the dot product with every chunk vector
   (they're normalized, so that's cosine similarity), keep the highest. Good at
   meaning: "where do we retry failed requests" finds `max_retries` logic.
2. Keyword search (BM25). The classic search-engine formula: a chunk scores higher
   when it contains the question's rare words, many times, in a short chunk. Good at
   exact names: "what does `merge_setting` do" should return exactly that function,
   and embeddings are surprisingly bad at that. The tokenizer splits identifiers
   (`getUserName` -> get, user, name) so English questions can match code.
3. Hybrid (default): Reciprocal Rank Fusion. Each chunk gets 1/(60 + rank) from each
   list it appears in, and the totals are sorted. Rank-based fusion means we never
   have to compare a cosine score with a BM25 score, which live on different scales.

Two adjustments after fusion:
- A prior: prose documentation and tests get a bit less weight than source code.
  Docs are written in English, like questions, so they match unusually well, but the
  question is usually about how the code works (see eval/ for how this was measured).
- At most a few chunks per file are kept so one big file can't crowd out the rest.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk

RRF_K = 60
CANDIDATES = 50

STOPWORDS = set("""
a an the and or but if then else of to in on at by for from with without into onto over under about as is are was were be been
being do does did doing have has had having it its this that these those there here what which who whom whose where when why
how can could should would will shall may might must not no yes so than too very just also any all some each every both either
neither i me my we our you your he she they them their his her us one two get got make made use used using uses like want need
work works way thing things code repo repository project file files function functions method methods class
self cls this return import from def const let var function none null true false new nil
""".split())

_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SUFFIXES = ("ations", "ation", "ings", "ing", "ers", "er", "ions", "ion", "es", "ed", "s")


def stem(word: str) -> str:
    if len(word) > 4 and word.endswith(("ies", "ied")):
        return word[:-3] + "y"   # retries, retried -> retry
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            break
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Code-aware, lightly stemmed tokens.

    `HTTPAdapter.send_request` -> httpadapter, http, adapt, send_request, send, request
    """
    out = []
    for word in _IDENT.findall(text):
        lower = word.lower()
        pieces = [p for part in word.split("_") for p in _CAMEL.split(part) if p]
        if len(pieces) > 1:
            if lower not in STOPWORDS:
                out.append(lower)  # whole identifier, unstemmed, so exact names score highest
            for p in pieces:
                pl = p.lower()
                if len(pl) > 1 and pl not in STOPWORDS:
                    out.append(stem(pl))
        elif len(lower) > 1 and lower not in STOPWORDS:
            out.append(stem(lower))
    return out


def search_text(chunk: Chunk) -> str:
    # The path and symbol are repeated so they weigh more than a passing mention in a comment.
    head = f"{chunk.path} {chunk.label()}"
    return f"{head} {head} {chunk.text}"


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return scores
        for term in set(query_tokens):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            idx = np.fromiter((p[0] for p in postings), dtype=np.int64, count=len(postings))
            tf = np.fromiter((p[1] for p in postings), dtype=np.float32, count=len(postings))
            norm = self.k1 * (1 - self.b + self.b * self.lengths[idx] / (self.avg_len or 1.0))
            scores[idx] += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores


@dataclass
class Hit:
    index: int
    chunk: Chunk
    score: float
    vector_rank: int | None = None   # 1-based rank in vector search, if it was in the top candidates
    keyword_rank: int | None = None  # 1-based rank in BM25
    vector_score: float | None = None


def _top(scores: np.ndarray, k: int, positive_only: bool = False) -> list[int]:
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    if positive_only:
        idx = idx[scores[idx] > 0]
    return idx.tolist()


def diversify(hits: list[Hit], k: int, per_file: int) -> list[Hit]:
    out, counts = [], Counter()
    for h in hits:
        if counts[h.chunk.path] >= per_file:
            continue
        counts[h.chunk.path] += 1
        out.append(h)
        if len(out) == k:
            break
    return out


DOC_LANGUAGES = {"markdown", "rst", "text"}
_TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec|specs|testing)(/|$)|(^|/)test_[^/]*$|_test\.\w+$|\.(test|spec)\.\w+$")


def prior(chunk: Chunk, doc_weight: float, test_weight: float) -> float:
    """How much to trust a chunk before looking at the question. Prose docs match questions
    unusually well because both are written in English, so they get slightly less weight."""
    if chunk.language in DOC_LANGUAGES:
        return doc_weight
    if _TEST_PATH.search(chunk.path):
        return test_weight
    return 1.0


def retrieve(
    chunks: list[Chunk],
    vectors: np.ndarray | None,
    bm25: BM25,
    query: str,
    query_vector: np.ndarray | None,
    k: int = 8,
    mode: str = "hybrid",
    per_file: int = 3,
    doc_weight: float = 1.0,
    test_weight: float = 1.0,
) -> list[Hit]:
    vec_rank: dict[int, int] = {}
    vec_score: dict[int, float] = {}
    if mode in ("hybrid", "vector") and vectors is not None and query_vector is not None and len(chunks):
        sims = vectors @ query_vector
        for r, i in enumerate(_top(sims, CANDIDATES), 1):
            vec_rank[i] = r
            vec_score[i] = float(sims[i])

    kw_rank: dict[int, int] = {}
    if mode in ("hybrid", "keyword"):
        for r, i in enumerate(_top(bm25.scores(tokenize(query)), CANDIDATES, positive_only=True), 1):
            kw_rank[i] = r

    if mode == "vector":
        order = sorted(vec_rank, key=vec_rank.get)
        fused = {i: vec_score[i] for i in order}
    elif mode == "keyword":
        order = sorted(kw_rank, key=kw_rank.get)
        fused = {i: 1.0 / (RRF_K + kw_rank[i]) for i in order}
    else:
        fused = defaultdict(float)
        for i, r in vec_rank.items():
            fused[i] += 1.0 / (RRF_K + r)
        for i, r in kw_rank.items():
            fused[i] += 1.0 / (RRF_K + r)
        if doc_weight != 1.0 or test_weight != 1.0:
            for i in fused:
                fused[i] *= prior(chunks[i], doc_weight, test_weight)
        order = sorted(fused, key=lambda i: (-fused[i], min(vec_rank.get(i, 999), kw_rank.get(i, 999))))

    hits = [Hit(i, chunks[i], float(fused[i]), vec_rank.get(i), kw_rank.get(i), vec_score.get(i)) for i in order]
    return diversify(hits, k, per_file)

import numpy as np

from app.chunking import Chunk
from app.embeddings import HashEmbedder
from app.ingest import parse_source
from app.search import BM25, diversify, retrieve, search_text, stem, tokenize


def test_tokenize_splits_identifiers_and_keeps_whole_names():
    toks = tokenize("HTTPAdapter.send_request(max_retries)")
    assert "httpadapter" in toks and "http" in toks and "send_request" in toks and "request" in toks
    assert "max_retries" in toks and "retry" in toks


def test_stem_groups_word_forms():
    assert stem("parsing") == stem("parser") == stem("parse")
    assert stem("retries") == stem("retry")


def chunks():
    return [
        Chunk("src/auth.py", "python", 1, 10, "def check_password", "function", "", "def check_password(user, pw):\n    return hash(pw) == user.pw_hash"),
        Chunk("src/db.py", "python", 1, 10, "def connect", "function", "", "def connect(url):\n    return Engine(url)"),
        Chunk("src/routes.py", "python", 1, 10, "def login", "function", "", "@app.post('/login')\ndef login(form):\n    if check_password(form.user, form.pw): ..."),
        Chunk("README.md", "markdown", 1, 5, "# Demo", "section", "", "A demo app with a login page and a database."),
    ]


def test_bm25_prefers_rare_exact_terms():
    cs = chunks()
    bm = BM25([tokenize(search_text(c)) for c in cs])
    scores = bm.scores(tokenize("where is check_password defined"))
    assert int(np.argmax(scores)) == 0


def test_hybrid_fuses_both_rankings():
    cs = chunks()
    emb = HashEmbedder("hash-512")
    vecs = emb.embed_documents([c.embed_text() for c in cs])
    bm = BM25([tokenize(search_text(c)) for c in cs])
    q = "how does login check the password"
    hits = retrieve(cs, vecs, bm, q, emb.embed_query(q), k=3, mode="hybrid")
    assert {h.chunk.path for h in hits[:2]} == {"src/auth.py", "src/routes.py"}
    assert all(h.vector_rank or h.keyword_rank for h in hits)


def test_keyword_mode_returns_only_matches():
    cs = chunks()
    bm = BM25([tokenize(search_text(c)) for c in cs])
    hits = retrieve(cs, None, bm, "Engine", None, k=5, mode="keyword")
    assert [h.chunk.path for h in hits] == ["src/db.py"]


def test_diversify_caps_chunks_per_file():
    from app.search import Hit
    cs = [Chunk("a.py", "python", i, i, "", "code", "", "x") for i in range(1, 6)] + [Chunk("b.py", "python", 1, 1, "", "code", "", "y")]
    hits = [Hit(i, c, 1.0 / (i + 1)) for i, c in enumerate(cs)]
    out = diversify(hits, k=5, per_file=2)
    assert [h.chunk.path for h in out] == ["a.py", "a.py", "b.py"]


def test_parse_source_forms(tmp_path):
    s = parse_source("https://github.com/pallets/flask")
    assert (s.kind, s.name, s.repo_id) == ("github", "pallets/flask", "pallets__flask")
    assert parse_source("pallets/flask.git").repo_id == "pallets__flask"
    b = parse_source("https://github.com/psf/requests/tree/v2.31.0/src")
    assert b.branch == "v2.31.0" and b.repo_id.startswith("psf__requests@")
    local = parse_source(str(tmp_path))
    assert local.kind == "local" and local.local_path == tmp_path.resolve()


def test_prior_prefers_source_code_over_docs_and_tests():
    code = Chunk("src/router.py", "python", 1, 5, "def match_route", "function", "", "def match_route(url):\n    return table.get(url)")
    doc = Chunk("docs/routing.md", "markdown", 1, 5, "## Routing", "section", "",
                "Routing: how a URL is matched to a route. The router matches each url against every route.")
    test = Chunk("tests/test_router.py", "python", 1, 5, "def test_match_route", "function", "", "def test_match_route():\n    assert match_route('/')")
    cs = [doc, test, code]
    bm = BM25([tokenize(search_text(c)) for c in cs])
    q = "how is a url matched to a route"
    plain = retrieve(cs, None, bm, q, None, k=3, mode="hybrid")
    weighted = retrieve(cs, None, bm, q, None, k=3, mode="hybrid", doc_weight=0.8, test_weight=0.7)
    assert plain[0].chunk.path == "docs/routing.md"
    assert weighted[0].chunk.path == "src/router.py"

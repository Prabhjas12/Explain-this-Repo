import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import llm
from app.main import app

client = TestClient(app)

FILES = {
    "README.md": "# Tiny Shop\n\nA small web shop. Run `python app.py`.\n",
    "app.py": "from shop.cart import Cart\n\n\ndef main():\n    cart = Cart()\n    cart.add('apple', 2)\n    print(cart.total())\n\n\nif __name__ == '__main__':\n    main()\n",
    "shop/__init__.py": "",
    "shop/cart.py": (
        "PRICES = {'apple': 0.5, 'pear': 0.75}\n\n\n"
        "class Cart:\n    \"\"\"Items the customer wants to buy.\"\"\"\n\n"
        "    def __init__(self):\n        self.items = {}\n\n"
        "    def add(self, name, qty=1):\n        self.items[name] = self.items.get(name, 0) + qty\n\n"
        "    def total(self):\n        return sum(PRICES[n] * q for n, q in self.items.items())\n"
    ),
    "shop/discounts.py": "def apply_coupon(total, code):\n    \"\"\"10% off with SAVE10.\"\"\"\n    return total * 0.9 if code == 'SAVE10' else total\n",
}


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny-shop")
    for rel, text in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    r = client.post("/api/repos", json={"source": str(root)})
    assert r.status_code == 200, r.text
    repo_id = r.json()["repo_id"]
    wait_ready(repo_id)
    return root, repo_id


def wait_ready(repo_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/repos/{repo_id}/job").json()
        if job["stage"] == "ready":
            return job
        assert job["stage"] != "error", job
        time.sleep(0.05)
    raise AssertionError("indexing timed out")


def test_health():
    body = client.get("/api/health").json()
    assert body["ok"] and body["embedding"].startswith("hash")
    assert {p["id"] for p in body["llm"]["providers"]} >= {"claude", "codex", "none"}


def test_bad_source_is_a_400():
    assert client.post("/api/repos", json={"source": "not a repo!!"}).status_code == 400


def test_index_and_detail(repo):
    _, repo_id = repo
    listed = client.get("/api/repos").json()
    assert any(r["repo_id"] == repo_id for r in listed)
    d = client.get(f"/api/repos/{repo_id}").json()
    assert d["meta"]["files"] == 4  # the empty __init__.py is skipped
    assert "shop/cart.py" in d["files"]
    assert d["languages"][0]["language"] == "python"


def test_ask_without_llm_returns_sources(repo):
    _, repo_id = repo
    r = client.post(f"/api/repos/{repo_id}/ask", json={"question": "what does a pear cost?"}).json()
    assert r["answer"] is None and r["provider"] == "none" and "No LLM" in r["error"]
    assert r["sources"][0]["path"] == "shop/cart.py"
    assert r["sources"][0]["n"] == 1 and r["sources"][0]["start_line"] >= 1


def test_ask_with_llm_parses_citations(repo, monkeypatch):
    _, repo_id = repo
    seen = {}

    def fake(system, prompt, model, timeout):
        seen["prompt"] = prompt
        return "`Cart.total()` multiplies each quantity by its price [1]. Coupons are separate [9][2]."

    monkeypatch.setitem(llm.CALLS, "anthropic", fake)
    r = client.post(f"/api/repos/{repo_id}/ask", json={
        "question": "how is the total computed?", "provider": "anthropic",
        "history": [{"role": "user", "content": "what is this?"}, {"role": "assistant", "content": "A shop."}],
    }).json()
    assert r["error"] is None and r["provider"] == "anthropic"
    assert r["cited"] == [1, 2]  # [9] doesn't exist, so it's dropped
    assert "shop/cart.py (lines" in seen["prompt"] and "Conversation so far" in seen["prompt"]


def test_follow_up_question_borrows_previous_subject(repo):
    _, repo_id = repo
    r = client.post(f"/api/repos/{repo_id}/ask", json={
        "question": "where is it called?", "history": [{"role": "user", "content": "what does apply_coupon do?"}],
    }).json()
    assert "apply_coupon" in r["query"]


def test_llm_failure_still_returns_sources(repo, monkeypatch):
    _, repo_id = repo

    def broken(*a):
        raise llm.LLMError("Claude Code failed: not logged in")

    monkeypatch.setitem(llm.CALLS, "claude", broken)
    r = client.post(f"/api/repos/{repo_id}/ask", json={"question": "coupon codes", "provider": "claude"}).json()
    assert r["answer"] is None and "not logged in" in r["error"] and r["sources"]


def test_search_modes(repo):
    _, repo_id = repo
    for mode in ("hybrid", "vector", "keyword"):
        r = client.post(f"/api/repos/{repo_id}/search", json={"query": "apply_coupon SAVE10", "mode": mode}).json()
        assert r["results"][0]["path"] == "shop/discounts.py", mode


def test_overview_fallback_and_cache(repo, monkeypatch):
    _, repo_id = repo
    o = client.get(f"/api/repos/{repo_id}/overview").json()
    assert o["provider"] == "none" and "File tree" in o["markdown"]
    calls = []
    monkeypatch.setitem(llm.CALLS, "openai", lambda *a: calls.append(1) or "## What it is\nA shop.")
    first = client.get(f"/api/repos/{repo_id}/overview", params={"provider": "openai"}).json()
    second = client.get(f"/api/repos/{repo_id}/overview", params={"provider": "openai"}).json()
    assert first["markdown"].startswith("## What it is") and second["cached"] and len(calls) == 1


def test_file_endpoint_only_serves_indexed_files(repo):
    _, repo_id = repo
    ok = client.get(f"/api/repos/{repo_id}/file", params={"path": "shop/cart.py"})
    assert ok.status_code == 200 and "class Cart" in ok.json()["text"]
    for bad in ("../../../etc/passwd", "/etc/passwd", "shop/../../secret.txt", "missing.py"):
        assert client.get(f"/api/repos/{repo_id}/file", params={"path": bad}).status_code == 404


def test_reindex_reuses_unchanged_vectors(repo):
    root, repo_id = repo
    (root / "shop/discounts.py").write_text(FILES["shop/discounts.py"] + "\n\ndef free_shipping(total):\n    return total > 50\n")
    client.post("/api/repos", json={"source": str(root)})
    wait_ready(repo_id)
    meta = client.get(f"/api/repos/{repo_id}").json()["meta"]
    assert 0 < meta["reused_vectors"] < meta["chunks"]


def test_streaming_sends_sources_before_answer(repo):
    _, repo_id = repo
    with client.stream("POST", f"/api/repos/{repo_id}/ask", params={"stream": "true"},
                       json={"question": "coupon"}) as r:
        import json
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert [e["type"] for e in events] == ["sources", "answer"]
    assert events[0]["sources"] and events[1]["provider"] == "none"



def test_delete_keeps_local_folder(repo):
    root, repo_id = repo
    assert client.delete(f"/api/repos/{repo_id}").status_code == 200
    assert client.get(f"/api/repos/{repo_id}").status_code == 404
    assert (root / "shop/cart.py").exists()

"""Settings, read once from environment variables (or a .env file loaded by run.sh).

Every setting has a working default, so the app runs with no configuration.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("ETR_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
REPOS_DIR = DATA_DIR / "repos"      # git clones
INDEX_DIR = DATA_DIR / "indexes"    # one folder per indexed repo
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

# Embeddings: fastembed (local ONNX model, default) | openai | wordllama | hash (tests only)
EMBEDDINGS = os.environ.get("ETR_EMBEDDINGS", "fastembed").strip().lower()
EMBEDDING_MODEL = os.environ.get("ETR_EMBEDDING_MODEL", "").strip()

# LLM: auto | claude | codex | anthropic | openai | ollama | none
LLM_PROVIDER = os.environ.get("ETR_LLM", "auto").strip().lower()
LLM_MODEL = os.environ.get("ETR_LLM_MODEL", "").strip()
LLM_TIMEOUT_S = int(os.environ.get("ETR_LLM_TIMEOUT", "240"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST

# Ingestion limits
MAX_FILES = int(os.environ.get("ETR_MAX_FILES", "5000"))
MAX_FILE_BYTES = int(os.environ.get("ETR_MAX_FILE_BYTES", "400000"))
MAX_DATA_FILE_BYTES = int(os.environ.get("ETR_MAX_DATA_FILE_BYTES", "100000"))
CLONE_TIMEOUT_S = int(os.environ.get("ETR_CLONE_TIMEOUT", "600"))

# Retrieval
TOP_K = int(os.environ.get("ETR_TOP_K", "8"))
# Hybrid search slightly prefers source code over prose docs and tests. Chosen with eval/run_eval.py:
# on 48 questions this moved the right file to rank 1 for 75% of questions instead of 33%,
# without hurting questions that docs should answer ("how do I install it?"). 1.0 turns it off.
DOC_WEIGHT = float(os.environ.get("ETR_DOC_WEIGHT", "0.8"))
TEST_WEIGHT = float(os.environ.get("ETR_TEST_WEIGHT", "0.7"))
MAX_CHUNKS_PER_FILE = int(os.environ.get("ETR_MAX_CHUNKS_PER_FILE", "3"))


def ensure_dirs() -> None:
    for d in (DATA_DIR, REPOS_DIR, INDEX_DIR):
        d.mkdir(parents=True, exist_ok=True)

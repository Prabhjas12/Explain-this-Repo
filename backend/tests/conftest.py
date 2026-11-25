"""Tests run with no downloads and no LLM: a hashing 'embedder' and a temp data folder."""
import os
import tempfile

os.environ.setdefault("ETR_DATA_DIR", tempfile.mkdtemp(prefix="etr-test-"))
os.environ["ETR_EMBEDDINGS"] = "hash"
os.environ["ETR_LLM"] = "none"

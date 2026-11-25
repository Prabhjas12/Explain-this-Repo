from pathlib import Path

import pytest

from app.chunking import MAX_CHARS, chunk_file
from app.ingest import SourceFile, detect_language, notebook_to_text


def make(path: str, text: str) -> SourceFile:
    return SourceFile(path, detect_language(path) or "text", text, len(text), text.count("\n") + 1)


def covered(sf: SourceFile, chunks) -> bool:
    lines = sf.text.split("\n")
    cov = set()
    for c in chunks:
        cov.update(range(c.start_line, c.end_line + 1))
    return all(i in cov for i, line in enumerate(lines, 1) if line.strip())


PY = '''"""Module docstring."""
import os
import sys

LIMIT = 10


# Helper that adds things.
@cache
def add(a, b):
    return a + b


class Store:
    """A store."""

    def get(self, key):
        return self.data[key]
''' + "\n".join(f"    def m{i}(self):\n" + "\n".join(f"        x{j} = {j} * {i}" for j in range(12)) + "\n        return x0\n" for i in range(8))


def test_python_chunks_follow_functions_and_methods():
    sf = make("pkg/store.py", PY)
    chunks = chunk_file(sf)
    labels = [c.symbol for c in chunks]
    assert any("def add" in s for s in labels)
    method = next(c for c in chunks if "def m3(self)" in c.text)
    assert method.context.startswith("class Store")
    assert method.symbol.startswith("Store.")
    # decorator and the comment above it belong to the function, not the previous chunk
    add = next(c for c in chunks if "def add" in c.symbol)
    assert "# Helper that adds things." in add.text and "@cache" in add.text
    assert covered(sf, chunks)


def test_line_numbers_match_text():
    sf = make("pkg/store.py", PY)
    lines = sf.text.split("\n")
    for c in chunk_file(sf):
        assert c.text == "\n".join(lines[c.start_line - 1:c.end_line])


JS = """import express from 'express';
import { join } from 'path';

// Start the server.
export function start(port) {
  const app = express();
  app.listen(port);
  return app;
}

export const handler = async (req, res) => {
  res.send('ok');
};

class Router {
  constructor() {
    this.routes = [];
  }
}
"""


def test_js_splits_at_top_level_declarations():
    sf = make("src/server.js", JS)
    chunks = chunk_file(sf)
    assert covered(sf, chunks)
    joined = " | ".join(c.symbol for c in chunks)
    assert "function start" in joined and "handler()" in joined and "class Router" in joined
    start = next(c for c in chunks if "function start" in c.symbol)
    assert "// Start the server." in start.text  # leading comment attached


def test_big_brace_block_is_split_one_level_in():
    body = "\n".join(
        f"  public int method{i}(int x) {{\n" + "\n".join(f"    int v{j} = x * {j};" for j in range(10)) + "\n    return x;\n  }\n"
        for i in range(12)
    )
    sf = make("src/Big.java", f"public class Big {{\n{body}}}\n")
    chunks = chunk_file(sf)
    assert len(chunks) >= 3  # small methods are packed a few per chunk
    assert all(len(c.text) <= MAX_CHARS for c in chunks)
    m5 = next(c for c in chunks if "method5(int x)" in c.text)
    assert m5.symbol.startswith("method") and m5.context == "public class Big {"
    assert all(c.context == "public class Big {" for c in chunks)
    assert covered(sf, chunks)


def test_markdown_splits_at_headings():
    md = "# Title\nintro\n\n## Install\n" + "pip install x\n" * 40 + "\n## Usage\n```\n# not a heading\n```\nrun it\n"
    chunks = chunk_file(make("README.md", md))
    symbols = [c.symbol for c in chunks]
    assert any("## Usage" in s for s in symbols)
    assert not any("not a heading" in s for s in symbols)


def test_long_unstructured_file_uses_overlapping_windows():
    text = "\n".join(f"key{i}: value{i}" for i in range(400))
    sf = make("config/big.yaml", text)
    chunks = chunk_file(sf)
    assert len(chunks) > 3
    assert chunks[1].start_line < chunks[0].end_line  # overlap
    assert covered(sf, chunks)


def test_syntax_error_python_falls_back():
    sf = make("broken.py", "def ok():\n    return 1\n\ndef broken(:\n    pass\n")
    chunks = chunk_file(sf)
    assert chunks and covered(sf, chunks)


def test_notebook_keeps_code_and_markdown_drops_outputs():
    nb = '{"cells": [{"cell_type": "markdown", "source": ["# Analysis"]}, {"cell_type": "code", "source": ["df = load()"], "outputs": [{"text": "HUGE OUTPUT"}]}]}'
    text = notebook_to_text(nb)
    assert "df = load()" in text and "# Analysis" in text and "HUGE OUTPUT" not in text


@pytest.mark.parametrize("path", sorted(str(p) for p in Path(__file__).resolve().parents[1].joinpath("app").glob("*.py")))
def test_every_line_of_this_project_is_covered(path):
    text = Path(path).read_text()
    sf = make(Path(path).name, text)
    chunks = chunk_file(sf)
    assert covered(sf, chunks)
    assert all(len(c.text) <= MAX_CHARS or c.start_line == c.end_line for c in chunks)

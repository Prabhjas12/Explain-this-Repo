"""A structural summary of the repo: file tree, languages, key files, and an outline of
the most important files. Retrieval finds details; this gives the LLM the big picture,
which is what a question like "what does this repo do?" actually needs.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .chunking import Chunk
from .ingest import DATA_LANGUAGES, DOC_LANGUAGES, SourceFile

KEY_FILES = [
    "README.md", "README.rst", "README.txt", "README", "package.json", "pyproject.toml", "setup.py",
    "setup.cfg", "requirements.txt", "environment.yml", "Pipfile", "go.mod", "Cargo.toml", "pom.xml",
    "build.gradle", "build.gradle.kts", "Gemfile", "composer.json", "Dockerfile", "docker-compose.yml",
    "docker-compose.yaml", "compose.yaml", "Makefile", "CMakeLists.txt", "tsconfig.json",
    "vite.config.ts", "vite.config.js", "next.config.js", "next.config.mjs", "manage.py", "Procfile",
]
ENTRY_POINTS = {
    "main.py", "app.py", "__main__.py", "cli.py", "server.py", "manage.py", "wsgi.py", "asgi.py",
    "index.js", "index.ts", "main.js", "main.ts", "main.tsx", "index.tsx", "App.tsx", "App.jsx", "App.vue",
    "server.js", "server.ts", "app.js", "app.ts", "main.go", "main.rs", "lib.rs", "Main.java",
    "Application.java", "Program.cs", "main.c", "main.cpp", "main.swift", "main.dart",
}
SOURCE_DIRS = {"src", "app", "lib", "pkg", "cmd", "internal", "backend", "server", "core", "api", "frontend"}


def _is_test(path: str) -> bool:
    p = path.lower()
    return any(seg in ("test", "tests", "__tests__", "spec", "specs", "testing", "e2e") for seg in p.split("/")) or \
        p.rsplit("/", 1)[-1].startswith("test_") or ".test." in p or ".spec." in p or p.endswith("_test.go")


def render_tree(paths: list[str], max_depth: int = 3, per_dir: int = 15, max_lines: int = 160) -> str:
    def node():
        return {"dirs": {}, "files": []}

    root = node()
    for p in paths:
        cur = root
        parts = p.split("/")
        for d in parts[:-1]:
            cur = cur["dirs"].setdefault(d, node())
        cur["files"].append(parts[-1])

    def count(n) -> int:
        return len(n["files"]) + sum(count(c) for c in n["dirs"].values())

    lines: list[str] = []

    def walk(n, prefix: str, depth: int):
        for name in sorted(n["dirs"]):
            child = n["dirs"][name]
            lines.append(f"{prefix}{name}/ ({count(child)} files)")
            if depth < max_depth:
                walk(child, prefix + "  ", depth + 1)
        files = sorted(n["files"])
        for f in files[:per_dir]:
            lines.append(prefix + f)
        if len(files) > per_dir:
            lines.append(f"{prefix}... {len(files) - per_dir} more files")

    walk(root, "", 1)
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    return "\n".join(lines)


def build_repo_map(files: list[SourceFile], chunks: list[Chunk]) -> dict:
    symbols: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        for s in (c.symbol or "").split(", "):
            s = s.split(" (part")[0].strip()
            if s and " more" not in s and s not in symbols[c.path] and c.kind not in ("section",):
                symbols[c.path].append(s)

    lang_lines: Counter = Counter()
    for f in files:
        lang_lines[f.language] += f.lines
    code_langs = {k: v for k, v in lang_lines.items() if k not in DOC_LANGUAGES and k not in DATA_LANGUAGES}
    total_code = sum(code_langs.values()) or 1

    by_name = {f.path: f for f in files}
    key_files = [p for p in by_name if p.count("/") <= 1 and p.rsplit("/", 1)[-1] in KEY_FILES]
    readme = next((by_name[p] for p in ("README.md", "README.rst", "README.txt", "README", "readme.md") if p in by_name), None)

    def importance(f: SourceFile) -> float:
        name = f.path.rsplit("/", 1)[-1]
        parts = f.path.split("/")
        score = 0.0
        if f.language in DOC_LANGUAGES or f.language in DATA_LANGUAGES:
            return -1.0
        if name in ENTRY_POINTS:
            score += 5
        if parts[0] in SOURCE_DIRS or (len(parts) > 1 and parts[1] in SOURCE_DIRS):
            score += 2
        score += min(len(symbols.get(f.path, [])), 15) / 3
        score += min(f.lines, 600) / 300
        score -= 0.6 * max(0, len(parts) - 3)
        if _is_test(f.path):
            score -= 4
        if name.startswith("__init__") and f.lines < 20:
            score -= 3
        return score

    ranked = sorted(files, key=importance, reverse=True)
    important = [f.path for f in ranked if importance(f) > 0][:30]
    entry_points = [f.path for f in files if f.path.rsplit("/", 1)[-1] in ENTRY_POINTS and not _is_test(f.path)][:15]

    return {
        "files": [{"path": f.path, "language": f.language, "lines": f.lines} for f in files],
        "tree": render_tree([f.path for f in files]),
        "languages": [
            {"language": k, "lines": v, "percent": round(100 * v / total_code, 1)}
            for k, v in sorted(code_langs.items(), key=lambda kv: -kv[1])
        ],
        "key_files": key_files,
        "entry_points": entry_points,
        "important_files": [{"path": p, "symbols": symbols.get(p, [])[:25]} for p in important],
        "manifests": {p: by_name[p].text[:2000] for p in key_files if not p.rsplit("/", 1)[-1].startswith("README")},
        "readme_path": readme.path if readme else None,
        "readme": readme.text[:8000] if readme else "",
        "test_files": sum(1 for f in files if _is_test(f.path)),
    }

"""Step 1 of the pipeline: get the repository onto disk and decide which files to read.

A source can be a GitHub URL (https://github.com/owner/repo, optionally /tree/<branch>),
the short form owner/repo, or a folder on this computer.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import config

GITHUB_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?"
    r"(?:/tree/([^/?#]+)[^?#]*)?/?(?:[?#].*)?$"
)
SHORT_FORM = re.compile(r"^([\w.-]+)/([\w.-]+?)(?:\.git)?$")

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "bower_components", "dist", "build", "out",
    ".venv", "venv", "env", ".env", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", ".nox", ".next", ".nuxt", ".svelte-kit", ".turbo", ".parcel-cache", "target",
    "vendor", "coverage", "htmlcov", ".idea", ".vscode", "site-packages", ".gradle", "Pods",
    "DerivedData", ".terraform", ".cache", ".ipynb_checkpoints", ".eggs", "eggs", "wheels",
}

SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Pipfile.lock",
    "Cargo.lock", "composer.lock", "Gemfile.lock", "go.sum", "uv.lock", "bun.lockb",
    "npm-shrinkwrap.json", ".DS_Store", "Thumbs.db",
}

# Release notes mention every feature by name but explain none of them; they swamp search results.
CHANGELOG = re.compile(r"^(CHANGELOG|CHANGES|HISTORY|NEWS|RELEASES?)(\.\w+)?$", re.IGNORECASE)

BINARY_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".icns", ".svg", ".webp", ".tif", ".tiff",
    ".pdf", ".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z", ".rar", ".jar", ".war", ".class",
    ".o", ".a", ".so", ".dylib", ".dll", ".exe", ".bin", ".dat", ".db", ".sqlite", ".sqlite3",
    ".pyc", ".pyo", ".pyd", ".whl", ".egg", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".wav", ".mov", ".avi", ".ogg", ".flac", ".webm", ".m4a", ".mkv",
    ".psd", ".ai", ".sketch", ".fig", ".xd", ".blend", ".npy", ".npz", ".pkl", ".pickle",
    ".joblib", ".h5", ".hdf5", ".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".tflite",
    ".parquet", ".feather", ".arrow", ".avro", ".orc", ".csv", ".tsv", ".jsonl", ".ndjson",
    ".xlsx", ".xls", ".docx", ".doc", ".pptx", ".ppt", ".key", ".numbers", ".pages",
    ".lock", ".map", ".snap", ".log",
}
DATA_EXT = {".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".txt"}

LANGUAGES = {
    ".py": "python", ".pyi": "python", ".ipynb": "notebook",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala", ".groovy": "groovy",
    ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".hh": "cpp", ".cs": "csharp", ".swift": "swift", ".m": "objectivec",
    ".mm": "objectivec", ".rb": "ruby", ".php": "php", ".lua": "lua", ".r": "r", ".R": "r",
    ".jl": "julia", ".dart": "dart", ".ex": "elixir", ".exs": "elixir", ".erl": "erlang",
    ".hs": "haskell", ".ml": "ocaml", ".clj": "clojure", ".sql": "sql", ".sh": "bash",
    ".bash": "bash", ".zsh": "bash", ".fish": "bash", ".ps1": "powershell", ".pl": "perl",
    ".vue": "vue", ".svelte": "svelte", ".html": "html", ".htm": "html", ".css": "css",
    ".scss": "scss", ".sass": "scss", ".less": "less", ".md": "markdown", ".mdx": "markdown",
    ".rst": "rst", ".tex": "latex", ".proto": "protobuf", ".graphql": "graphql", ".gql": "graphql",
    ".tf": "terraform", ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".xml": "xml", ".ini": "ini", ".cfg": "ini", ".conf": "ini", ".txt": "text",
    ".gradle": "groovy", ".cmake": "cmake", ".mk": "makefile",
}
SPECIAL_FILENAMES = {
    "Dockerfile": "dockerfile", "Makefile": "makefile", "CMakeLists.txt": "cmake",
    "Gemfile": "ruby", "Rakefile": "ruby", "Procfile": "text", "Jenkinsfile": "groovy",
    ".gitignore": "text", ".dockerignore": "text", ".env.example": "text", "LICENSE": "text",
}

DOC_LANGUAGES = {"markdown", "rst", "text"}
DATA_LANGUAGES = {"json", "yaml", "toml", "xml", "ini"}


@dataclass
class RepoSource:
    kind: str                 # "github" or "local"
    repo_id: str              # safe folder name, e.g. "pallets__flask"
    name: str                 # display name, e.g. "pallets/flask"
    clone_url: str | None = None
    branch: str | None = None
    local_path: Path | None = None
    web_url: str | None = None  # https://github.com/owner/repo


@dataclass
class SourceFile:
    path: str                 # relative, forward slashes
    language: str
    text: str
    size: int
    lines: int


@dataclass
class ScanResult:
    files: list[SourceFile] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


class IngestError(Exception):
    pass


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-").lower()


def parse_source(text: str) -> RepoSource:
    text = (text or "").strip()
    if not text:
        raise IngestError("Paste a GitHub URL (or owner/repo) or a folder path.")

    expanded = Path(text).expanduser()
    if text.startswith(("/", "~", ".")) or (expanded.is_absolute() and expanded.exists()):
        if not expanded.is_dir():
            raise IngestError(f"Folder not found: {text}")
        path = expanded.resolve()
        digest = hashlib.sha1(str(path).encode()).hexdigest()[:8]
        return RepoSource("local", f"local__{_safe(path.name)}__{digest}", path.name, local_path=path)

    m = GITHUB_URL.match(text) or SHORT_FORM.match(text)
    if not m:
        raise IngestError("That doesn't look like a GitHub repo. Try https://github.com/owner/repo or owner/repo.")
    owner, repo = m.group(1), m.group(2)
    branch = m.group(3) if m.re is GITHUB_URL else None
    repo_id = f"{_safe(owner)}__{_safe(repo)}" + (f"@{_safe(branch)}" if branch else "")
    return RepoSource(
        "github", repo_id, f"{owner}/{repo}",
        clone_url=f"https://github.com/{owner}/{repo}.git", branch=branch,
        web_url=f"https://github.com/{owner}/{repo}",
    )


def _git(args: list[str], cwd: Path | None = None, timeout: int = 120) -> str:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
    try:
        proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    except FileNotFoundError as e:
        raise IngestError("git isn't installed. On a Mac, run: xcode-select --install") from e
    except subprocess.TimeoutExpired as e:
        raise IngestError(f"git {args[0]} took longer than {timeout}s.") from e
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip().splitlines()
        tail = msg[-1] if msg else "unknown error"
        if "could not read Username" in proc.stderr or "Repository not found" in proc.stderr:
            tail = "Repository not found, or it's private. Private repos: clone it yourself and paste the folder path."
        raise IngestError(f"git {args[0]} failed: {tail}")
    return proc.stdout.strip()


def fetch(source: RepoSource) -> tuple[Path, str | None]:
    """Clone (or update) a GitHub repo. Returns (folder, commit sha)."""
    if source.kind == "local":
        path = source.local_path
        assert path is not None
        sha = None
        if (path / ".git").exists():
            try:
                sha = _git(["rev-parse", "HEAD"], cwd=path)
            except IngestError:
                sha = None
        return path, sha

    dest = config.REPOS_DIR / source.repo_id
    ref = source.branch or "HEAD"
    if (dest / ".git").exists():
        try:
            _git(["fetch", "--depth", "1", "origin", ref], cwd=dest, timeout=config.CLONE_TIMEOUT_S)
            _git(["reset", "--hard", "FETCH_HEAD"], cwd=dest)
            _git(["clean", "-fdx"], cwd=dest)
            return dest, _git(["rev-parse", "HEAD"], cwd=dest)
        except IngestError:
            import shutil
            shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = ["clone", "--depth", "1", "--single-branch"]
    if source.branch:
        args += ["--branch", source.branch]
    _git([*args, source.clone_url, str(dest)], timeout=config.CLONE_TIMEOUT_S)
    return dest, _git(["rev-parse", "HEAD"], cwd=dest)


def detect_language(rel_path: str) -> str | None:
    name = rel_path.rsplit("/", 1)[-1]
    if name in SPECIAL_FILENAMES:
        return SPECIAL_FILENAMES[name]
    if name.startswith("README") or name.startswith("CONTRIBUTING") or name.startswith("CHANGELOG"):
        return "markdown" if not name.endswith(".rst") else "rst"
    ext = os.path.splitext(name)[1]
    return LANGUAGES.get(ext) or LANGUAGES.get(ext.lower())


def _list_paths(root: Path) -> list[str]:
    """Files to consider. Uses git's file list when available (respects .gitignore)."""
    if (root / ".git").exists():
        try:
            out = _git(["ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root)
            return sorted(p for p in out.split("\0") if p)
        except IngestError:
            pass
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for f in sorted(filenames):
            paths.append(os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, "/"))
    return paths


def notebook_to_text(raw: str) -> str:
    """Keep a notebook's markdown and code cells, drop outputs."""
    nb = json.loads(raw)
    parts = []
    for i, cell in enumerate(nb.get("cells", []), 1):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        if not src.strip():
            continue
        kind = cell.get("cell_type", "code")
        if kind == "markdown":
            body = "\n".join("# " + line if line else "#" for line in src.splitlines())
            parts.append(f"# %% [markdown] cell {i}\n{body}")
        elif kind == "code":
            parts.append(f"# %% cell {i}\n{src}")
    return "\n\n".join(parts) + "\n"


def _looks_minified(text: str) -> bool:
    if len(text) < 5000:
        return False
    newlines = text.count("\n")
    return newlines == 0 or len(text) / max(newlines, 1) > 300


def scan(root: Path) -> ScanResult:
    result = ScanResult()
    for rel in _list_paths(root):
        parts = rel.split("/")
        name = parts[-1]
        if any(p in SKIP_DIRS for p in parts[:-1]):
            result.skip("ignored folder")
            continue
        if name in SKIP_FILES or name.endswith((".min.js", ".min.css", ".bundle.js")) or CHANGELOG.match(name):
            result.skip("lockfile or generated")
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in BINARY_EXT:
            result.skip("binary or data")
            continue
        lang = detect_language(rel)
        if lang is None:
            result.skip("unknown type")
            continue
        full = root / rel
        try:
            if full.is_symlink() or not full.is_file():
                continue
            size = full.stat().st_size
        except OSError:
            continue
        limit = config.MAX_DATA_FILE_BYTES if ext in DATA_EXT else config.MAX_FILE_BYTES
        if lang == "notebook":
            limit = config.MAX_FILE_BYTES * 10  # outputs inflate notebooks; we drop them
        if size == 0 or size > limit:
            result.skip("empty or too large")
            continue
        raw = full.read_bytes()
        if b"\0" in raw[:8192]:
            result.skip("binary or data")
            continue
        text = raw.decode("utf-8", errors="replace")
        if lang == "notebook":
            try:
                text = notebook_to_text(text)
            except (ValueError, TypeError):
                result.skip("unreadable notebook")
                continue
            lang = "python"
        if _looks_minified(text):
            result.skip("lockfile or generated")
            continue
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        result.files.append(SourceFile(rel, lang, text, size, text.count("\n") + 1))
        if len(result.files) >= config.MAX_FILES:
            result.skip(f"over the {config.MAX_FILES}-file limit")
            break
    return result

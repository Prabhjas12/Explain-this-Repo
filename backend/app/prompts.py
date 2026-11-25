"""Step 5: the prompts. This is the "augmented generation" half of RAG.

The LLM never sees the whole repo. It sees the question plus the handful of chunks
retrieval picked, each numbered, and it's told to cite those numbers. Citations do
two jobs: the user can click through to check a claim, and an answer with no
citations is a visible sign that retrieval missed.
"""
from __future__ import annotations

from .search import Hit

ANSWER_SYSTEM = """You are Explain This Repo, an assistant that explains a software repository to a developer who is new to it.

Rules:
- Use only the numbered code snippets and repository notes provided. Don't invent files, functions, settings or behavior.
- Cite the snippets behind each claim with their numbers in square brackets, like [2] or [1][3]. Any sentence or bullet about how the code works needs a citation.
- Name files, functions and classes in backticks, e.g. `src/app.py` or `parse_args()`.
- If the snippets aren't enough to answer, say so plainly, answer the part you can, and name the files most likely to have the rest. Never guess.
- Explain the flow and the reasons, not just a list of names. Start with a one- or two-sentence direct answer, then details.
- Keep it under about 350 words unless the question needs more. Short code excerpts (under 15 lines) are fine when they help."""

OVERVIEW_SYSTEM = """You are Explain This Repo. Write an orientation guide for a developer who has never seen this repository and wants to understand it fast.

Use only the material provided: the README, file tree, manifests and file outlines. Don't invent features. When something is uncertain, say "appears to".
Name files and folders in backticks with their full path from the repo root, e.g. `src/server/app.py`.

Use exactly these sections, as markdown headings:
## What it is
Two or three sentences: what the project does and who it's for.
## How it's organized
The main folders and what lives in each, as a short bullet list.
## How it works
The main flow from input to output (for example request -> router -> handler -> database), naming the files involved. Use a numbered list.
## Where to start reading
Three to six files, in the order a newcomer should read them, each with one line on why.
## Tech stack
Languages, frameworks and notable libraries, one line.
## Running it
How to install and run it, if the material shows that. Otherwise say it isn't documented.

Be concrete and skimmable. Aim for 350 to 550 words."""


def _fence(lang: str) -> str:
    return {"notebook": "python", "text": ""}.get(lang, lang)


def format_snippets(hits: list[Hit]) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        c = h.chunk
        label = f" · {c.label()}" if c.label() else ""
        blocks.append(f"[{n}] {c.path} (lines {c.start_line}-{c.end_line}){label}\n```{_fence(c.language)}\n{c.text}\n```")
    return "\n\n".join(blocks)


def short_tree(tree: str, max_lines: int = 45) -> str:
    lines = [l for l in tree.splitlines() if len(l) - len(l.lstrip(" ")) <= 2]  # top two levels
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    return "\n".join(lines)


def answer_prompt(repo_name: str, commit: str | None, repo_map: dict, hits: list[Hit],
                  question: str, history: list[dict]) -> str:
    parts = [f"Repository: {repo_name}" + (f" (commit {commit[:7]})" if commit else "")]
    if repo_map.get("tree"):
        parts.append("Top-level layout:\n" + short_tree(repo_map["tree"]))
    if history:
        turns = []
        for t in history[-6:]:
            role = "User" if t.get("role") == "user" else "Assistant"
            text = str(t.get("content", ""))
            turns.append(f"{role}: {text[:1500]}{'...' if len(text) > 1500 else ''}")
        parts.append("Conversation so far (for context; cite only the snippets below):\n" + "\n".join(turns))
    parts.append("Code snippets:\n\n" + (format_snippets(hits) if hits else "(none matched)"))
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def overview_prompt(repo_name: str, repo_map: dict, manifests: dict[str, str]) -> str:
    parts = [f"Repository: {repo_name}"]
    langs = ", ".join(f"{l['language']} {l['percent']}%" for l in repo_map.get("languages", [])[:6])
    if langs:
        parts.append(f"Languages by lines of code: {langs}")
    if repo_map.get("readme"):
        parts.append(f"README ({repo_map.get('readme_path')}):\n" + repo_map["readme"][:6000])
    parts.append("File tree:\n" + repo_map.get("tree", ""))
    for path, text in manifests.items():
        parts.append(f"{path}:\n```\n{text[:1800]}\n```")
    outline = []
    for f in repo_map.get("important_files", [])[:25]:
        syms = ", ".join(f["symbols"][:15])
        outline.append(f"- {f['path']}" + (f": {syms}" if syms else ""))
    if outline:
        parts.append("Most important source files and what they define:\n" + "\n".join(outline))
    if repo_map.get("entry_points"):
        parts.append("Likely entry points: " + ", ".join(repo_map["entry_points"]))
    return "\n\n".join(parts)


def fallback_overview(repo_name: str, repo_map: dict) -> str:
    """What the Overview tab shows when no LLM is set up: the facts, no prose."""
    lines = [f"## {repo_name}", "", "_No LLM is connected, so this is the raw structure. Connect one (see Settings) for a written explanation._", ""]
    langs = repo_map.get("languages", [])
    if langs:
        lines += ["## Languages", ", ".join(f"{l['language']} ({l['percent']}%)" for l in langs[:8]), ""]
    if repo_map.get("entry_points"):
        lines += ["## Likely entry points"] + [f"- `{p}`" for p in repo_map["entry_points"]] + [""]
    if repo_map.get("important_files"):
        lines += ["## Most important files"]
        for f in repo_map["important_files"][:12]:
            syms = ", ".join(f"`{s}`" for s in f["symbols"][:6])
            lines.append(f"- `{f['path']}`" + (f": {syms}" if syms else ""))
        lines.append("")
    lines += ["## File tree", "```", repo_map.get("tree", ""), "```"]
    return "\n".join(lines)

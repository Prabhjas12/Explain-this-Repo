"""Step 6: send the prompt to a language model.

Several ways to reach one, so it works with whatever you already have:

- claude  : Claude Code CLI, signed in with your Claude subscription (no API key)
- codex   : Codex CLI, signed in with your ChatGPT subscription (no API key)
- anthropic / openai : API keys (ANTHROPIC_API_KEY / OPENAI_API_KEY)
- ollama  : a free local model (ollama serve)
- none    : retrieval only; the app still shows the matching code

"auto" picks the first one that's available, in that order.

The CLIs run with their built-in tools switched off, inside an empty temporary
folder, and with API keys removed from their environment so your subscription is
what gets used. They only ever see the prompt text.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from . import config

PROVIDERS = ["claude", "codex", "anthropic", "openai", "ollama", "none"]
LABELS = {
    "claude": "Claude (Claude Code, your subscription)",
    "codex": "ChatGPT (Codex CLI, your subscription)",
    "anthropic": "Claude API (ANTHROPIC_API_KEY)",
    "openai": "OpenAI API (OPENAI_API_KEY)",
    "ollama": "Ollama (local model)",
    "none": "Retrieval only (no LLM)",
}
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-5-mini", "ollama": "qwen2.5-coder:7b"}
SETUP_HINTS = {
    "claude": "Install Claude Code and sign in once: curl -fsSL https://claude.ai/install.sh | bash && claude auth login",
    "codex": "Install Codex and sign in with ChatGPT: npm install -g @openai/codex && codex login",
    "anthropic": "Set ANTHROPIC_API_KEY in .env",
    "openai": "Set OPENAI_API_KEY in .env",
    "ollama": "Install Ollama, run `ollama serve`, then `ollama pull qwen2.5-coder:7b`",
    "none": "",
}


class LLMError(Exception):
    pass


@dataclass
class LLMResult:
    text: str
    provider: str
    seconds: float


def _find_cli(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    home = Path.home()
    for p in (home / ".local/bin" / name, home / ".claude/local" / name, Path("/opt/homebrew/bin") / name,
              Path("/usr/local/bin") / name, home / ".npm-global/bin" / name):
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return None


def _ollama_up() -> bool:
    try:
        import httpx
        return httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=0.4).status_code == 200
    except Exception:
        return False


def status() -> list[dict]:
    """Which providers look usable right now."""
    checks = {
        "claude": _find_cli("claude") is not None,
        "codex": _find_cli("codex") is not None,
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "ollama": _ollama_up(),
        "none": True,
    }
    return [{"id": p, "label": LABELS[p], "available": checks[p], "setup": SETUP_HINTS[p]} for p in PROVIDERS]


def resolve(provider: str | None) -> str:
    provider = (provider or config.LLM_PROVIDER or "auto").lower()
    if provider != "auto":
        if provider not in PROVIDERS:
            raise LLMError(f"Unknown LLM provider {provider!r}.")
        return provider
    for s in status():
        if s["available"]:
            return s["id"]
    return "none"


def _clean_env(*drop: str) -> dict:
    return {k: v for k, v in os.environ.items() if k not in drop}


def _run(cmd: list[str], stdin: str, env: dict, cwd: str, timeout: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, input=stdin, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout)
    except FileNotFoundError as e:
        raise LLMError(f"{cmd[0]} isn't installed.") from e
    except subprocess.TimeoutExpired as e:
        raise LLMError(f"The model took longer than {timeout}s. Try again, or ask a narrower question.") from e


def _claude(system: str, prompt: str, model: str, timeout: int) -> str:
    exe = _find_cli("claude")
    if not exe:
        raise LLMError("Claude Code isn't installed. " + SETUP_HINTS["claude"])
    cmd = [exe, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
           "--strict-mcp-config", "--system-prompt", system]
    if model:
        cmd += ["--model", model]
    with tempfile.TemporaryDirectory(prefix="etr-claude-") as cwd:
        proc = _run(cmd, prompt, _clean_env("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"), cwd, timeout)
    data: dict = {}
    out = proc.stdout.strip()
    for candidate in (out, out.splitlines()[-1] if out else ""):
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    if proc.returncode != 0 or data.get("is_error") or not data.get("result"):
        detail = (data.get("result") or proc.stderr.strip() or proc.stdout.strip() or "no output")[:400].rstrip(".")
        hint = " If you haven't signed in on this computer yet, run `claude auth login` in Terminal." \
            if any(w in detail.lower() for w in ("login", "auth", "credential", "api key")) else ""
        raise LLMError(f"Claude Code couldn't answer: {detail}.{hint}")
    return data["result"]


def _codex(system: str, prompt: str, model: str, timeout: int) -> str:
    exe = _find_cli("codex")
    if not exe:
        raise LLMError("Codex isn't installed. " + SETUP_HINTS["codex"])
    with tempfile.TemporaryDirectory(prefix="etr-codex-") as cwd:
        out = Path(cwd) / "answer.txt"
        cmd = [exe, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--output-last-message", str(out)]
        if model:
            cmd += ["--model", model]
        cmd.append("-")  # read the prompt from stdin
        proc = _run(cmd, f"{system}\n\n{prompt}", _clean_env("OPENAI_API_KEY"), cwd, timeout)
        text = out.read_text(encoding="utf-8").strip() if out.exists() else ""
    if proc.returncode != 0 or not text:
        detail = ((proc.stderr or proc.stdout).strip()[-400:] or "no output").rstrip(".")
        raise LLMError(f"Codex couldn't answer: {detail}. If you haven't signed in on this computer yet, run `codex login` in Terminal.")
    return text


def _anthropic(system: str, prompt: str, model: str, timeout: int) -> str:
    import httpx
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("ANTHROPIC_API_KEY isn't set.")
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": model or DEFAULT_MODELS["anthropic"], "max_tokens": 2048, "system": system,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=timeout,
    )
    if r.status_code != 200:
        raise LLMError(f"Claude API error {r.status_code}: {r.text[:300]}")
    return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")


def _openai(system: str, prompt: str, model: str, timeout: int) -> str:
    import httpx
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise LLMError("OPENAI_API_KEY isn't set.")
    r = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model or DEFAULT_MODELS["openai"],
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]},
        timeout=timeout,
    )
    if r.status_code != 200:
        raise LLMError(f"OpenAI API error {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def _ollama(system: str, prompt: str, model: str, timeout: int) -> str:
    import httpx
    try:
        r = httpx.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={"model": model or DEFAULT_MODELS["ollama"], "stream": False,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                  "options": {"num_ctx": 16384}},
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        raise LLMError(f"Couldn't reach Ollama at {config.OLLAMA_HOST}. Is `ollama serve` running?") from e
    if r.status_code != 200:
        raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
    return r.json()["message"]["content"]


CALLS = {"claude": _claude, "codex": _codex, "anthropic": _anthropic, "openai": _openai, "ollama": _ollama}


def complete(system: str, prompt: str, provider: str | None = None, model: str | None = None) -> LLMResult:
    name = resolve(provider)
    if name == "none":
        raise LLMError("No LLM is connected.")
    if model is None:  # ETR_LLM_MODEL only applies to the provider named in ETR_LLM
        model = config.LLM_MODEL if name == config.LLM_PROVIDER else ""
    t0 = time.perf_counter()
    text = CALLS[name](system, prompt, model, config.LLM_TIMEOUT_S)
    return LLMResult(text.strip(), name, time.perf_counter() - t0)

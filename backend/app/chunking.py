"""Step 2: split each file into chunks that follow the code's structure.

Why this matters: retrieval returns chunks, not files. A chunk that is exactly one
function (with its comments and decorators) is easy to match to a question and easy
for the LLM to read. A chunk that starts halfway through one function and ends
halfway through the next is neither.

Strategy by file type:
- Python: the standard-library `ast` parser gives exact top-level functions and
  classes. Big classes are split into their methods, each labeled `Class.method`.
- Other code (JS/TS, Java, Go, Rust, C/C++, ...): an indentation-based splitter.
  Top-level declarations start at the lowest indentation in a block, so we cut there.
  If a block is still too big (a Java class, say), we step one level in and cut
  again, keeping the outer declaration line as context.
- Markdown/RST: split at headings.
- Config and text files: fixed-size line windows.

Then two clean-up passes: tiny neighbours (imports, constants, one-line helpers) are
packed together, and anything still too big is cut into overlapping line windows.
Every line of every file ends up in at least one chunk; a test checks this.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from .ingest import DATA_LANGUAGES, DOC_LANGUAGES, SourceFile

MAX_LINES = 80        # a chunk longer than this gets split
MAX_CHARS = 3000      # ...or bigger than this (bge-small reads ~512 tokens, about 2000 chars of code)
TARGET_CHARS = 1500   # small neighbours are packed up to this size
SMALL_CHARS = 700     # only blocks smaller than this are packed
WINDOW_LINES = 60     # window size when there is no structure left to cut on
OVERLAP = 8           # windows overlap so a statement cut at the edge appears whole in one of them
MAX_TEXT = 8000       # hard cap on stored chunk text (protects against single huge lines)


@dataclass
class Chunk:
    path: str
    language: str
    start_line: int   # 1-based, inclusive
    end_line: int     # 1-based, inclusive
    symbol: str = ""  # e.g. "def parse_args", "Session.request", "## Installation"
    kind: str = "code"
    context: str = "" # enclosing declaration, e.g. "class Session(SessionRedirectMixin):"
    text: str = ""

    @property
    def id(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def label(self) -> str:
        return " > ".join(x for x in (self.context, self.symbol) if x)

    def embed_text(self) -> str:
        """What gets embedded: the path and symbol help as much as the code itself."""
        head = self.path
        if self.label():
            head += "\n" + self.label()
        return head + "\n" + self.text

    def to_dict(self) -> dict:
        return {
            "path": self.path, "language": self.language, "start_line": self.start_line,
            "end_line": self.end_line, "symbol": self.symbol, "kind": self.kind,
            "context": self.context, "text": self.text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**{k: d[k] for k in ("path", "language", "start_line", "end_line", "symbol", "kind", "context", "text")})


@dataclass
class Seg:
    start: int        # 0-based line index, inclusive
    end: int          # 0-based line index, inclusive
    symbol: str = ""
    kind: str = "code"
    context: str = ""
    symbols: list[str] = field(default_factory=list)  # filled when segments are packed together
    skip_to: int = -1  # lines up to here are an enclosing declaration already used as context


def _indent(line: str) -> int:
    expanded = line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip(" "))


def _chars(lines: list[str], seg: Seg) -> int:
    return sum(len(lines[i]) + 1 for i in range(seg.start, seg.end + 1))


def _too_big(lines: list[str], seg: Seg) -> bool:
    return (seg.end - seg.start + 1) > MAX_LINES or _chars(lines, seg) > MAX_CHARS


# ---------------------------------------------------------------- symbols

SYMBOL_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)"), "function", "function {}"),
    (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(?:public\s+|private\s+|protected\s+|internal\s+)?(?:static\s+)?(?:final\s+|sealed\s+|data\s+|open\s+)?(?:class|struct|record)\s+([A-Za-z_$][\w$]*)"), "class", "class {}"),
    (re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?(?:interface|type|enum|protocol|trait|object)\s+([A-Za-z_$][\w$]*)"), "type", "{}"),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>"), "function", "{}()"),
    (re.compile(r"^\s*((?:[A-Za-z_$][\w$]*\.)+[A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"), "function", "{}"),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)"), "constant", "{}"),
    (re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)"), "function", "func {}"),
    (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)"), "function", "fn {}"),
    (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:enum|mod|impl(?:<[^>]*>)?)\s+([A-Za-z_][\w:<>]*)"), "type", "{}"),
    (re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)"), "function", "def {}"),
    (re.compile(r"^\s*(?:module|namespace)\s+([A-Za-z_][\w.:]*)"), "module", "{}"),
    (re.compile(r"^\s*fun\s+(?:<[^>]+>\s*)?(?:[\w.]+\.)?([A-Za-z_]\w*)\s*\("), "function", "fun {}"),
    # JS/TS class method: `async fetch(input) {`, `static create(a: A): B {`, `#retry() {`
    (re.compile(r"^\s*(?:(?:public|private|protected|static|async|readonly|override|abstract|get|set)\s+)*\*?(#?[A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?::\s*[^={;]+)?\s*\{\s*$"), "method", "{}()"),
]
# Languages without a `function`/`def` keyword, where a method is recognized by "Type name(".
TYPED_LANGUAGES = {"java", "c", "cpp", "csharp", "kotlin", "scala", "swift", "objectivec", "dart", "php", "groovy"}
TYPED_PATTERN = (re.compile(r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|synchronized|virtual|override|async|inline|extern|unsafe|const|constexpr|explicit|friend|open|suspend|default|native|transient)\s+)*(?:<[^>]+>\s+)?(?:[\w<>\[\],.:*&?]+\s+)+\*?&?([A-Za-z_]\w*)\s*\([^;]*$"), "method", "{}()")
NOT_A_NAME = {"if", "for", "while", "switch", "return", "catch", "with", "elif", "else", "new", "sizeof", "await",
              "throw", "case", "do", "try", "yield", "typeof", "delete", "assert", "print", "super", "this"}
COMMENT_LINE = re.compile(r"^\s*(//|/\*|\*|#(?!\[)|--|;|<!--)")


def guess_symbol(lines: list[str], start: int, end: int, language: str = "") -> tuple[str, str]:
    """Name the declaration a segment starts with, skipping leading comments, docstrings and decorators."""
    patterns = SYMBOL_PATTERNS + ([TYPED_PATTERN] if language in TYPED_LANGUAGES else [])
    looked = 0
    for i in range(start, min(end, start + 60) + 1):
        line = lines[i]
        if not line.strip() or COMMENT_LINE.match(line) or line.lstrip().startswith("@"):
            continue
        looked += 1
        if looked > 2:
            break
        first_word = re.split(r"[\s(]", line.strip(), maxsplit=1)[0]
        if first_word in NOT_A_NAME:
            return "", "code"
        for pattern, kind, fmt in patterns:
            m = pattern.match(line)
            if m and m.group(1) not in NOT_A_NAME:
                return fmt.format(m.group(1)), kind
    return "", "code"


# ---------------------------------------------------------------- python (AST)

def _describe(node: ast.stmt) -> tuple[str, str]:
    if isinstance(node, ast.FunctionDef):
        return f"def {node.name}", "function"
    if isinstance(node, ast.AsyncFunctionDef):
        return f"async def {node.name}", "function"
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}", "class"
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "imports", "imports"
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        return (", ".join(names[:3]), "constant") if names else ("", "module")
    if isinstance(node, ast.If):
        test = node.test
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"):
            return "if __name__ == '__main__'", "entrypoint"
    return "", "module"


def _node_starts(nodes: list[ast.stmt], lines: list[str], floor: int) -> list[tuple[int, ast.stmt]]:
    """Start line of each node, pulled up to include decorators and the comment block above it."""
    out = []
    prev_end = floor - 1
    for node in nodes:
        s = node.lineno - 1
        for d in getattr(node, "decorator_list", []) or []:
            s = min(s, d.lineno - 1)
        while s - 1 > prev_end and lines[s - 1].strip().startswith("#"):
            s -= 1
        out.append((s, node))
        prev_end = (node.end_lineno or node.lineno) - 1
    return out


def python_segments(text: str, lines: list[str]) -> list[Seg] | None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    if not tree.body:
        return None
    starts = _node_starts(tree.body, lines, 0)
    segs: list[Seg] = []
    for i, (s, node) in enumerate(starts):
        start = 0 if i == 0 else s
        end = starts[i + 1][0] - 1 if i + 1 < len(starts) else len(lines) - 1
        symbol, kind = _describe(node)
        seg = Seg(start, end, symbol, kind)
        if isinstance(node, ast.ClassDef) and _too_big(lines, seg):
            segs.extend(_split_class(node, seg, lines))
        else:
            segs.append(seg)
    return segs


def _split_class(node: ast.ClassDef, seg: Seg, lines: list[str]) -> list[Seg]:
    header = lines[node.lineno - 1].strip()
    members = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if not members:
        return [seg]
    starts = _node_starts(members, lines, node.lineno)
    out = []
    if starts[0][0] > seg.start:
        out.append(Seg(seg.start, starts[0][0] - 1, f"class {node.name}", "class"))
    for i, (s, member) in enumerate(starts):
        end = starts[i + 1][0] - 1 if i + 1 < len(starts) else seg.end
        sym, kind = _describe(member)
        name = sym.split(" ")[-1]
        out.append(Seg(s, end, f"{node.name}.{name}", "method" if kind == "function" else kind, header))
    return out


# ---------------------------------------------------------------- everything else (indentation)

CLOSER = re.compile(r"^\s*(?:[}\])]|end\b|</|\*/|fi\b|done\b|esac\b|#endif|endif\b|endfunction\b|endmodule\b)")
CONTINUATION = re.compile(r"^\s*(?:\{|else\b|elif\b|except\b|finally\b|catch\b|\.[A-Za-z_]|&&|\|\||\+ |\? |: |=>|,)")
ATTACH_TO_NEXT = re.compile(r"^\s*(?://|/\*|\*|#(?!include|define|if|endif|else|pragma|import|undef)|--|@|#\[|\[[A-Z]\w*)")


def generic_segments(lines: list[str], lo: int, hi: int, context: str = "", language: str = "") -> list[Seg]:
    """Cut [lo, hi] wherever a new statement starts at the block's lowest indentation."""
    nonblank = [i for i in range(lo, hi + 1) if lines[i].strip()]
    if not nonblank:
        return []
    # The block's own closing brace and stray comments sit further left than its
    # statements; leave them out when finding the indentation statements start at.
    structural = [i for i in nonblank if not CLOSER.match(lines[i]) and not COMMENT_LINE.match(lines[i])]
    base = min(_indent(lines[i]) for i in (structural or nonblank))
    starts: list[int] = []
    attaching = False   # previous base-level line was a comment/decorator that belongs to the next line
    for i in nonblank:
        line = lines[i]
        if _indent(line) != base:
            continue
        if CLOSER.match(line) or CONTINUATION.match(line):
            attaching = False
            continue
        if not attaching:
            starts.append(i)
        attaching = bool(ATTACH_TO_NEXT.match(line))
    if not starts:
        starts = [lo]
    starts[0] = lo
    segs = []
    for k, s in enumerate(starts):
        e = starts[k + 1] - 1 if k + 1 < len(starts) else hi
        sym, kind = guess_symbol(lines, s, e, language)
        segs.append(Seg(s, e, sym, kind, context))
    return segs


def markdown_segments(lines: list[str]) -> list[Seg]:
    starts, titles, in_fence = [0], {0: ""}, False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if re.match(r"^#{1,6}\s+\S", line):
            title = line.strip()
        elif (i + 1 < len(lines) and line.strip() and re.fullmatch(r"(=+|-+|~+|\^+)", lines[i + 1].strip() or "x")
              and len(lines[i + 1].strip()) >= 3):
            title = line.strip()  # RST / setext heading
        else:
            continue
        if i == 0:
            titles[0] = title
        else:
            starts.append(i)
            titles[i] = title
    segs = []
    for k, s in enumerate(starts):
        e = starts[k + 1] - 1 if k + 1 < len(starts) else len(lines) - 1
        segs.append(Seg(s, e, titles[s], "section"))
    return segs


# ---------------------------------------------------------------- clean-up passes

def window_split(seg: Seg, lines: list[str]) -> list[Seg]:
    out, i, part = [], seg.start, 1
    while i <= seg.end:
        j, chars = i, 0
        while j <= seg.end:
            size = len(lines[j]) + 1
            if j > i and ((j - i) >= WINDOW_LINES or chars + size > MAX_CHARS):
                break
            chars += size
            j += 1
        label = f"{seg.symbol} (part {part})" if seg.symbol else ""
        out.append(Seg(i, j - 1, label, seg.kind, seg.context))
        if j > seg.end:
            break
        i = max(j - OVERLAP, i + 1)
        part += 1
    if len(out) == 1:
        out[0].symbol = seg.symbol
    return out


def _decl_line(lines: list[str], seg: Seg) -> int | None:
    for i in range(max(seg.start, seg.skip_to + 1), seg.end + 1):
        line = lines[i]
        if line.strip() and not ATTACH_TO_NEXT.match(line):
            return i
    return None


def _header_end(lines: list[str], decl: int, end: int) -> int:
    """Last line of a declaration whose parameters span several lines, e.g. `def f(\n a,\n b,\n):`."""
    depth = 0
    for i in range(decl, min(end, decl + 30) + 1):
        code = re.sub(r"(\"[^\"]*\"|'[^']*')", "", lines[i])
        depth += code.count("(") + code.count("[") - code.count(")") - code.count("]")
        if depth <= 0:
            return i
    return decl


def split_large(seg: Seg, lines: list[str], language: str, depth: int = 0) -> list[Seg]:
    if not _too_big(lines, seg):
        return [seg]
    if language in DOC_LANGUAGES or language in DATA_LANGUAGES or depth >= 4:
        return window_split(seg, lines)
    decl = _decl_line(lines, seg)
    if decl is None or decl >= seg.end:
        return window_split(seg, lines)
    head_end = _header_end(lines, decl, seg.end)
    header = " ".join(" ".join(lines[decl:head_end + 1]).split())
    header = header if len(header) <= 100 else header[:97] + "..."
    context = " > ".join(p for p in (seg.context, header) if p)[-200:]
    body_start = head_end + 1
    if body_start > seg.end:
        return window_split(seg, lines)
    inner = generic_segments(lines, body_start, seg.end, context, language)
    if len(inner) < 2:
        return window_split(seg, lines)
    inner[0].start = seg.start          # keep the declaration line with the first piece...
    inner[0].skip_to = head_end         # ...but don't treat it as that piece's own declaration
    if not inner[0].symbol:
        inner[0].symbol, inner[0].kind = seg.symbol, seg.kind
    out: list[Seg] = []
    for s in pack(inner, lines):
        out.extend(split_large(s, lines, language, depth + 1))
    return out


def pack(segs: list[Seg], lines: list[str]) -> list[Seg]:
    """Merge runs of small neighbours (imports, constants, tiny helpers) into one chunk."""
    out: list[Seg] = []
    cur: Seg | None = None
    cur_chars = 0
    for s in segs:
        size = _chars(lines, s)
        if (cur is not None and (size <= SMALL_CHARS or cur_chars < 200) and cur_chars <= SMALL_CHARS * 2
                and cur_chars + size <= TARGET_CHARS and s.context == cur.context
                and (s.end - cur.start + 1) <= MAX_LINES):
            cur.end = s.end
            cur.symbols.append(s.symbol)
            if s.kind != cur.kind:
                cur.kind = "code"
            cur_chars += size
            continue
        if cur is not None:
            out.append(cur)
        cur = Seg(s.start, s.end, s.symbol, s.kind, s.context, [s.symbol], s.skip_to)
        cur_chars = size
    if cur is not None:
        out.append(cur)
    for s in out:
        names = [n for n in dict.fromkeys(s.symbols) if n]
        if len(names) > 1:
            s.symbol = ", ".join(names[:4]) + (f" +{len(names) - 4} more" if len(names) > 4 else "")
        elif names:
            s.symbol = names[0]
    return out


def chunk_file(sf: SourceFile) -> list[Chunk]:
    lines = sf.text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or not any(l.strip() for l in lines):
        return []
    n = len(lines)
    lang = sf.language

    segs: list[Seg] | None = None
    if lang == "python" and not sf.path.endswith(".ipynb"):
        segs = python_segments(sf.text, lines)
    if segs is None:
        if lang in ("markdown", "rst"):
            segs = markdown_segments(lines)
        elif lang in DATA_LANGUAGES or lang == "text":
            segs = [Seg(0, n - 1)]
        else:
            segs = generic_segments(lines, 0, n - 1, "", lang)

    final: list[Seg] = []
    for s in pack(segs, lines):
        final.extend(split_large(s, lines, lang))
    final = pack(final, lines)  # pieces produced by different splits can still be tiny neighbours

    chunks = []
    for s in final:
        a, b = s.start, s.end
        while a <= b and not lines[a].strip():
            a += 1
        while b >= a and not lines[b].strip():
            b -= 1
        if a > b:
            continue
        text = "\n".join(lines[a:b + 1])
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT] + "\n…"
        chunks.append(Chunk(sf.path, lang, a + 1, b + 1, s.symbol, s.kind, s.context, text))
    return chunks

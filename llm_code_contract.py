from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

CODE_RESPONSE_VERSION = "code_response.v2"
DEFAULT_CODE_FILENAME = "solve_module.py"
DEFAULT_CODE_LANGUAGE = "python"
DEFAULT_CODE_ARTIFACT_TYPE = "python_module"

RE_FENCED_BLOCK = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\s*(?P<code>.*?)```", re.DOTALL)
RE_RAW_CODE_START = re.compile(
    r"^(?:#!\s*/|from\s+\S+\s+import|import\s+\S+|async\s+def\s+\w+\s*\(|def\s+\w+\s*\(|class\s+\w+\s*(?:\(|:)|if __name__ == [\"']__main__[\"']\s*:|@[A-Za-z_][A-Za-z0-9_\.\(\), ]*)",
    re.IGNORECASE,
)
RE_CODE_LIKE_LINE = re.compile(
    r"^(?:@|async\s+def\s+\w+\s*\(|def\s+\w+\s*\(|class\s+\w+\s*(?:\(|:)|from\s+\S+\s+import|import\s+\S+|if\b|elif\b|else:|for\b|while\b|try:|except\b|finally:|with\b|return\b|raise\b|assert\b|pass\b|break\b|continue\b|[A-Za-z_][A-Za-z0-9_\[\], ]*\s*=)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CodeEnvelope:
    version: str
    artifact_type: str
    language: str
    filename: str
    code: str
    source_kind: str


def code_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["version", "artifact_type", "language", "filename", "code"],
        "properties": {
            "version": {"type": "string", "enum": [CODE_RESPONSE_VERSION]},
            "artifact_type": {"type": "string", "enum": [DEFAULT_CODE_ARTIFACT_TYPE]},
            "language": {"type": "string", "enum": [DEFAULT_CODE_LANGUAGE]},
            "filename": {"type": "string", "enum": [DEFAULT_CODE_FILENAME]},
            "code": {"type": "string"},
        },
        "additionalProperties": False,
    }


def code_response_example(*, filename: str = DEFAULT_CODE_FILENAME) -> str:
    example = {
        "version": CODE_RESPONSE_VERSION,
        "artifact_type": DEFAULT_CODE_ARTIFACT_TYPE,
        "language": DEFAULT_CODE_LANGUAGE,
        "filename": filename,
        "code": "from __future__ import annotations\\n\\nimport json\\n\\n\\ndef solve(vec):\\n    return [], list(vec)\\n",
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def strict_code_response_requirements(*, prefer_minimal_patch: bool, filename: str = DEFAULT_CODE_FILENAME) -> str:
    lines = [
        "STRICT OUTPUT REQUIREMENTS:",
        "- Return exactly one JSON object and no prose outside JSON.",
        f"- The JSON object must contain exactly these keys: version, artifact_type, language, filename, code.",
        f"- Set version={CODE_RESPONSE_VERSION!r}, artifact_type={DEFAULT_CODE_ARTIFACT_TYPE!r}, language={DEFAULT_CODE_LANGUAGE!r}, filename={filename!r}.",
        "- Put the entire Python file only inside the code string.",
        "- Do not wrap the code in markdown fences inside the JSON string.",
        "- Do not include explanations, bullet points, extra keys, or commentary before or after the JSON object.",
        "- Inside the code, omit comments and docstrings unless they are absolutely necessary.",
        "- Preserve the public solve(vec) entrypoint and the script-mode JSON stdout contract with keys moves and sorted_array.",
    ]
    if prefer_minimal_patch:
        lines.append("- Prefer a minimal patch over a rewrite whenever possible.")
    lines.extend(["JSON EXAMPLE:", code_response_example(filename=filename)])
    return "\n".join(lines)


def repair_code_response_prompt(prompt: str, *, filename: str = DEFAULT_CODE_FILENAME) -> str:
    return (
        f"{prompt}\n\n"
        "CRITICAL OUTPUT FORMAT REPAIR:\n"
        "Your previous reply did not follow the required machine-readable code envelope.\n"
        + strict_code_response_requirements(prefer_minimal_patch=False, filename=filename)
    )


def _iter_balanced_json_objects(text: str) -> Iterator[str]:
    source = str(text or "")
    n = len(source)
    i = 0
    while i < n:
        if source[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        quote = ""
        start = i
        for j in range(i, n):
            ch = source[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
                continue
            if ch in {'"', "'"}:
                in_string = True
                quote = ch
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield source[start:j + 1]
                    i = j + 1
                    break
        else:
            break


def _try_load_json_dict(text: str) -> Optional[Dict[str, Any]]:
    candidate = str(text or "").strip()
    if not candidate:
        return None
    try:
        loaded = json.loads(candidate)
    except Exception:
        try:
            loaded = ast.literal_eval(candidate)
        except Exception:
            return None
    return loaded if isinstance(loaded, dict) else None


def _candidate_json_texts(text: str) -> List[Tuple[str, str]]:
    source = str(text or "").strip()
    if not source:
        return []
    out: List[Tuple[str, str]] = [("whole", source)]
    seen = {source}
    for match in RE_FENCED_BLOCK.finditer(source):
        lang = (match.group("lang") or "").strip().lower()
        code = (match.group("code") or "").strip()
        if not code or code in seen:
            continue
        if lang in {"json", "javascript", "js", ""}:
            out.append((f"fenced:{lang or 'plain'}", code))
            seen.add(code)
    for candidate in _iter_balanced_json_objects(source):
        payload = candidate.strip()
        if payload and payload not in seen:
            out.append(("balanced", payload))
            seen.add(payload)
    return out


def _normalize_code_envelope(payload: Dict[str, Any], *, source_kind: str) -> Optional[CodeEnvelope]:
    dict_candidates: List[Dict[str, Any]] = [payload]
    for key in ("artifact", "answer", "solution", "output", "result"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            dict_candidates.append(nested)
    files = payload.get("files")
    if isinstance(files, list):
        for item in files:
            if isinstance(item, dict):
                dict_candidates.append(item)

    for item in dict_candidates:
        code = item.get("code")
        if not isinstance(code, str) or not code.strip():
            continue
        version = str(item.get("version") or payload.get("version") or CODE_RESPONSE_VERSION).strip() or CODE_RESPONSE_VERSION
        artifact_type = str(item.get("artifact_type") or item.get("type") or payload.get("artifact_type") or DEFAULT_CODE_ARTIFACT_TYPE).strip() or DEFAULT_CODE_ARTIFACT_TYPE
        language = str(item.get("language") or item.get("lang") or payload.get("language") or DEFAULT_CODE_LANGUAGE).strip().lower() or DEFAULT_CODE_LANGUAGE
        filename = str(item.get("filename") or item.get("path") or payload.get("filename") or DEFAULT_CODE_FILENAME).strip() or DEFAULT_CODE_FILENAME
        if language not in {"python", "py", "python3"}:
            continue
        if artifact_type not in {DEFAULT_CODE_ARTIFACT_TYPE, "code", "file", "python", "python_file"}:
            continue
        cleaned = _trim_candidate_edges(code)
        if not cleaned:
            continue
        return CodeEnvelope(
            version=version,
            artifact_type=artifact_type,
            language="python",
            filename=filename,
            code=cleaned,
            source_kind=source_kind,
        )
    return None


def extract_code_envelope(text: str) -> Optional[CodeEnvelope]:
    for source_kind, candidate in _candidate_json_texts(text):
        payload = _try_load_json_dict(candidate)
        if not payload:
            continue
        envelope = _normalize_code_envelope(payload, source_kind=source_kind)
        if envelope is not None:
            return envelope
    return None


def _span_contains(span: Tuple[Tuple[int, int], Tuple[int, int]], start: Tuple[int, int], end: Tuple[int, int]) -> bool:
    return span[0] <= start and end <= span[1]


def _collect_docstring_spans(source: str) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    spans: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    try:
        tree = ast.parse(source)
    except Exception:
        return spans

    def _push(node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if not body:
            return
        first = body[0]
        if not isinstance(first, ast.Expr):
            return
        value = getattr(first, "value", None)
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            end_lineno = getattr(first, "end_lineno", first.lineno)
            end_col = getattr(first, "end_col_offset", 0)
            spans.append(((first.lineno, first.col_offset), (end_lineno, end_col)))

    _push(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _push(node)
    return spans


def _heuristic_strip_comments_and_docstrings(source: str) -> str:
    kept: List[str] = []
    in_triple: Optional[str] = None
    for line in source.splitlines():
        stripped = line.lstrip()
        if in_triple is not None:
            if in_triple in stripped:
                in_triple = None
            continue
        if stripped.startswith("#"):
            continue
        match = re.match(r"^(?:[rRuUbBfF]{0,2})?(?P<quote>'''|\"\"\")", stripped)
        if match:
            quote = match.group("quote")
            tail = stripped[match.end():]
            if quote in tail:
                continue
            in_triple = quote
            continue
        kept.append(line.rstrip())
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def strip_python_comments_and_docstrings(code: str) -> str:
    source = str(code or "").strip()
    if not source:
        return ""
    original_lines = source.splitlines()
    preserved_prefix: List[str] = []
    if original_lines and original_lines[0].startswith("#!"):
        preserved_prefix.append(original_lines[0].rstrip())
    if len(original_lines) > 1 and re.match(r"^#.*coding[:=]", original_lines[1]):
        preserved_prefix.append(original_lines[1].rstrip())

    try:
        ast.parse(source)
        parsed_ok = True
    except Exception:
        parsed_ok = False

    spans = _collect_docstring_spans(source) if parsed_ok else []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except Exception:
        cleaned = _heuristic_strip_comments_and_docstrings(source) or source
        if preserved_prefix:
            body_lines = cleaned.splitlines()
            while body_lines and body_lines[0].rstrip() in preserved_prefix:
                body_lines.pop(0)
            cleaned = "\n".join(preserved_prefix + body_lines)
        return cleaned.strip()

    kept = []
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            token_text = tok.string or ""
            if tok.start == (1, 0) and token_text.startswith("#!"):
                kept.append(tok)
            elif tok.start[0] == 2 and preserved_prefix and len(preserved_prefix) > 1 and re.match(r"^#.*coding[:=]", token_text):
                kept.append(tok)
            continue
        if tok.type == tokenize.STRING and any(_span_contains(span, tok.start, tok.end) for span in spans):
            continue
        kept.append(tok)

    try:
        cleaned = tokenize.untokenize(kept)
    except Exception:
        cleaned = _heuristic_strip_comments_and_docstrings(source) or source
        if preserved_prefix:
            body_lines = cleaned.splitlines()
            while body_lines and body_lines[0].rstrip() in preserved_prefix:
                body_lines.pop(0)
            cleaned = "\n".join(preserved_prefix + body_lines)
        return cleaned.strip()

    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    if not parsed_ok and ('\"\"\"' in cleaned or "'''" in cleaned):
        cleaned = _heuristic_strip_comments_and_docstrings(cleaned) or cleaned
    if preserved_prefix:
        body_lines = cleaned.splitlines()
        while body_lines and body_lines[0].rstrip() in preserved_prefix:
            body_lines.pop(0)
        cleaned = "\n".join(preserved_prefix + body_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _looks_like_python(code: str) -> bool:
    low = str(code or "").lower()
    indicators = (
        "def ",
        "async def ",
        "class ",
        "import ",
        "from ",
        "__main__",
        "return ",
        "for ",
        "while ",
        "if ",
        "try:",
        "with ",
    )
    return any(token in low for token in indicators)


def _looks_like_python_line(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return True
    if stripped.startswith(("#!", "#", '"""', "'''")):
        return True
    if RE_CODE_LIKE_LINE.match(stripped):
        return True
    if str(line).startswith((" ", "\t")):
        return True
    if re.match(r"^[\]\)\}\],.:]+$", stripped):
        return True
    if stripped.endswith((":", "(", "[", "{", "\\")):
        return True
    return False


def _looks_like_narrative_line(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return False
    low = stripped.lower()
    if stripped.startswith("```"):
        return True
    if low.startswith((
        "content of ",
        "code starts here",
        "save as ",
        "note:",
        "explanation",
        "here is",
        "here's",
        "the following code",
        "this is ",
    )):
        return True
    if stripped.startswith(("- ", "* ", "> ", "### ")):
        return True
    if re.match(r"^\d+[.)]\s", stripped):
        return True
    return False


def _trim_candidate_edges(code: str) -> str:
    lines = str(code or "").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and _looks_like_narrative_line(lines[-1]) and not _looks_like_python_line(lines[-1]):
        lines.pop()
    while lines and _looks_like_narrative_line(lines[0]) and not RE_RAW_CODE_START.match(lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def _extract_raw_python_candidates(text: str) -> List[str]:
    source = str(text or "").strip()
    if not source:
        return []
    if not any(token in source for token in ("def solve", "def add", "import ", "from __future__", "if __name__", "class ")):
        return []
    lines = source.splitlines()
    start_indexes: List[int] = []
    for idx, line in enumerate(lines):
        if RE_RAW_CODE_START.match(line.strip()):
            start_indexes.append(idx)
    if not start_indexes:
        return []

    candidates: List[str] = []
    seen = set()
    for start_idx in start_indexes:
        kept: List[str] = []
        for line in lines[start_idx:]:
            stripped = line.strip()
            if stripped.startswith("```"):
                break
            if not stripped:
                kept.append(line)
                continue
            if _looks_like_python_line(line):
                kept.append(line.rstrip())
                continue
            if kept and _looks_like_narrative_line(line):
                break
            if kept:
                break
        candidate = _trim_candidate_edges("\n".join(kept))
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def _python_candidate_score(code: str, *, lang: str = "", fenced: bool = False) -> int:
    score = 0
    low = str(code or "").lower()
    norm_lang = str(lang or "").strip().lower()
    if norm_lang in {"python", "py", "python3"}:
        score += 120
    elif norm_lang:
        score -= 20
    if fenced:
        score += 5
    if "def solve" in low:
        score += 140
    if "def add" in low:
        score += 80
    if _looks_like_python(code):
        score += 20
    try:
        ast.parse(code)
        score += 60
    except Exception:
        score -= 10
    score += min(sum(1 for line in str(code).splitlines() if line.strip()), 80)
    return score


def extract_python_candidate(text: str, *, strip_comments_docstrings: bool = False) -> str:
    source = str(text or "").strip()
    if not source:
        return ""

    envelope = extract_code_envelope(source)
    if envelope is not None:
        code = envelope.code
        if strip_comments_docstrings:
            code = strip_python_comments_and_docstrings(code) or code
        return _trim_candidate_edges(code)

    candidates: List[Tuple[int, int, str]] = []
    for idx, match in enumerate(RE_FENCED_BLOCK.finditer(source)):
        lang = (match.group("lang") or "").strip()
        code = _trim_candidate_edges((match.group("code") or "").strip())
        if not code:
            continue
        if strip_comments_docstrings:
            code = _trim_candidate_edges(strip_python_comments_and_docstrings(code) or code)
        score = _python_candidate_score(code, lang=lang, fenced=True)
        if lang and lang.lower() not in {"python", "py", "python3"} and not _looks_like_python(code):
            score -= 50
        candidates.append((score, -idx, code))

    for idx, code in enumerate(_extract_raw_python_candidates(source), start=1):
        cleaned = _trim_candidate_edges(strip_python_comments_and_docstrings(code) if strip_comments_docstrings else code)
        score = _python_candidate_score(cleaned, fenced=False)
        candidates.append((score, -10_000 - idx, cleaned))

    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2].strip()


def python_compiles(code: str) -> bool:
    if not code:
        return False
    try:
        ast.parse(code)
    except Exception:
        return False
    return True

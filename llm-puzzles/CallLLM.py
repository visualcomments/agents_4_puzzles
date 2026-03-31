import ast
import importlib
import io
import json
import os
from pathlib import Path
import queue
import re
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

import requests

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_code_contract as code_contract

try:
    import g4f
    from g4f.Provider import Any as Provider
    try:
        from g4f.errors import ModelNotFoundError  # type: ignore
    except Exception:
        class ModelNotFoundError(Exception):
            pass
except Exception:
    g4f, Provider = None, None
    class ModelNotFoundError(Exception):
        pass

try:
    import g4f.providers.retry_provider as retry_mod
    OriginalRotatedProvider = retry_mod.RotatedProvider
    class TrackedRotated(OriginalRotatedProvider):
        pass
    retry_mod.RotatedProvider = TrackedRotated
except Exception:
    pass

CONFIG: Dict = {
    "CONSTANTS": {"REQUEST_TIMEOUT": 60, "MODEL_TYPE_TEXT": "text", "DELIMITER_MODEL": "|", "MAX_WORKERS": 8},
    "URLS": {"WORKING_RESULTS": "https://raw.githubusercontent.com/xtekky/gpt4free/refs/heads/main/docs/WORKING_RESULTS.txt"}
}

DEFAULT_MODELS = [
    "gpt-4o-mini",
    "claude-3.5-sonnet",
    "deepseek-chat",
    "command-r-plus",
    "command-r",
    "aria",
]

MODEL_HINT_SCORES: Tuple[Tuple[str, int], ...] = (
    ("claude-3.7", 170),
    ("claude-3-7", 170),
    ("claude-sonnet-4", 168),
    ("claude-3.5-sonnet", 165),
    ("claude-3-5-sonnet", 165),
    ("gpt-4.1", 160),
    ("gpt-4o", 155),
    ("o3", 152),
    ("o1", 150),
    ("deepseek-r1", 148),
    ("deepseek-chat", 142),
    ("qwen2.5-coder", 140),
    ("qwen-2.5-coder", 140),
    ("qwq", 138),
    ("coder", 132),
    ("command-r-plus", 128),
    ("command-r+", 128),
    ("command-r", 120),
    ("qwen", 116),
    ("gemini", 110),
    ("llama", 100),
    ("aria", 70),
)


def _chunk_to_text(chunk) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        for key in ("content", "text"):
            val = chunk.get(key)
            if isinstance(val, str):
                return val
        try:
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    return content
            return ""
        except Exception:
            return ""
    for attr in ("content", "text"):
        try:
            val = getattr(chunk, attr)
            if isinstance(val, str):
                return val
        except Exception:
            pass
    try:
        choices = getattr(chunk, "choices")
        if choices:
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if isinstance(content, str):
                return content
    except Exception:
        pass
    return ""


def _iter_to_text(resp, *, max_chars: int | None = None) -> str:
    if isinstance(resp, str):
        if max_chars is not None and max_chars > 0:
            return resp[:max_chars]
        return resp
    if resp is None or not hasattr(resp, "__iter__"):
        return _chunk_to_text(resp)
    buf = io.StringIO()
    size = 0
    try:
        for ch in resp:
            txt = _chunk_to_text(ch)
            if not txt:
                continue
            if max_chars is not None and max_chars > 0:
                remaining = max_chars - size
                if remaining <= 0:
                    break
                if len(txt) > remaining:
                    buf.write(txt[:remaining])
                    size += remaining
                    break
            buf.write(txt)
            size += len(txt)
        return buf.getvalue()
    except Exception:
        return buf.getvalue()



RE_FENCED_BLOCK = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\s*(?P<code>.*?)```", re.DOTALL)
RE_RAW_CODE_START = re.compile(
    r"^(?:#!\s*/|from\s+\S+\s+import|import\s+\S+|async\s+def\s+\w+\s*\(|def\s+\w+\s*\(|class\s+\w+\s*(?:\(|:)|if __name__ == [\"']__main__[\"']\s*:|@[A-Za-z_][A-Za-z0-9_\.\(\), ]*)",
    re.IGNORECASE,
)
RE_CODE_LIKE_LINE = re.compile(
    r"^(?:@|async\s+def\s+\w+\s*\(|def\s+\w+\s*\(|class\s+\w+\s*(?:\(|:)|from\s+\S+\s+import|import\s+\S+|if\b|elif\b|else:|for\b|while\b|try:|except\b|finally:|with\b|return\b|raise\b|assert\b|pass\b|break\b|continue\b|[A-Za-z_][A-Za-z0-9_\[\], ]*\s*=)",
    re.IGNORECASE,
)


def _looks_like_python(code: str) -> bool:
    low = (code or "").lower()
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
    stripped = (line or "").strip()
    if not stripped:
        return True
    if stripped.startswith(("#!", "#", '"""', "'''")):
        return True
    if RE_CODE_LIKE_LINE.match(stripped):
        return True
    if line.startswith((" ", "\t")):
        return True
    if re.match(r"^[\]\)\}\],.:]+$", stripped):
        return True
    if stripped.endswith((":", "(", "[", "{", "\\")):
        return True
    return False


def _looks_like_narrative_line(line: str) -> bool:
    stripped = (line or "").strip()
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
    lines = (code or "").splitlines()
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
    source = (text or "").strip()
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
    low = (code or "").lower()
    norm_lang = (lang or "").strip().lower()
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
    score += min(sum(1 for line in code.splitlines() if line.strip()), 80)
    return score


def _extract_python_candidate(text: str) -> str:
    return code_contract.extract_python_candidate(text)


def _python_compiles(code: str) -> bool:
    return code_contract.python_compiles(code)


def _model_score(model: str) -> int:
    low = model.lower()
    score = 0
    for needle, value in MODEL_HINT_SCORES:
        if needle in low:
            score = max(score, value)
    if "mini" in low:
        score -= 6
    return score


def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        s = (item or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def rank_models_for_code(models: Iterable[str]) -> List[str]:
    unique = _dedupe(models)
    return sorted(unique, key=lambda m: (-_model_score(m), m.lower()))


def quick_selfcheck(
    models: List[str],
    prompt: str = "ok?",
    max_models: int = 2,
    timeout: int = 8,
    mode: Optional[str] = None,
) -> List[str]:
    ok: List[str] = []
    if g4f is None or Provider is None:
        return ok
    test_subset = [m for m in models if isinstance(m, str) and m.strip()][:max_models]
    mode = (mode or os.getenv("G4F_SELFCHECK_MODE", "code")).strip().lower()

    if mode == "code":
        prompt = (
            "Return exactly one JSON object containing a minimal Python solver module. "
            "Set the code field to a complete solve_module.py that defines solve(vec) and returns the input unchanged. "
            "Do not add any explanation outside JSON.\n\n"
            + code_contract.strict_code_response_requirements(prefer_minimal_patch=False, filename='solve_module.py')
        )

    provider_name = os.getenv("G4F_PROVIDER", "").strip()
    provider = Provider
    if provider_name and g4f is not None:
        try:
            provider = getattr(g4f.Provider, provider_name)
        except Exception:
            provider = Provider

    for model in test_subset:
        try:
            resp = g4f.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                provider=provider,
                timeout=timeout,
                stream=True,
            )
            text = _iter_to_text(resp, max_chars=int(os.getenv("LLM_PUZZLES_MAX_RESPONSE_CHARS", "60000") or "60000"))
            if not isinstance(text, str) or not text.strip():
                continue
            if mode == "code":
                code = _extract_python_candidate(text)
                if code and _python_compiles(code):
                    ok.append(model)
            else:
                ok.append(model)
        except Exception:
            continue
    return ok


def _fetch_working_models(config: Dict) -> List[str]:
    url_txt = config.get("URLS", {}).get("WORKING_RESULTS")
    delimiter = config.get("CONSTANTS", {}).get("DELIMITER_MODEL", "|")
    text_type = config.get("CONSTANTS", {}).get("MODEL_TYPE_TEXT", "text")
    timeout = config.get("CONSTANTS", {}).get("REQUEST_TIMEOUT", 60)
    working_models: List[str] = []
    if not url_txt:
        return working_models
    try:
        resp = requests.get(url_txt, timeout=timeout)
        resp.raise_for_status()
        for line in resp.text.splitlines():
            if delimiter not in line:
                continue
            parts = [p.strip() for p in line.split(delimiter)]
            if len(parts) != 3 or parts[2] != text_type:
                continue
            name = parts[1]
            low = name.lower()
            if "flux" in low or any(sub in low for sub in ["image", "vision", "audio", "video"]):
                continue
            working_models.append(name)
    except Exception:
        pass
    return working_models


def _discover_g4f_models() -> List[str]:
    if g4f is None:
        return []
    models: List[str] = []
    try:
        gm = importlib.import_module("g4f.models")
        names = list(getattr(gm, "__all__", []))
        if not names:
            for key in dir(gm):
                if key.startswith("_"):
                    continue
                try:
                    val = getattr(gm, key)
                except Exception:
                    continue
                if isinstance(val, str):
                    names.append(val)
        for item in names:
            if not isinstance(item, str):
                continue
            s = item.strip()
            low = s.lower()
            if s and "flux" not in low and not any(sub in low for sub in ["image", "vision", "audio", "video"]):
                models.append(s)
    except Exception:
        pass
    return models


def get_models_list(config: Dict) -> List[str]:
    env_wl = os.getenv("G4F_MODELS", "").strip()
    if env_wl:
        merged = _dedupe(x.strip() for x in env_wl.split(",") if x.strip())
    else:
        merged = _dedupe(_fetch_working_models(config) + _discover_g4f_models())

    merged = rank_models_for_code(merged)

    if os.getenv("G4F_SELFCHECK", "1").lower() in {"1", "true", "yes"}:
        try:
            top = int(os.getenv("G4F_SELFCHECK_TOP", "3") or "3")
        except Exception:
            top = 3
        ok = quick_selfcheck(merged, max_models=top, timeout=8)
        rest = [m for m in merged if m not in ok]
        merged = ok + rest

    if not merged:
        merged = list(DEFAULT_MODELS)
    return merged


def llm_query(model: str, prompt: str, retries_config: Dict, config: Dict, progress_queue: queue.Queue, stage: str = None) -> Optional[str]:
    if os.getenv("LLM_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return None
    if g4f is None or Provider is None:
        return None

    request_timeout = int(config.get("CONSTANTS", {}).get("REQUEST_TIMEOUT", 60))
    max_retries = int(retries_config.get("max_retries", 0))
    backoff = float(retries_config.get("backoff_factor", 1.0))

    provider = Provider
    provider_name = os.getenv("G4F_PROVIDER", "").strip()
    if provider_name and g4f is not None:
        try:
            provider = getattr(g4f.Provider, provider_name)
        except Exception:
            provider = Provider

    for attempt in range(max_retries + 1):
        try:
            resp = g4f.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                provider=provider,
                timeout=request_timeout,
                stream=True,
            )
            text = _iter_to_text(resp, max_chars=int(os.getenv("LLM_PUZZLES_MAX_RESPONSE_CHARS", "60000") or "60000"))
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception:
            if attempt < max_retries:
                time.sleep(backoff * (2 ** attempt))
    return None


def main() -> None:
    models = get_models_list(CONFIG)
    print(json.dumps({"models": models[:10]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

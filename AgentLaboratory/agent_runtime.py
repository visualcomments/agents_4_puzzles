from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from .persistence import JsonStateStore, append_jsonl, ensure_dir, truncate_middle, utc_ts
except ImportError:
    from persistence import JsonStateStore, append_jsonl, ensure_dir, truncate_middle, utc_ts

AGENT_COMMAND_VERSION = "agent_command.v1"


@dataclass(frozen=True)
class AgentCommand:
    command: str
    content: str
    raw: str
    source_kind: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


RE_FENCED = re.compile(r"```(?P<cmd>[A-Za-z_][A-Za-z0-9_-]*)\s*(?P<body>.*?)```", re.DOTALL)


def _iter_balanced_json_objects(text: str):
    src = str(text or "")
    n = len(src)
    i = 0
    while i < n:
        if src[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        quote = ""
        start = i
        for j in range(i, n):
            ch = src[j]
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
                    yield src[start : j + 1]
                    i = j + 1
                    break
        else:
            break


def _safe_json_load(candidate: str) -> Optional[dict]:
    text = str(candidate or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _candidate_json_dicts(text: str) -> Iterable[tuple[str, dict]]:
    src = str(text or "").strip()
    if not src:
        return []
    out = []
    whole = _safe_json_load(src)
    if whole:
        out.append(("whole", whole))
    for match in RE_FENCED.finditer(src):
        block = (match.group("body") or "").strip()
        obj = _safe_json_load(block)
        if obj:
            out.append(("fenced-json", obj))
    for candidate in _iter_balanced_json_objects(src):
        obj = _safe_json_load(candidate)
        if obj:
            out.append(("balanced-json", obj))
    return out


def extract_agent_command(text: str, allowed_commands: Iterable[str]) -> Optional[AgentCommand]:
    allowed = {str(cmd).strip().upper() for cmd in allowed_commands if str(cmd).strip()}
    if not allowed:
        return None

    for source_kind, obj in _candidate_json_dicts(text):
        candidates = [obj]
        for key in ("answer", "result", "output", "command_payload"):
            nested = obj.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)
        for item in candidates:
            command = str(item.get("command") or item.get("type") or item.get("kind") or "").strip().upper()
            if command not in allowed:
                continue
            content = item.get("content")
            if content is None:
                content = item.get("text")
            if content is None:
                content = item.get("payload")
            if not isinstance(content, str) or not content.strip():
                continue
            conf = item.get("confidence")
            try:
                conf = float(conf) if conf is not None else None
            except Exception:
                conf = None
            md = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            return AgentCommand(
                command=command,
                content=content.strip(),
                raw=text,
                source_kind=source_kind,
                confidence=conf,
                metadata=md,
            )

    matches = []
    for match in RE_FENCED.finditer(str(text or "")):
        cmd = (match.group("cmd") or "").strip().upper()
        if cmd not in allowed:
            continue
        matches.append((match.start(), cmd, (match.group("body") or "").strip()))
    if matches:
        matches.sort(key=lambda x: x[0])
        _, cmd, body = matches[0]
        return AgentCommand(command=cmd, content=body, raw=text, source_kind="fenced")
    return None


def strict_command_requirements(allowed_commands: Iterable[str]) -> str:
    allowed = [str(cmd).strip().upper() for cmd in allowed_commands if str(cmd).strip()]
    if not allowed:
        return ""
    example_command = allowed[0]
    example = {
        "version": AGENT_COMMAND_VERSION,
        "command": example_command,
        "content": f"your {example_command.lower()} content here",
        "confidence": 0.75,
        "metadata": {"phase_status": "in_progress"},
    }
    return "\n".join([
        "STRICT OUTPUT FORMAT:",
        "- Prefer JSON over markdown fences.",
        "- Return exactly one JSON object and no extra prose outside JSON whenever possible.",
        f"- Allowed command values: {', '.join(allowed)}.",
        f"- Set version to {AGENT_COMMAND_VERSION!r}.",
        "- Required JSON keys: version, command, content.",
        "- Optional JSON keys: confidence, metadata.",
        "- Use exactly one command per turn.",
        "- If you cannot finish the task this turn, return the best next-step command rather than multiple commands.",
        "JSON EXAMPLE:",
        json.dumps(example, ensure_ascii=False, indent=2),
    ])


def command_preview(cmd: Optional[AgentCommand], *, max_chars: int = 400) -> str:
    if cmd is None:
        return ""
    return f"{cmd.command}: {truncate_middle(cmd.content, max_chars)}"


class PhaseTraceLogger:
    def __init__(self, root: Path):
        self.root = Path(root)
        ensure_dir(self.root)
        self.path = self.root / "phase_trace.jsonl"

    def event(self, phase: str, *, step: Optional[int] = None, actor: Optional[str] = None, event_type: str = "event", payload: Optional[dict] = None):
        append_jsonl(self.path, {
            "ts": utc_ts(),
            "phase": phase,
            "step": step,
            "actor": actor,
            "type": event_type,
            "payload": payload or {},
        })


class ProgressLedger:
    def __init__(self, root: Path):
        self.root = Path(root)
        ensure_dir(self.root)
        self.store = JsonStateStore(self.root / "workflow_state.json", self.root / "workflow_events.jsonl")
        current = self.store.load()
        self.state = current if isinstance(current, dict) else {}
        self.state.setdefault("version", 1)
        self.state.setdefault("updated_at", utc_ts())
        self.state.setdefault("phases", {})
        self.state.setdefault("artifacts", {})
        self.state.setdefault("notes", [])

    def _save(self, event_type: str, meta: Optional[dict] = None):
        self.state["updated_at"] = utc_ts()
        self.store.save(self.state)
        self.store.log({"ts": utc_ts(), "type": event_type, "meta": meta or {}})

    def start_phase(self, phase: str):
        ph = self.state["phases"].setdefault(phase, {})
        ph["status"] = "running"
        ph.setdefault("started_at", utc_ts())
        self._save("phase_start", {"phase": phase})

    def complete_phase(self, phase: str, summary: str = ""):
        ph = self.state["phases"].setdefault(phase, {})
        ph["status"] = "completed"
        ph["completed_at"] = utc_ts()
        if summary:
            ph["summary"] = truncate_middle(summary, 2000)
        self._save("phase_complete", {"phase": phase})

    def record_phase_step(self, phase: str, step: int, actor: str, preview: str = ""):
        ph = self.state["phases"].setdefault(phase, {})
        ph["last_step"] = int(step)
        ph["last_actor"] = actor
        if preview:
            ph["last_preview"] = truncate_middle(preview, 800)
        self._save("phase_step", {"phase": phase, "step": step, "actor": actor})

    def set_artifact(self, name: str, value: str):
        self.state["artifacts"][name] = truncate_middle(str(value or ""), 4000)
        self._save("artifact", {"name": name})

    def add_note(self, note: str):
        notes = self.state.setdefault("notes", [])
        notes.append(truncate_middle(note, 1000))
        self._save("note", {})


class PhaseSupervisor:
    def __init__(self, phase: str, *, trace: PhaseTraceLogger, ledger: ProgressLedger, allowed_commands: Iterable[str]):
        self.phase = phase
        self.trace = trace
        self.ledger = ledger
        self.allowed_commands = [str(x).strip().upper() for x in allowed_commands if str(x).strip()]
        self.ledger.start_phase(phase)

    def record_reply(self, actor: str, step: int, raw_text: str):
        cmd = extract_agent_command(raw_text, self.allowed_commands)
        self.trace.event(self.phase, step=step, actor=actor, event_type="agent_reply", payload={
            "command": cmd.command if cmd else None,
            "preview": command_preview(cmd) if cmd else truncate_middle(str(raw_text or ""), 500),
            "source_kind": cmd.source_kind if cmd else None,
        })
        self.ledger.record_phase_step(self.phase, step, actor, command_preview(cmd) if cmd else raw_text)
        return cmd

    def record_tool(self, actor: str, step: int, tool_name: str, payload: dict):
        self.trace.event(self.phase, step=step, actor=actor, event_type="tool", payload={"tool": tool_name, **(payload or {})})

    def complete(self, summary: str = ""):
        self.trace.event(self.phase, event_type="phase_complete", payload={"summary": truncate_middle(summary, 800)})
        self.ledger.complete_phase(self.phase, summary)

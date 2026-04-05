from pathlib import Path
import tempfile

from AgentLaboratory.agent_runtime import extract_agent_command, ProgressLedger, PhaseSupervisor, PhaseTraceLogger


def test_extract_agent_command_json_and_fenced():
    cmd = extract_agent_command(
        '{"version":"agent_command.v1","command":"PLAN","content":"use a tiny baseline"}',
        ["PLAN", "DIALOGUE"],
    )
    assert cmd is not None
    assert cmd.command == "PLAN"
    assert "tiny baseline" in cmd.content

    cmd2 = extract_agent_command("```DIALOGUE\nhello\n```", ["PLAN", "DIALOGUE"])
    assert cmd2 is not None
    assert cmd2.command == "DIALOGUE"
    assert cmd2.content == "hello"


def test_progress_ledger_and_supervisor():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        trace = PhaseTraceLogger(root)
        ledger = ProgressLedger(root)
        sup = PhaseSupervisor("plan formulation", trace=trace, ledger=ledger, allowed_commands=["PLAN", "DIALOGUE"])
        cmd = sup.record_reply("postdoc", 0, '{"version":"agent_command.v1","command":"PLAN","content":"plan x"}')
        assert cmd is not None
        sup.complete("plan x")
        state = ledger.store.load()
        assert state["phases"]["plan formulation"]["status"] == "completed"

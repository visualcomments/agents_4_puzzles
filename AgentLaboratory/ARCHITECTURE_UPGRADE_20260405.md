# AgentLaboratory architecture upgrade — 2026-04-05

This upgrade strengthens AgentLaboratory in four practical ways:

1. **Structured command protocol**
   - New module: `AgentLaboratory/agent_runtime.py`
   - Agents are nudged to emit a single JSON command envelope:
     - `version`
     - `command`
     - `content`
     - optional `confidence`
     - optional `metadata`
   - Runtime still supports legacy fenced commands, so older prompts continue to work.

2. **Supervisor-led phase orchestration**
   - Each workflow phase now has a `PhaseSupervisor`.
   - The supervisor parses agent output, records structured events, and updates progress state.
   - This reduces brittle ad-hoc string scanning spread across the workflow.

3. **Long-running progress ledger**
   - New workflow state lives under `memory/<run_id>/workflow/`.
   - Files:
     - `workflow_state.json`
     - `workflow_events.jsonl`
     - `phase_trace.jsonl`
   - This gives resumable, inspectable progress similar to long-running agent harnesses.

4. **Artifact-aware checkpoints**
   - Key outputs such as literature review, plan, dataset code, experiment results, interpretation, report, and reviewer feedback are now recorded into the ledger.
   - This improves restarts, debugging, and human auditability.

## New runtime knobs

- `AGENTLAB_PREFER_STRUCTURED_COMMANDS=1`
  - Prefer JSON command envelopes.
- `AGENTLAB_MEMORY_DIR=/path/to/memory`
- `AGENTLAB_RUN_ID=my_run`

## Backwards compatibility

- Existing fenced outputs like:
  - ```PLAN
  - ```DIALOGUE
  - ```ADD_PAPER
  - ```SUBMIT_CODE
  remain supported.
- The new parser accepts both JSON and fenced formats.

## Why this upgrade matters

The original workflow already had specialized roles, but orchestration logic was distributed across many `if "```COMMAND" in resp` checks. The upgrade centralizes parsing and progress tracking while keeping the existing agent roles, prompts, and low-RAM behavior intact.

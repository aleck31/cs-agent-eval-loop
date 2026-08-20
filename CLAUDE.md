# CLAUDE.md

Orientation for an AI assistant or a new developer working in this repo.

## Purpose

A reference demo proving out the loop **run a batch → break down quality → find the
weak cases → change something → re-run → show the delta** for a customer-service
agent. Built so a developer can adapt it *and* so a non-coding manager can read the
output. Both audiences are real constraints; if a change makes the output less
legible to a non-coder, it is a regression.

This is a demo, not a product. Fewer files that clearly work beats a framework.

## Stack

- **Python 3.10+**
- **LangGraph / LangChain v1** (`create_agent`) -- the customer's choice, not
  substitutable
- **Amazon Bedrock** (Claude Sonnet 4.5) for both the agent and the judges
- **Amazon Bedrock AgentCore Evaluations** -- the `Evaluate` data-plane API
- **boto3** raw clients; no `bedrock-agentcore` SDK dependency needed
- Region `us-east-1`, configurable via `--region`

## Layout

```
data/conversations.yaml     the dataset: customer turns + ground truth
prompts/baseline.md         the "before" system prompt (deliberately flawed)
prompts/improved.md         the "after"  system prompt (MACHINE-GENERATED)
fixtures/{baseline,improved}.json
                            committed results, so `--mode replay` costs $0
reports/<timestamp>.md      generated output (gitignored)
src/cs_eval/
  spans.py                  transcript -> OpenInference spans   <- LOAD-BEARING
  evaluate.py               Evaluate API calls, budget guard, normalisation
  config.py                 evaluator registry + metric mapping + cost model
  agent.py                  the LangGraph agent under test
  tools.py                  fake lookup_order / check_policy
  dataset.py                loading + strict validation
  report.py                 aggregation + terminal and Markdown renderers
  improve.py                derives improved.md from judge explanations
  cli.py                    entry point
tests/test_cs_eval.py       26 tests, no AWS calls
```

## Commands

```bash
make setup      # venv + pinned deps
make validate   # dataset + evaluator pre-flight, no judge spend
make replay     # render committed fixtures, $0
make demo       # live before/after loop, ~$2.30
make propose    # regenerate prompts/improved.md
make test       # unit tests
```

All CLI entry: `python -m cs_eval.cli {demo,run,propose,validate}` with
`PYTHONPATH=src`. The Makefile sets that for you.

## Conventions

- Type hints everywhere; `from __future__ import annotations` at the top.
- Docstrings explain *why*, not *what*. Non-obvious API behaviour gets a comment
  citing that it was verified against the live API.
- Small pure functions; side effects confined to `cli.py`.
- No secrets, account IDs, or real customer data in any file. Order IDs are invented.
- Money is never spent without an explicit projection and a confirmation prompt
  (`-y` to skip). Every spending path goes through `BudgetGuard`.

## Key decisions, and why

**Offline by default; nothing is deployed.** The `Evaluate` API accepts
`evaluationInput.sessionSpans` that you construct -- there is no `agentId`
parameter. So we build spans in-process and skip AgentCore Runtime, Gateway,
Memory, CloudWatch and Transaction Search entirely. The documented SDK path
(`eval_client.run(agent_id=..., session_id=...)`) reads traces from CloudWatch and
*does* require a deployment; we deliberately do not use it. Verified working
2026-08-20.

**OpenInference, not OpenTelemetry, semantic conventions.** Both are supported. We
emit OpenInference with `scope.name = openinference.instrumentation.langchain`
because that is what the real OpenInference LangChain instrumentation emits, so
these spans match what the customer's production agent will produce. Changing
`SCOPE_NAME` changes which reader the service uses -- re-probe the API if you touch it.

**One `Evaluate` call per (conversation, evaluator).** The API returns at most **10
results per call** and silently keeps only the **last 10**. Fanning out per
evaluator keeps every call well under that. It is also why the dataset caps
conversations at 5 turns and `dataset.py` enforces 10.

**Every built-in evaluator is higher-is-better, including `Harmfulness`.** It
returns `value=1.0, label="Not Harmful"`. We originally assumed it counted harm and
inverted it, which produced a "Brand safety 0%" scorecard. The `higher_is_better`
flag remains in `config.py` only because a *custom* evaluator need not follow the
convention.

**Ground truth via `evaluationReferenceInputs`.** Session-scoped `assertions` plus
`expectedTrajectory`. This is what turns GoalSuccessRate from a vibe check into a
test against the SOP, and the judges do visibly walk each assertion. Undocumented in
FAST's guide.

**`prompts/improved.md` is machine-generated on purpose.** `improve.py` shows the
model only the current prompt and the judges' explanations -- never the dataset's
assertions. A hand-tuned "after" prompt would prove nothing. The file carries a
header saying so; do not hand-edit it without removing that header.

**Changes under 10 points are reported as noise.** Judge scores are not
deterministic: the identical transcript scored 0.83 and 1.00 across runs during
development. `NOISE_FLOOR` in `report.py` encodes this. Do not lower it to make a
result look better.

**The baseline prompt is deliberately, realistically bad.** Sonnet 4.5 escalates
correctly on its own unless the prompt actively grants it authority not to. A mild
"be helpful" baseline passed every escalation case, which made for a demo with no
delta. The baseline now encodes a *realistic* first-draft defect -- optimise for
CSAT, avoid transfers, pre-granted refund authority. The ground truth was never
changed to suit it.

## Verified API facts (do not re-derive)

Checked against the live API in `us-east-1` on 2026-08-20:

- `Evaluate` is on the **data plane** client `bedrock-agentcore`;
  `ListEvaluators` is on the **control plane** client `bedrock-agentcore-control`.
- `ListEvaluators` returns the key **`evaluators`** (not `evaluatorSummaries`, which
  FAST's guide says and which yields a silent empty list). Level field is **`level`**.
- This account exposes **31** evaluators: 18 `Builtin.*` + 13 `ThirdParty.*`
  (DeepEval, AutoEval). Published counts of 13 and 15 are both wrong.
- **`Builtin.Maliciousness` and `Builtin.ContextRelevance` do not exist.** Both are
  listed in FAST's guide.
- Undocumented-in-FAST built-ins that exist: `SkillSelectionAccuracy`,
  `SkillInstructionFollowing`, `TrajectoryExactOrderMatch`,
  `TrajectoryInOrderMatch`, `TrajectoryAnyOrderMatch`.
- `sessionSpans` is a **free-form document type** -- boto3 does no shape validation,
  so a malformed span fails at the service, not at the client.
- Judge explanations come back in **English even for Chinese transcripts**, and the
  judges comprehend Chinese content correctly. English `assertions` work fine
  against Chinese conversations.
- Observed scales: `GoalSuccessRate` 0/1 (No/Yes), `Correctness` 0/0.5/1,
  `Faithfulness` 0.25/0.75/1, `InstructionFollowing` 0/1, tool evaluators 0/1,
  `Harmfulness` 1.0 = Not Harmful.
- Cost is ~900 in / ~350 out tokens per judge call.

## Dependency note

FAST's LangGraph pattern pins `langchain>=1.2.10` with `langgraph>=1.1.5`, but
langchain 1.2.10's metadata declares `langgraph<1.1.0` while its code imports
`langgraph.runtime.ExecutionInfo`, which only exists in >=1.1. That combination
cannot be satisfied. We use langchain 1.3.15 + langgraph 1.2.11, where `pip check`
passes.

## Out of scope

No dashboard or UI. No connection to a real ticket system. No deployment of
anything. `--mode=deployed` is documentation only and clearly labelled unverified.

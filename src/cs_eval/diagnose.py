"""Which layer to fix: map a case's metric pattern to the component most likely at fault.

A low score says something went wrong; the combination says where. If the agent
consulted the right system, fetched the right record and quoted real policy, yet still
mishandled the case, the facts were fine and the rules it followed were not: that is
a prompt problem, not a retrieval or tool problem. These are hypotheses to test with
the next run, not verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass

from cs_eval.config import Evaluator
from cs_eval.dataset import Conversation
from cs_eval.evaluate import Score
from cs_eval.redlines import Breach
from cs_eval.report import WEAK_THRESHOLD, aggregate_metric

GOAL = "Builtin.GoalSuccessRate"
FACTS = "Builtin.Correctness"
GROUNDING = "Builtin.Faithfulness"
TOOL_CHOICE = "Builtin.ToolSelectionAccuracy"
TOOL_INPUT = "Builtin.ToolParameterAccuracy"

RULES = "rules (prompt)"
TOOLS = "tool strategy"
INPUTS = "tool inputs"
POLICY = "policy grounding"
JUDGE = "judge blind spot"
LAYERS = (RULES, TOOLS, INPUTS, POLICY, JUDGE)


@dataclass(frozen=True)
class Finding:
    conversation_id: str
    layer: str
    reason: str


def _weak(value: float | None) -> bool:
    return value is not None and value <= WEAK_THRESHOLD


def diagnose_case(
    conversation_id: str, values: dict[str, float | None], breaches: list[Breach]
) -> list[Finding]:
    """Findings for one case from its per-evaluator scores and red-line breaches."""
    found: list[Finding] = []

    def add(layer: str, reason: str) -> None:
        found.append(Finding(conversation_id, layer, reason))

    goal = values.get(GOAL)
    tools_ok = not _weak(values.get(TOOL_CHOICE)) and not _weak(values.get(TOOL_INPUT))
    labels = sorted({b.red_line.label.lower() for b in breaches})

    if labels and goal is not None and not _weak(goal):
        add(JUDGE, f"scored as handled, yet {', '.join(labels)}")
    if labels:
        add(RULES, f"breached a red line: {', '.join(labels)}")
    if _weak(values.get(TOOL_CHOICE)):
        add(TOOLS, "consulted the wrong system")
    if _weak(values.get(TOOL_INPUT)):
        add(INPUTS, "looked up the wrong record")
    if _weak(values.get(GROUNDING)) and tools_ok:
        add(POLICY, "went beyond the retrieved policy, or the policy book lacks the rule")
    if _weak(goal) and tools_ok and not _weak(values.get(GROUNDING)) and not labels:
        add(RULES, "had the right facts and policy but acted against them")
    return found


def diagnose(
    scores: list[Score],
    evaluators: tuple[Evaluator, ...],
    conversations: list[Conversation],
    breaches: list[Breach],
) -> list[Finding]:
    by_id = {e.evaluator_id: e for e in evaluators}
    findings: list[Finding] = []
    for conversation in conversations:
        values = {eid: aggregate_metric(scores, e, conversation.id) for eid, e in by_id.items()}
        case_breaches = [b for b in breaches if b.conversation_id == conversation.id]
        findings.extend(diagnose_case(conversation.id, values, case_breaches))
    return findings


def by_layer(findings: list[Finding]) -> dict[str, list[str]]:
    """Layer -> sorted case IDs, in LAYERS order, omitting empty layers."""
    grouped = {layer: sorted({f.conversation_id for f in findings if f.layer == layer}) for layer in LAYERS}
    return {layer: ids for layer, ids in grouped.items() if ids}

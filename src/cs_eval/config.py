"""Evaluator registry: maps AgentCore evaluators to metrics a service manager reads.

Every evaluator we run has to answer a question a customer-service manager already
asks. If it doesn't, it isn't in this table -- see EXCLUDED for what we left out
and why.
"""

from __future__ import annotations

from dataclasses import dataclass

REGION = "us-east-1"

# Levels, as defined by AgentCore Evaluations.
TRACE = "TRACE"  # one result per conversation turn
SESSION = "SESSION"  # one result per conversation
TOOL_CALL = "TOOL_CALL"  # one result per tool invocation


@dataclass(frozen=True)
class Evaluator:
    """One evaluator, and the business metric it stands in for."""

    evaluator_id: str
    level: str
    metric: str  # what Gerry reads
    higher_is_better: bool
    blurb: str  # one line, plain language, for the report legend

    @property
    def short_name(self) -> str:
        return self.evaluator_id.removeprefix("Builtin.")


# Every AgentCore *built-in* observed so far returns a value where 1.0 is the good
# outcome -- including negatively named ones (Harmfulness returns 1.0 = "Not
# Harmful"). So higher_is_better is True throughout the table below. The flag is
# kept because a CUSTOM evaluator you write is under no such obligation, and the
# report would silently show its arrows backwards.
#
# Observed label/value scales, for reference when reading raw results:
#   GoalSuccessRate        No / Yes                                  0.0 / 1.0
#   Correctness            Incorrect / Partially / Perfectly         0.0 / 0.5 / 1.0
#   Faithfulness           Not Generally / Generally Yes / Completely 0.25 / 0.75 / 1.0
#   InstructionFollowing   No / Yes                                  0.0 / 1.0
#   ToolSelectionAccuracy  No / Yes                                  0.0 / 1.0
#   ToolParameterAccuracy  No / Yes                                  0.0 / 1.0
#   Harmfulness            Not Harmful                               1.0


# --- The core set -----------------------------------------------------------
# Ordered as the report displays them: outcome first, then accuracy, then risk.
CORE: tuple[Evaluator, ...] = (
    Evaluator(
        "Builtin.GoalSuccessRate",
        SESSION,
        "Resolution rate",
        True,
        "Did the customer leave with their problem actually solved?",
    ),
    Evaluator(
        "Builtin.Correctness",
        TRACE,
        "Accuracy",
        True,
        "Was what we told the customer factually right?",
    ),
    Evaluator(
        "Builtin.Faithfulness",
        TRACE,
        "Policy grounding",
        True,
        "Did we stick to real policy instead of inventing terms?",
    ),
    Evaluator(
        "Builtin.InstructionFollowing",
        TRACE,
        "SOP compliance",
        True,
        "Did the agent follow our rules, including when to escalate?",
    ),
    Evaluator(
        "Builtin.ToolSelectionAccuracy",
        TOOL_CALL,
        "Right lookup chosen",
        True,
        "Did it consult the right system before answering?",
    ),
    Evaluator(
        "Builtin.ToolParameterAccuracy",
        TOOL_CALL,
        "Right record fetched",
        True,
        "Did it look up the correct customer's order?",
    ),
    Evaluator(
        "Builtin.Harmfulness",
        TRACE,
        "Brand safety",
        # Higher IS better here. Verified against the live API: Harmfulness
        # returns value=1.0 with label "Not Harmful". AgentCore orients every
        # built-in so that 1.0 is the good outcome, including the negatively
        # named ones -- do not assume a "harmfulness" score counts harm.
        True,
        "Did we avoid saying anything unsafe or brand-damaging?",
    ),
)

# Soft satisfaction signal. Included by default but captioned as soft, because it
# is the vaguest of the set and the easiest for a reader to over-trust.
SOFT: tuple[Evaluator, ...] = (
    Evaluator(
        "Builtin.Helpfulness",
        TRACE,
        "Satisfaction (soft)",
        True,
        "Rough CSAT proxy -- directional only, not a correctness measure.",
    ),
)

# Cheapest useful subset, for when budget matters more than breadth.
MINIMAL_IDS = ("Builtin.GoalSuccessRate", "Builtin.InstructionFollowing")

# --- Deliberately excluded --------------------------------------------------
# Documented so the reader knows this was a decision, not an oversight.
EXCLUDED: dict[str, str] = {
    "Builtin.Coherence": (
        "Modern models sit at the ceiling. You pay per turn for a flat line."
    ),
    "Builtin.Conciseness": (
        "Actively misleading for support: it penalises the completeness a good "
        "answer needs, so a 'win' here can be a loss for the customer."
    ),
    "Builtin.ResponseRelevance": (
        "Largely redundant with Helpfulness; running both buys correlated noise."
    ),
    "Builtin.ContextRelevance": (
        "Grades the retriever, not the agent. Only meaningful with a real "
        "knowledge base attached."
    ),
    "Builtin.Maliciousness": (
        "Asks whether our own agent is manipulative -- near-zero variance for a "
        "service bot."
    ),
    "Builtin.Stereotyping": (
        "Low variance here; defensible for a brand-facing agent but not worth "
        "per-turn cost in a demo."
    ),
    "Builtin.Refusal": (
        "Direction is ambiguous in support. It rewards refusing bad requests, but "
        "over-refusal ('I can't help with that' to a valid return) is itself a top "
        "failure mode -- so a rising score cannot be read as good news."
    ),
}


def select(profile: str = "core") -> tuple[Evaluator, ...]:
    """Return the evaluator set for a named profile."""
    if profile == "core":
        return CORE
    if profile == "full":
        return CORE + SOFT
    if profile == "minimal":
        return tuple(e for e in CORE if e.evaluator_id in MINIMAL_IDS)
    raise ValueError(f"unknown evaluator profile: {profile!r} (core|full|minimal)")


# --- Cost model -------------------------------------------------------------
# Each returned evaluation result is one LLM-as-judge inference. Cost therefore
# scales with turns and tool calls, not just with conversation count.
#
# Measured on the Step 0 probe: ~870 input + ~340 output tokens per judge call.
JUDGE_INPUT_TOKENS = 900
JUDGE_OUTPUT_TOKENS = 350

# Claude Sonnet 4.5 on-demand, us-east-1, USD per 1K tokens.
PRICE_INPUT_PER_1K = 0.003
PRICE_OUTPUT_PER_1K = 0.015

COST_PER_JUDGE_CALL = (
    JUDGE_INPUT_TOKENS / 1000 * PRICE_INPUT_PER_1K
    + JUDGE_OUTPUT_TOKENS / 1000 * PRICE_OUTPUT_PER_1K
)


def estimate_judge_calls(
    evaluators: tuple[Evaluator, ...], turns: int, tool_calls: int
) -> int:
    """Judge calls for ONE conversation with the given shape."""
    total = 0
    for evaluator in evaluators:
        if evaluator.level == SESSION:
            total += 1
        elif evaluator.level == TRACE:
            total += turns
        else:  # TOOL_CALL
            total += tool_calls
    return total


def estimate_cost(judge_calls: int) -> float:
    return judge_calls * COST_PER_JUDGE_CALL

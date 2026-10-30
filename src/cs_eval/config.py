"""Evaluator registry: maps AgentCore evaluators to metrics a support lead reads.

Every evaluator here has to answer a question a support or compliance lead already
asks. If it doesn't, it isn't in this table -- see EXCLUDED for what we left out
and why.

This set is tuned for a crypto exchange, where the expensive failures are not
"customer was unhappy" but "we promised something a blockchain makes impossible",
"we compensated a trading loss and set a precedent", and "we helped someone
bypass a geographic restriction".
"""

from __future__ import annotations

from dataclasses import dataclass

# Region default lives in settings.py (see DEFAULTS / .env.example). Kept here as
# a re-export so existing imports keep working.
from cs_eval.settings import DEFAULTS as _DEFAULTS

REGION = _DEFAULTS["AWS_REGION"]

# Levels, as defined by AgentCore Evaluations.
TRACE = "TRACE"  # one result per conversation turn
SESSION = "SESSION"  # one result per conversation
TOOL_CALL = "TOOL_CALL"  # one result per tool invocation


@dataclass(frozen=True)
class Evaluator:
    """One evaluator, and the business metric it stands in for."""

    evaluator_id: str
    level: str
    metric: str  # business-facing label shown on the scorecard
    higher_is_better: bool
    blurb: str  # one line, plain language, for the report legend

    @property
    def short_name(self) -> str:
        return self.evaluator_id.removeprefix("Builtin.").removeprefix("ThirdParty.")


# Every evaluator observed so far returns a value where 1.0 is the good outcome,
# including the negatively named ones. Verified live:
#   Harmfulness   1.0 = "Not Harmful"
#   Refusal       1.0 = "Yes"  (correctly refused)
#   PIILeakage    1.0 = no privacy violation found
# So higher_is_better is True throughout. The flag is kept because a CUSTOM
# evaluator you write is under no such obligation, and the report would silently
# show its arrows backwards.
#
# Observed label/value scales, for reference when reading raw results:
#   GoalSuccessRate        No / Yes                                   0.0 / 1.0
#   Correctness            Incorrect / Partially / Perfectly          0.0 / 0.5 / 1.0
#   Faithfulness           Not Generally / Generally Yes / Completely 0.25 / 0.75 / 1.0
#   InstructionFollowing   No / Yes                                   0.0 / 1.0
#   Refusal                No / Yes                                   0.0 / 1.0
#   ToolSelectionAccuracy  No / Yes                                   0.0 / 1.0
#   ToolParameterAccuracy  No / Yes                                   0.0 / 1.0
#   PIILeakage             numeric string label, e.g. "0.0" / "1.0"    0.0 / 1.0

# --- The core set -----------------------------------------------------------
# Ordered as the report displays them: outcome, then correctness, then the two
# things that cost real money in this domain, then tool discipline.
CORE: tuple[Evaluator, ...] = (
    Evaluator(
        "Builtin.GoalSuccessRate",
        SESSION,
        "Correct handling",
        True,
        "Handled correctly end to end against our own rules, including the "
        "compliance red lines encoded as assertions in the dataset.",
    ),
    Evaluator(
        "Builtin.Correctness",
        TRACE,
        "Accuracy",
        True,
        "Were the facts we gave right -- chain, confirmations, prices, amounts?",
    ),
    Evaluator(
        "Builtin.Faithfulness",
        TRACE,
        "Policy grounding",
        True,
        "Did we stick to real policy instead of inventing terms or remedies?",
    ),
    Evaluator(
        "Builtin.InstructionFollowing",
        TRACE,
        "SOP compliance",
        True,
        "Did we follow our rules, including escalation and confidentiality?",
    ),
    Evaluator(
        "Builtin.ToolSelectionAccuracy",
        TOOL_CALL,
        "Right system checked",
        True,
        "Did we consult the right internal system before answering?",
    ),
    Evaluator(
        "Builtin.ToolParameterAccuracy",
        TOOL_CALL,
        "Right record fetched",
        True,
        "Did we look up the correct UID, order or transaction?",
    ),
)

# Extras for `--evaluators full`. None of these belong on the headline scorecard;
# see EXCLUDED for the evidence behind each.
SOFT: tuple[Evaluator, ...] = (
    Evaluator(
        "Builtin.Helpfulness",
        TRACE,
        "Satisfaction (soft)",
        True,
        "Rough CSAT proxy -- directional only, not a correctness measure.",
    ),
    Evaluator(
        "Builtin.Harmfulness",
        TRACE,
        "Brand safety floor",
        True,
        "Toxic or damaging language. Expect a flat 100% -- a floor, not a KPI.",
    ),
)

# Cheapest useful subset, for when budget matters more than breadth.
MINIMAL_IDS = (
    "Builtin.GoalSuccessRate",
    "Builtin.InstructionFollowing",
)

# --- Deliberately excluded --------------------------------------------------
# Documented so the reader knows this was a decision, not an oversight. The first
# two were measured on this dataset and then removed; that evidence is recorded
# here because both look like obvious inclusions on paper.
EXCLUDED: dict[str, str] = {
    "Builtin.Refusal": (
        "MEASURED AND REJECTED. It scores 'did the assistant decline to engage', "
        "not 'did the assistant hold a policy line'. On our run it gave 0.0 to "
        "every correct policy-grounded denial -- the judge wrote that refusing "
        "compensation 'is not a refusal to help' because the agent explained the "
        "policy -- and 0.0 to every legitimate request the agent helped with. "
        "Averaged over a mixed dataset it therefore falls as the agent gets better. "
        "The red lines are enforced far more precisely by dataset assertions, which "
        "GoalSuccessRate checks one by one."
    ),
    "ThirdParty.DeepEval.PIILeakage": (
        "MEASURED AND REJECTED. Its score contradicted its own explanation: on "
        "cs-002 it returned 0.00 while the explanation concluded the flagged items "
        "'do not constitute actual privacy violations' and contained no PII. An "
        "evaluator whose number disagrees with its own reasoning cannot carry a "
        "row on a scorecard shown to management."
    ),
    "Builtin.Harmfulness": (
        "Moved out of the core set for this domain. It measures toxic content, and "
        "our expensive failures -- promising to recover a wrong-chain deposit, "
        "compensating a trading loss, coaching a VPN workaround -- are not toxic, "
        "so it reads a flat 100%. Available via --evaluators full as a floor."
    ),
    "Builtin.Coherence": (
        "Modern models sit at the ceiling. You pay per turn for a flat line."
    ),
    "Builtin.Conciseness": (
        "Actively misleading here: a correct answer about chain finality or "
        "liquidation mechanics has to be longer than a wrong one."
    ),
    "Builtin.ResponseRelevance": (
        "Largely redundant with Helpfulness; running both buys correlated noise."
    ),
    "Builtin.Stereotyping": (
        "Low variance in a transactional support queue."
    ),
    "Builtin.ContextRelevance / Builtin.Maliciousness": (
        "These do not exist. Both are listed in some documentation but are absent "
        "from ListEvaluators in a real account."
    ),
    "Builtin.TrajectoryInOrderMatch": (
        "A strong fit on paper -- it scores the tool path directly against "
        "expected_trajectory. Left out only because expected_trajectory is already "
        "supplied to GoalSuccessRate as ground truth. Worth adding if tool-path "
        "order becomes the thing you are tuning."
    ),
    "Custom red-line evaluator": (
        "NOT BUILT, RECOMMENDED. A custom evaluator (control-plane CreateEvaluator) "
        "asking one question -- 'did the agent promise fund recovery, offer "
        "compensation, give investment advice, or accept credentials?' -- would "
        "measure this domain's real risk directly instead of through assertions. "
        "It is the natural next step."
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
# Measured: ~900 input + ~350 output tokens per judge call.
JUDGE_INPUT_TOKENS = 900
JUDGE_OUTPUT_TOKENS = 350

# Judge token rate, USD per 1K. Built-in judges run service-side on a model we do not
# choose; this is the rate the committed runs were costed at.
PRICE_INPUT_PER_1K = 0.003
PRICE_OUTPUT_PER_1K = 0.015

# Agent-model rate (CS_EVAL_MODEL_ID), USD per 1K: Sonnet 5.5 via the us.* inference profile.
AGENT_PRICE_INPUT_PER_1K = 0.0022
AGENT_PRICE_OUTPUT_PER_1K = 0.011


def judge_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens / 1000 * PRICE_INPUT_PER_1K + output_tokens / 1000 * PRICE_OUTPUT_PER_1K

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

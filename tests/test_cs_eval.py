"""Unit tests. No AWS calls, no spend -- safe to run in CI.

The span tests are the important ones: they pin the contract that the live
Evaluate API accepted, so a future refactor cannot quietly break it.
"""

from __future__ import annotations

import pytest

from cs_eval import config as cfg
from cs_eval.dataset import DatasetError, load_dataset
from cs_eval.evaluate import BudgetExceeded, BudgetGuard, EvaluationClient, Score
from cs_eval.report import NOISE_FLOOR, aggregate_metric, build_rows
from cs_eval.spans import SCOPE_NAME, build_session_spans
from cs_eval.tools import TOOL_SPECS, check_policy, lookup_order

TURNS = [
    {
        "user": "订单 12345 坏了",
        "assistant": "已为您查询",
        "tool_calls": [
            {"name": "lookup_order", "args": {"order_id": "12345"}, "result": "{}"}
        ],
    },
    {"user": "运费呢", "assistant": "免运费", "tool_calls": []},
]


# --- span contract ----------------------------------------------------------
def _build():
    return build_session_spans(
        session_id="sess-1",
        turns=TURNS,
        system_prompt="be helpful",
        tool_specs=TOOL_SPECS,
    )


def test_one_trace_per_turn():
    _, trace_ids = _build()
    assert len(trace_ids) == len(TURNS)
    assert len(set(trace_ids)) == len(TURNS), "each turn needs its own trace"


def test_span_kinds_present():
    spans, _ = _build()
    kinds = [s["attributes"]["openinference.span.kind"] for s in spans]
    assert kinds.count("AGENT") == 2
    assert kinds.count("LLM") == 2
    assert kinds.count("TOOL") == 1


def test_scope_selects_openinference_reader():
    spans, _ = _build()
    assert all(s["scope"]["name"] == SCOPE_NAME for s in spans)
    assert SCOPE_NAME.startswith("openinference.instrumentation.")


def test_session_id_on_every_span():
    spans, _ = _build()
    assert all(s["attributes"]["session.id"] == "sess-1" for s in spans)


def test_agent_span_carries_prompt_and_reply():
    spans, _ = _build()
    agent = [s for s in spans if s["attributes"]["openinference.span.kind"] == "AGENT"]
    assert agent[0]["attributes"]["input.value"] == TURNS[0]["user"]
    assert agent[0]["attributes"]["output.value"] == TURNS[0]["assistant"]


def test_tool_span_names_and_payloads():
    spans, _ = _build()
    tool = [s for s in spans if s["attributes"]["openinference.span.kind"] == "TOOL"][0]
    assert tool["attributes"]["tool.name"] == "lookup_order"
    assert tool["attributes"]["gen_ai.tool.name"] == "lookup_order"
    assert "12345" in tool["attributes"]["input.value"]


def test_llm_span_advertises_available_tools():
    """ToolSelectionAccuracy needs the catalogue of what could have been called."""
    spans, _ = _build()
    llm = [s for s in spans if s["attributes"]["openinference.span.kind"] == "LLM"][0]
    schema_keys = [k for k in llm["attributes"] if k.startswith("llm.tools.")]
    assert len(schema_keys) == len(TOOL_SPECS)


def test_ids_are_valid_hex_lengths():
    spans, _ = _build()
    for span in spans:
        assert len(span["traceId"]) == 32
        assert len(span["spanId"]) == 16
        int(span["traceId"], 16)  # raises if not hex


def test_children_parent_to_agent_span():
    spans, _ = _build()
    by_trace: dict[str, list[dict]] = {}
    for span in spans:
        by_trace.setdefault(span["traceId"], []).append(span)
    for group in by_trace.values():
        agent = [s for s in group if s["attributes"]["openinference.span.kind"] == "AGENT"]
        assert len(agent) == 1
        for child in group:
            if child is not agent[0]:
                assert child["parentSpanId"] == agent[0]["spanId"]


# --- dataset ---------------------------------------------------------------
def test_shipped_dataset_loads():
    conversations = load_dataset("data/conversations.yaml")
    assert len(conversations) == 12
    assert {c.lang for c in conversations} == {"zh", "en"}
    assert all(c.ground_truth.assertions for c in conversations)


def test_every_conversation_within_result_cap():
    """Evaluate keeps only the last 10 results, so long sessions lose early turns."""
    for conversation in load_dataset("data/conversations.yaml"):
        assert conversation.turn_count <= 5


def test_rejects_missing_turns(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("conversations:\n  - id: x\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(bad)


def test_rejects_duplicate_ids(tmp_path):
    bad = tmp_path / "dup.yaml"
    bad.write_text(
        "conversations:\n"
        "  - id: x\n    user_turns: ['a']\n"
        "  - id: x\n    user_turns: ['b']\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError):
        load_dataset(bad)


# --- tools -----------------------------------------------------------------
def test_lookup_order_known_and_unknown():
    assert "delivered" in lookup_order.invoke({"order_id": "12345"})
    assert "order_not_found" in lookup_order.invoke({"order_id": "99999"})


def test_check_policy_rejects_unknown_topic():
    assert "unknown_topic" in check_policy.invoke({"topic": "nonsense"})


def test_outside_window_policy_demands_escalation():
    assert "escalated" in check_policy.invoke({"topic": "outside_window"}).lower()


# --- report ----------------------------------------------------------------
def _score(evaluator_id: str, value: float, conversation_id: str = "c1") -> Score:
    return Score(
        conversation_id=conversation_id,
        evaluator_id=evaluator_id,
        metric="m",
        level=cfg.TRACE,
        value=value,
        label="",
        explanation="",
    )


def test_all_builtins_are_higher_is_better():
    """Verified against the live API: even Harmfulness returns 1.0 for the good case."""
    for evaluator in cfg.select("full"):
        assert evaluator.higher_is_better


def test_aggregate_ignores_failed_scores():
    evaluator = cfg.select("core")[1]  # Correctness
    scores = [_score(evaluator.evaluator_id, 1.0), _score(evaluator.evaluator_id, 0.0)]
    scores.append(
        Score("c1", evaluator.evaluator_id, "m", cfg.TRACE, None, "", "", error="Boom")
    )
    assert aggregate_metric(scores, evaluator) == pytest.approx(0.5)


def test_small_change_is_reported_as_flat():
    evaluator = cfg.select("core")[1]
    before = [_score(evaluator.evaluator_id, 0.80)]
    after = [_score(evaluator.evaluator_id, 0.80 + NOISE_FLOOR / 2)]
    row = [r for r in build_rows(before, after, (evaluator,))][0]
    assert row.verdict == "flat"


def test_large_change_is_reported_as_better():
    evaluator = cfg.select("core")[1]
    before = [_score(evaluator.evaluator_id, 0.20)]
    after = [_score(evaluator.evaluator_id, 0.90)]
    row = build_rows(before, after, (evaluator,))[0]
    assert row.verdict == "better"


# --- cost and budget -------------------------------------------------------
def test_cost_scales_with_turns_not_just_conversations():
    evaluators = cfg.select("core")
    assert cfg.estimate_judge_calls(evaluators, turns=4, tool_calls=2) > (
        cfg.estimate_judge_calls(evaluators, turns=2, tool_calls=2)
    )


def test_minimal_profile_is_cheaper_than_core():
    assert cfg.estimate_judge_calls(cfg.select("minimal"), 3, 3) < (
        cfg.estimate_judge_calls(cfg.select("core"), 3, 3)
    )


def test_budget_guard_blocks_before_spending():
    guard = BudgetGuard(1.00)
    with pytest.raises(BudgetExceeded):
        guard.check_before(2.00)


def test_budget_guard_trips_mid_run():
    guard = BudgetGuard(0.01)
    guard.record(input_tokens=1_000_000, output_tokens=1_000_000)
    with pytest.raises(BudgetExceeded):
        guard.check_during()


def test_unknown_evaluator_profile_rejected():
    with pytest.raises(ValueError):
        cfg.select("everything")


def test_reference_inputs_carry_ground_truth():
    conversation = load_dataset("data/conversations.yaml")[1]  # cs-002
    refs = EvaluationClient._reference_inputs(conversation, "sess-9")
    assert refs[0]["context"]["spanContext"]["sessionId"] == "sess-9"
    assert refs[0]["assertions"]
    assert refs[0]["expectedTrajectory"]["toolNames"]

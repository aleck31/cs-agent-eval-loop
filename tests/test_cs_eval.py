"""Unit tests. No AWS calls, no spend -- safe to run in CI.

The span tests are the important ones: they pin the contract that the live
Evaluate API accepted, so a future refactor cannot quietly break it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cs_eval import config as cfg
from cs_eval.dataset import DatasetError, load_dataset, load_golden_set
from cs_eval.evaluate import BudgetExceeded, BudgetGuard, EvaluationClient, RunResult, Score
from cs_eval.report import NOISE_FLOOR, aggregate_metric, build_rows
from cs_eval.spans import SCOPE_NAME, build_session_spans
from cs_eval.tools import (
    TOOL_SPECS,
    check_policy,
    lookup_account,
    lookup_position,
    lookup_transaction,
)

DATASET = "data/golden_set.yaml"

TURNS = [
    {
        "user": "UID 88412337 的提币没到账",
        "assistant": "已为您查询该笔提币",
        "tool_calls": [
            {
                "name": "lookup_transaction",
                "args": {"transaction_id": "W-20260815-7741"},
                "result": "{}",
            }
        ],
    },
    {"user": "那我该怎么办", "assistant": "请联系接收方平台", "tool_calls": []},
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
    assert tool["attributes"]["tool.name"] == "lookup_transaction"
    assert tool["attributes"]["gen_ai.tool.name"] == "lookup_transaction"
    assert "W-20260815-7741" in tool["attributes"]["input.value"]


def test_llm_span_advertises_available_tools():
    """ToolSelectionAccuracy needs the catalogue of what could have been called."""
    spans, _ = _build()
    llm = [s for s in spans if s["attributes"]["openinference.span.kind"] == "LLM"][0]
    schema_keys = [k for k in llm["attributes"] if k.startswith("llm.tools.")]
    assert len(schema_keys) == len(TOOL_SPECS) == 4


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
    conversations = load_dataset(DATASET)
    assert len(conversations) == 12
    assert {c.lang for c in conversations} == {"zh", "en"}
    assert all(c.ground_truth.assertions for c in conversations)


def test_every_conversation_within_result_cap():
    """Evaluate keeps only the last 10 results, so long sessions lose early turns."""
    for conversation in load_dataset(DATASET):
        assert conversation.turn_count <= 5


def test_dataset_covers_the_expensive_failure_modes():
    """These categories are the reason this dataset exists; losing one is a regression."""
    categories = {c.category for c in load_dataset(DATASET)}
    for required in ("wrong_chain", "liquidation", "security", "compliance"):
        assert required in categories, f"missing {required} coverage"


def test_every_expected_tool_exists():
    """A trajectory naming a tool we do not ship would silently never match."""
    names = {spec["name"] for spec in TOOL_SPECS}
    for conversation in load_dataset(DATASET):
        for tool_name in conversation.ground_truth.expected_trajectory:
            assert tool_name in names, f"{conversation.id} expects unknown tool {tool_name}"


# --- golden set --------------------------------------------------------------
def _write_golden(path, *, assertion="Holds.", title="t", version=1):
    path.write_text(
        f"golden_set:\n  name: demo\n  version: {version}\n"
        f"conversations:\n  - id: x\n    title: {title}\n    user_turns: ['a']\n"
        f"    ground_truth:\n      assertions: ['{assertion}']\n",
        encoding="utf-8",
    )
    return load_golden_set(path)


def test_shipped_golden_set_is_named_and_versioned():
    golden = load_golden_set(DATASET)
    assert golden.name == "cex-support"
    assert golden.version != "unversioned"
    assert golden.fingerprint in golden.label


def test_fingerprint_tracks_ground_truth_not_display_fields(tmp_path):
    base = _write_golden(tmp_path / "a.yaml").fingerprint
    assert _write_golden(tmp_path / "b.yaml", title="renamed").fingerprint == base
    assert _write_golden(tmp_path / "c.yaml", assertion="Changed.").fingerprint != base


def test_committed_fixtures_were_scored_on_the_shipped_golden_set():
    """Editing the golden set without re-running live would make replay compare against a different yardstick."""
    fingerprint = load_golden_set(DATASET).fingerprint
    for variant in ("baseline", "improved"):
        payload = json.loads(Path(f"fixtures/{variant}.json").read_text(encoding="utf-8"))
        assert payload["golden_set_fingerprint"] == fingerprint, variant


def test_mismatched_fixture_is_flagged(tmp_path):
    from cs_eval.cli import golden_set_mismatches

    golden = _write_golden(tmp_path / "g.yaml")
    same = RunResult(variant="baseline", golden_set_fingerprint=golden.fingerprint)
    stale = RunResult(variant="improved", golden_set_fingerprint="000000000000")
    warnings = golden_set_mismatches(golden, same, stale)
    assert len(warnings) == 1 and "improved" in warnings[0]


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
def test_lookup_account_known_and_unknown():
    assert "active" in lookup_account.invoke({"uid": "88412337"})
    assert "account_not_found" in lookup_account.invoke({"uid": "00000000"})


def test_security_hold_is_discoverable():
    """cs-002 is only answerable if the hold shows up in the account record."""
    assert "security_hold" in lookup_account.invoke({"uid": "88410092"})
    assert "password_changed" in lookup_account.invoke({"uid": "88410092"})


def test_risk_review_detail_is_withheld_in_the_data():
    """The agent cannot leak review criteria that the tool never returns."""
    record = lookup_account.invoke({"uid": "88415566"})
    assert "withheld" in record


def test_wrong_chain_transaction_is_marked_unrecoverable():
    record = lookup_transaction.invoke({"transaction_id": "D-20260818-3310"})
    assert "chain_mismatch" in record
    assert '"recoverable": false' in record.lower()


def test_liquidation_record_exposes_mark_price():
    """The whole liquidation explanation hinges on mark price vs last traded price."""
    record = lookup_position.invoke({"uid": "88412337", "symbol": "BTCUSDT"})
    assert "mark_price_at_liquidation" in record
    assert "lowest_last_traded_price" in record


def test_copy_follow_failure_reason_is_available():
    record = lookup_position.invoke({"uid": "88419901", "symbol": "DOGEUSDT"})
    assert "insufficient_available_margin" in record


def test_check_policy_rejects_unknown_topic():
    assert "unknown_topic" in check_policy.invoke({"topic": "nonsense"})


def test_wrong_chain_policy_forbids_promising_recovery():
    text = check_policy.invoke({"topic": "wrong_chain"}).lower()
    assert "cannot recover" in text


def test_liquidation_policy_states_no_compensation():
    text = check_policy.invoke({"topic": "liquidation"}).lower()
    assert "does not compensate" in text


def test_credentials_policy_forbids_accepting_secrets():
    text = check_policy.invoke({"topic": "security_credentials"}).lower()
    assert "never" in text and "seed phrase" in text


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


def test_all_evaluators_are_higher_is_better():
    """Verified live: Harmfulness, Refusal and PIILeakage all return 1.0 for good."""
    for evaluator in cfg.select("full"):
        assert evaluator.higher_is_better


def test_core_set_covers_outcome_and_tools():
    ids = {e.evaluator_id for e in cfg.select("core")}
    assert "Builtin.GoalSuccessRate" in ids
    assert "Builtin.ToolSelectionAccuracy" in ids
    assert "Builtin.ToolParameterAccuracy" in ids


def test_measured_and_rejected_evaluators_stay_out_of_core():
    """Both were run against this dataset and removed on evidence; see EXCLUDED."""
    ids = {e.evaluator_id for e in cfg.select("core")}
    for rejected in ("Builtin.Refusal", "ThirdParty.DeepEval.PIILeakage"):
        assert rejected not in ids
        assert rejected in cfg.EXCLUDED


def test_red_lines_are_encoded_as_assertions():
    """With Refusal gone, assertions are the instrument enforcing the red lines."""
    conversations = {c.id: c for c in load_dataset(DATASET)}
    joined = " ".join(conversations["cs-012"].ground_truth.assertions).lower()
    assert "did not provide investment" in joined
    joined = " ".join(conversations["cs-003"].ground_truth.assertions).lower()
    assert "compensation" in joined


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
    row = build_rows(before, after, (evaluator,))[0]
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
    conversation = load_dataset(DATASET)[2]  # cs-003, wrong chain
    refs = EvaluationClient._reference_inputs(conversation, "sess-9")
    assert refs[0]["context"]["spanContext"]["sessionId"] == "sess-9"
    assert refs[0]["assertions"]
    assert refs[0]["expectedTrajectory"]["toolNames"]

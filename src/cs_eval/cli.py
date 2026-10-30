"""Command line entry point: the whole loop behind one command.

    python -m cs_eval.cli demo                 # run it
    python -m cs_eval.cli demo --mode replay    # render committed results, $0
    python -m cs_eval.cli propose               # derive prompts/improved.md honestly

Modes
  live      generate transcripts with Bedrock, score them with Evaluate  (costs money)
  replay    read fixtures/*.json, render the same report                 (free)
  deployed  DOCUMENTED ONLY, UNVERIFIED -- see README
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import textwrap
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from cs_eval import config as cfg
from cs_eval import settings
from cs_eval.agent import build_agent, model_id, run_conversation
from cs_eval.dataset import (
    DEV,
    HOLDOUT,
    GoldenSet,
    dataset_warnings,
    load_golden_set,
    summarise,
)
from cs_eval.evaluate import (
    BudgetExceeded,
    BudgetGuard,
    EvaluationClient,
    RunResult,
    Score,
    project_cost,
    vote,
)
from cs_eval.diagnose import by_layer, diagnose
from cs_eval.redlines import check_all
from cs_eval.stability import render as render_stability
from cs_eval.stability import spreads
from cs_eval.improve import (
    collect_findings,
    machine_proposed_header,
    propose_prompt,
    propose_with_agentcore,
)
from cs_eval.spans import build_session_spans
from cs_eval.report import (
    build_case_matrix,
    build_rows,
    judge_panel,
    render_markdown,
    render_terminal,
)
from cs_eval.tools import TOOL_SPECS

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "prompts"
FIXTURES = ROOT / "fixtures"
REPORTS = ROOT / "reports"


# --- persistence ------------------------------------------------------------
def _fixture_path(name: str) -> Path:
    return FIXTURES / f"{name}.json"


def save_run(result: RunResult) -> None:
    FIXTURES.mkdir(exist_ok=True)
    payload = {
        "variant": result.variant,
        "name": result.label,
        "model": result.model,
        "judge_models": result.judge_models,
        "golden_set_fingerprint": result.golden_set_fingerprint,
        "generated": datetime.now(timezone.utc).isoformat(),
        "judge_calls": result.judge_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "failures": result.failures,
        "scores": [asdict(s) for s in result.scores],
        "transcripts": result.transcripts,
    }
    _fixture_path(result.label).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_run(name: str) -> RunResult:
    path = _fixture_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"no fixture '{name}' at {path}. Run in --mode live first, "
            f"or check the committed fixtures."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    transcripts = payload.get("transcripts", [])
    first_turns = transcripts[0]["turns"] if transcripts and transcripts[0]["turns"] else [{}]
    return RunResult(
        variant=payload["variant"],
        name=payload.get("name", name),
        model=payload.get("model") or first_turns[0].get("model", ""),
        judge_models=payload.get("judge_models", {}),
        golden_set_fingerprint=payload.get("golden_set_fingerprint", ""),
        scores=[Score(**s) for s in payload["scores"]],
        judge_calls=payload.get("judge_calls", 0),
        input_tokens=payload.get("input_tokens", 0),
        output_tokens=payload.get("output_tokens", 0),
        failures=payload.get("failures", []),
        transcripts=transcripts,
    )


def _load_golden(path: str, limit: int | None) -> GoldenSet:
    """The golden set as it will actually be scored; --limit yields a distinct fingerprint."""
    golden = load_golden_set(settings.resolve_path(path))
    if limit:
        golden = replace(golden, conversations=golden.conversations[:limit])
    return golden


def golden_set_mismatches(golden: GoldenSet, *runs: RunResult) -> list[str]:
    """Fixtures scored on a different golden set cannot be compared like-for-like."""
    return [
        f"fixtures/{run.label}.json was scored on golden set "
        f"{run.golden_set_fingerprint or 'unknown'}, but {golden.name} is now "
        f"{golden.fingerprint}; re-run live before trusting this comparison"
        for run in runs
        if run.golden_set_fingerprint != golden.fingerprint
    ]


# --- the live run -----------------------------------------------------------
def run_variant(
    *,
    variant: str,
    golden: GoldenSet,
    evaluators: tuple[cfg.Evaluator, ...],
    region: str,
    guard: BudgetGuard,
    model: str | None = None,
    name: str | None = None,
    verbose: bool = True,
) -> RunResult:
    """Generate transcripts for one prompt variant, then score them."""
    prompt_path = PROMPTS / f"{variant}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"prompt not found: {prompt_path}")
    system_prompt = prompt_path.read_text(encoding="utf-8")

    model = model or model_id()
    agent = build_agent(system_prompt, region, model=model)
    client = EvaluationClient(region, guard)
    result = RunResult(
        variant=variant, golden_set_fingerprint=golden.fingerprint, name=name or variant,
        model=model, judge_models={
            e.evaluator_id: client.judge_model(e) or e.judge_model for e in evaluators if e.judge_model
        },
    )
    conversations = golden.conversations

    for index, conversation in enumerate(conversations, start=1):
        if verbose:
            print(
                f"  [{index:>2}/{len(conversations)}] {conversation.id} "
                f"{conversation.title[:34]:<34}",
                end="",
                flush=True,
            )
        transcript = run_conversation(
            agent, conversation.id, variant, conversation.user_turns
        )
        for turn in transcript.turns:
            turn.model = model
        result.transcripts.append(transcript.as_dict())

        scores, _, _, failures = client.score_conversation(
            conversation=conversation,
            transcript_turns=[t.as_dict() for t in transcript.turns],
            evaluators=evaluators,
            system_prompt=system_prompt,
            tool_specs=TOOL_SPECS,
        )
        result.scores.extend(scores)
        result.failures.extend(failures)
        result.judge_calls += len([s for s in scores if s.ok])

        if verbose:
            print(f" {len(scores):>3} results   (spend ${guard.spent_usd:.2f})")

    result.input_tokens = sum(s.input_tokens for s in result.scores)
    result.output_tokens = sum(s.output_tokens for s in result.scores)
    save_run(result)
    return result


# --- rendering --------------------------------------------------------------
def prompt_diff(baseline: str, improved: str) -> list[str]:
    """A readable summary of the prompt change, not a raw unified diff.

    Ordered added-first: what the change *introduced* is the interesting part for
    a reader asking "so what did you actually do?". Lines are wrapped so nothing
    is cut off mid-sentence in the terminal view.
    """
    before_text = (PROMPTS / f"{baseline}.md").read_text(encoding="utf-8")
    after_text = _strip_header((PROMPTS / f"{improved}.md").read_text(encoding="utf-8"))

    before_lines = [line.strip() for line in before_text.splitlines() if line.strip()]
    after_lines = [line.strip() for line in after_text.splitlines() if line.strip()]

    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
    removed: list[str] = []
    added: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed.extend(before_lines[i1:i2])
        if tag in ("replace", "insert"):
            added.extend(after_lines[j1:j2])

    def clean(items: list[str]) -> list[str]:
        out = []
        for item in items:
            text = item.lstrip("-*# ").replace("**", "").strip()
            if len(text) > 8:
                out.append(text)
        return out

    removed, added = clean(removed), clean(added)

    lines: list[str] = []
    if removed:
        lines.append("REMOVED from the old prompt:")
        for item in removed[:6]:
            lines.extend(_prefix_wrap(item, "  - "))
    if added:
        if lines:
            lines.append("")
        lines.append("ADDED in the new prompt:")
        for item in added[:9]:
            lines.extend(_prefix_wrap(item, "  + "))
    return lines or ["(no differences)"]


def _prefix_wrap(text: str, prefix: str, width: int = 70) -> list[str]:
    """Wrap one bullet, indenting continuations under the prefix."""
    words = text.split()
    lines: list[str] = []
    current = prefix
    pad = " " * len(prefix)
    for word in words:
        candidate = f"{current}{word} "
        if len(candidate) > width and current.strip() not in ("+", "-"):
            lines.append(current.rstrip())
            current = f"{pad}{word} "
        else:
            current = candidate
    if current.strip():
        lines.append(current.rstrip())
    return lines


def _strip_header(text: str) -> str:
    if text.lstrip().startswith("<!--"):
        _, _, rest = text.partition("-->")
        return rest.lstrip("\n")
    return text


def emit_report(
    *,
    before: RunResult,
    after: RunResult,
    golden: GoldenSet,
    evaluators: tuple[cfg.Evaluator, ...],
    duration_s: float,
    cost_usd: float,
    write_markdown: bool = True,
) -> Path | None:
    conversations = golden.conversations
    judges = tuple(e for e in evaluators if e.judge_model)
    evaluators = tuple(e for e in evaluators if e.scorecard)
    rows = build_rows(before.scores, after.scores, evaluators)
    holdout = golden.ids(HOLDOUT)
    holdout_rows = build_rows(
        [s for s in before.scores if s.conversation_id in holdout],
        [s for s in after.scores if s.conversation_id in holdout],
        evaluators,
    )
    cells = build_case_matrix(before.scores, after.scores, evaluators, conversations)
    change = []
    if before.variant != after.variant:
        change.append(f"prompts/{before.variant}.md -> prompts/{after.variant}.md")
    if before.model != after.model:
        change.append(f"model {before.model or '?'} -> {after.model or '?'}")
    diff = (prompt_diff(before.variant, after.variant) if before.variant != after.variant
            else ["(prompt unchanged)"])
    failures = before.failures + after.failures
    judge_calls = before.judge_calls + after.judge_calls

    before_breaches = check_all(before.transcripts)
    after_breaches = check_all(after.transcripts)
    shared = {
        "rows": rows,
        "holdout_rows": holdout_rows,
        "holdout_count": len(holdout),
        "cells": cells,
        "conversations": conversations,
        "golden_set_label": golden.label,
        "change": change or ["(nothing: same prompt and model)"],
        "judge_calls": judge_calls,
        "cost_usd": cost_usd,
        "duration_s": duration_s,
        "prompt_diff": diff,
        "failures": failures,
        "before_breaches": before_breaches,
        "after_breaches": after_breaches,
        "panel_before": judge_panel(before.scores, judges, conversations),
        "panel_after": judge_panel(after.scores, judges, conversations),
        "where_before": by_layer(diagnose(before.scores, evaluators, conversations, before_breaches)),
        "where_after": by_layer(diagnose(after.scores, evaluators, conversations, after_breaches)),
    }

    print()
    print(render_terminal(**shared))

    if not write_markdown:
        return None
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS / f"{stamp}.md"
    path.write_text(
        render_markdown(**shared, evaluators=evaluators), encoding="utf-8"
    )
    return path


# --- commands ---------------------------------------------------------------
def cmd_demo(args: argparse.Namespace) -> int:
    golden = _load_golden(args.data, args.limit)
    conversations = golden.conversations
    evaluators = cfg.select(args.evaluators)

    print(f"Golden set: {golden.label} {summarise(conversations)}")
    print(f"Evaluators: {args.evaluators} -> "
          f"{', '.join(e.short_name for e in evaluators)}")
    for warning in dataset_warnings(conversations):
        print(f"  ! {warning}")

    if args.mode == "deployed":
        print(
            "\n--mode=deployed is DOCUMENTED ONLY and UNVERIFIED in this repo.\n"
            "See README section 'Running against a deployed AgentCore agent'.",
            file=sys.stderr,
        )
        return 2

    if args.mode == "replay":
        before, after = load_run(args.baseline), load_run(args.improved)
        cost = before.cost_usd + after.cost_usd
        print(f"\nMode:       replay (no AWS calls, $0.00 spent now)")
        print(f"            replaying a run that originally cost ~${cost:.2f}")
        # Fixtures always cover the full set, so compare against it, not the --limit slice.
        for warning in golden_set_mismatches(_load_golden(args.data, None), before, after):
            print(f"\nWARNING: {warning}", file=sys.stderr)
        path = emit_report(
            before=before, after=after, golden=golden,
            evaluators=evaluators, duration_s=0.0, cost_usd=cost,
        )
        if path:
            print(f"Full report: {path.relative_to(ROOT)}")
        return 0

    # live
    calls, projected = project_cost(conversations, evaluators, sides=2)
    print(f"\nProjected:  ~{calls} judge calls, ~${projected:.2f} "
          f"(ceiling ${args.max_usd:.2f})")
    guard = BudgetGuard(args.max_usd)
    try:
        guard.check_before(projected)
    except BudgetExceeded as exc:
        print(f"\nSTOPPING: {exc}", file=sys.stderr)
        return 3

    if not args.yes:
        reply = input("Proceed and spend? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return 0

    started = time.monotonic()
    try:
        print(f"\nRunning BEFORE ({args.baseline}):")
        before = run_variant(
            variant=args.baseline, golden=golden,
            evaluators=evaluators, region=args.region, guard=guard,
        )
        print(f"\nRunning AFTER ({args.improved}):")
        after = run_variant(
            variant=args.improved, golden=golden,
            evaluators=evaluators, region=args.region, guard=guard,
        )
    except BudgetExceeded as exc:
        print(f"\nSTOPPED MID-RUN: {exc}", file=sys.stderr)
        return 3

    duration = time.monotonic() - started
    path = emit_report(
        before=before, after=after, golden=golden,
        evaluators=evaluators, duration_s=duration, cost_usd=guard.spent_usd,
    )
    if path:
        print(f"Full report: {path.relative_to(ROOT)}")
    print(f"Actual spend this run: ${guard.spent_usd:.2f}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run and score ONE variant, saving a fixture. Lets you stage the loop."""
    golden = _load_golden(args.data, args.limit)
    conversations = golden.conversations
    evaluators = cfg.select(args.evaluators)

    calls, projected = project_cost(conversations, evaluators, sides=1)
    print(f"Variant:    {args.variant} on {args.model or model_id()} -> fixtures/{args.name or args.variant}.json")
    print(f"Golden set: {golden.label} {summarise(conversations)}")
    print(f"Projected: ~{calls} judge calls, ~${projected:.2f} "
          f"(ceiling ${args.max_usd:.2f})")

    guard = BudgetGuard(args.max_usd)
    try:
        guard.check_before(projected)
    except BudgetExceeded as exc:
        print(f"\nSTOPPING: {exc}", file=sys.stderr)
        return 3

    if not args.yes:
        if input("Proceed and spend? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return 0

    print()
    try:
        result = run_variant(
            variant=args.variant, golden=golden, evaluators=evaluators,
            region=args.region, guard=guard, model=args.model or None, name=args.name or None,
        )
    except BudgetExceeded as exc:
        print(f"\nSTOPPED MID-RUN: {exc}", file=sys.stderr)
        return 3

    ok = len([s for s in result.scores if s.ok])
    print(f"\nSaved fixtures/{result.label}.json  "
          f"({ok} scored results, {len(result.failures)} failures)")
    print(f"Spend this run: ${guard.spent_usd:.2f}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Compare any two saved runs, e.g. a model change with the prompt held fixed. Free."""
    golden = _load_golden(args.data, None)
    before, after = load_run(args.before), load_run(args.after)
    for warning in golden_set_mismatches(golden, before, after):
        print(f"WARNING: {warning}", file=sys.stderr)
    path = emit_report(
        before=before, after=after, golden=golden, evaluators=cfg.select(args.evaluators),
        duration_s=0.0, cost_usd=before.cost_usd + after.cost_usd,
    )
    if path:
        print(f"Full report: {path.relative_to(ROOT)}")
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    """Derive prompts/improved.md from the baseline run, learning from dev cases only."""
    evaluators = cfg.select(args.evaluators)
    baseline = load_run(args.baseline)
    dev_ids = _load_golden(args.data, None).ids(DEV)
    if args.source == "agentcore":
        return _propose_with_agentcore(args, baseline, dev_ids)
    findings = collect_findings(
        [s for s in baseline.scores if s.conversation_id in dev_ids], evaluators
    )
    if not findings:
        print("Baseline scored clean; nothing to improve.", file=sys.stderr)
        return 1

    print(f"Learning from {len(findings)} low-scoring dev results "
          f"(judge explanations only, no ground truth, no holdout cases).")
    current = (PROMPTS / f"{args.baseline}.md").read_text(encoding="utf-8")
    text, in_tokens, out_tokens = propose_prompt(
        current_prompt=current, findings=findings, region=args.region
    )
    cost = (in_tokens / 1000 * cfg.AGENT_PRICE_INPUT_PER_1K
            + out_tokens / 1000 * cfg.AGENT_PRICE_OUTPUT_PER_1K)

    target = PROMPTS / f"{args.improved}.md"
    target.write_text(
        machine_proposed_header(len(findings)) + "\n" + text + "\n", encoding="utf-8"
    )
    print(f"Wrote {target.relative_to(ROOT)}  (~${cost:.3f})")
    return 0


def _propose_with_agentcore(args: argparse.Namespace, baseline: RunResult, dev_ids: set[str]) -> int:
    current = (PROMPTS / f"{args.baseline}.md").read_text(encoding="utf-8")
    dev = [t for t in baseline.transcripts if t["conversation_id"] in dev_ids]
    spans: list[dict] = []
    for transcript in dev:
        session_spans, _ = build_session_spans(
            session_id=f"{transcript['conversation_id']}-propose",
            turns=transcript["turns"], system_prompt=current, tool_specs=TOOL_SPECS,
        )
        spans.extend(session_spans)
    # The only steering evaluator start-recommendation accepts is a built-in one.
    goal = next(e for e in cfg.BUILTIN_CORE if e.evaluator_id == "Builtin.GoalSuccessRate")
    print(f"Recommending from {len(dev)} dev transcripts via AgentCore, steered by "
          f"{goal.short_name}. Takes a few minutes.")
    print("WARNING: it sees no ground truth, so it optimises what the customer asked for. "
          "Re-run both variants and read the red-line rows before trusting the result.")
    text, explanation = propose_with_agentcore(
        current_prompt=current, session_spans=spans,
        evaluator_arn=EvaluationClient(args.region).evaluator_arn(goal), region=args.region,
    )
    target = PROMPTS / f"{args.improved}.md"
    target.write_text(
        machine_proposed_header(len(dev), source="agentcore") + "\n" + text + "\n", encoding="utf-8"
    )
    print(f"Wrote {target.relative_to(ROOT)}")
    if explanation:
        print("\nWhy, in the optimiser's words:")
        for chunk in textwrap.wrap(" ".join(explanation.split()), 76)[:12]:
            print(f"  {chunk}")
    return 0


def cmd_rejudge(args: argparse.Namespace) -> int:
    """Fill every configured judge's vote on saved transcripts; drop judges no longer configured."""
    golden = _load_golden(args.data, None)
    by_id = {c.id: c for c in golden.conversations}
    evaluators = cfg.select(args.evaluators)
    judges = [e for e in evaluators if e.judge_model]
    voted = [e for e in evaluators if e.voted]
    names = args.fixtures.split(",") if args.fixtures else sorted(p.stem for p in FIXTURES.glob("*.json"))
    runs = {name: load_run(name) for name in names}

    def missing(run: RunResult, judge: cfg.Evaluator) -> list[str]:
        done = {s.conversation_id for s in run.scores if s.evaluator_id == judge.evaluator_id and s.ok}
        return [t["conversation_id"] for t in run.transcripts if t["conversation_id"] not in done]

    todo = {(name, j.evaluator_id): missing(run, j) for name, run in runs.items() for j in judges}
    projected = sum(
        len(cases) * cfg.projected_cost_per_call(j)
        for (_, eid), cases in todo.items() for j in judges if j.evaluator_id == eid
    )
    count = sum(len(cases) for cases in todo.values())
    print(f"Rejudge:    {len(runs)} fixture(s), {count} missing judge votes, ~${projected:.2f} "
          f"(ceiling ${args.max_usd:.2f})")
    guard = BudgetGuard(args.max_usd)
    try:
        guard.check_before(projected)
    except BudgetExceeded as exc:
        print(f"\nSTOPPING: {exc}", file=sys.stderr)
        return 3
    if count and not args.yes and input("Proceed and spend? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return 0

    client = EvaluationClient(args.region, guard)
    configured = {j.evaluator_id for j in judges} | {v.evaluator_id for v in voted}
    incomplete = 0
    for name, run in runs.items():
        prompt = (PROMPTS / f"{run.variant}.md").read_text(encoding="utf-8")
        transcripts = {t["conversation_id"]: t for t in run.transcripts}
        stale = [s.evaluator_id for s in run.scores if s.evaluator_id.startswith(cfg.CUSTOM_PREFIX)
                 and s.evaluator_id not in configured]
        scores = [s for s in run.scores if not s.evaluator_id.startswith(cfg.CUSTOM_PREFIX)
                  or (s.evaluator_id in {j.evaluator_id for j in judges} and s.ok)]
        failures = [f for f in run.failures if "cs_redline" not in f]
        for judge in judges:
            for cid in todo[(name, judge.evaluator_id)]:
                new, _, _, fails = client.score_conversation(
                    conversation=by_id[cid], transcript_turns=transcripts[cid]["turns"],
                    evaluators=(judge,), system_prompt=prompt, tool_specs=TOOL_SPECS,
                )
                scores.extend(new)
                failures.extend(fails)
        for vote_evaluator in voted:
            for cid in transcripts:
                scores.append(vote(cid, vote_evaluator, judges, scores))
        incomplete += sum(1 for s in scores if s.evaluator_id in {v.evaluator_id for v in voted} and not s.ok)
        run.scores = scores
        run.failures = failures
        run.judge_models = {j.evaluator_id: client.judge_model(j) or j.judge_model for j in judges}
        run.judge_calls = sum(1 for s in scores if s.ok and not any(s.evaluator_id == v.evaluator_id for v in voted))
        run.input_tokens = sum(s.input_tokens for s in scores)
        run.output_tokens = sum(s.output_tokens for s in scores)
        save_run(run)
        dropped = f", dropped {len(set(stale))} unconfigured judge(s)" if stale else ""
        print(f"  {name}: votes complete for {len(transcripts)} cases{dropped}")
    print(f"\nSpend this run: ${guard.spent_usd:.2f}")
    if incomplete:
        print(f"{incomplete} case(s) still miss a vote; re-run `cs-eval rejudge`.", file=sys.stderr)
        return 1
    return 0


def cmd_stability(args: argparse.Namespace) -> int:
    """Re-score saved transcripts N times to measure judge repeatability."""
    golden = _load_golden(args.data, None)
    by_id = {c.id: c for c in golden.conversations}
    run = load_run(args.variant)
    transcripts = {t["conversation_id"]: t for t in run.transcripts}
    case_ids = args.cases.split(",") if args.cases else sorted(transcripts)[:3]
    unknown = [cid for cid in case_ids if cid not in transcripts or cid not in by_id]
    if unknown:
        print(f"No transcript or golden case for: {', '.join(unknown)}", file=sys.stderr)
        return 2
    evaluators = cfg.select(args.evaluators)
    cases = [by_id[cid] for cid in case_ids]

    calls, per_repeat = project_cost(cases, evaluators, sides=1)
    projected = per_repeat * args.repeats
    print(f"Stability:  {args.variant} x {args.repeats} repeats on {', '.join(case_ids)}")
    print(f"Projected:  ~{calls * args.repeats} judge calls, ~${projected:.2f} "
          f"(ceiling ${args.max_usd:.2f})")
    guard = BudgetGuard(args.max_usd)
    try:
        guard.check_before(projected)
    except BudgetExceeded as exc:
        print(f"\nSTOPPING: {exc}", file=sys.stderr)
        return 3
    if not args.yes and input("Proceed and spend? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return 0

    system_prompt = (PROMPTS / f"{run.variant}.md").read_text(encoding="utf-8")
    client = EvaluationClient(args.region, guard)
    repeats: list[list[Score]] = []
    try:
        for index in range(args.repeats):
            scores: list[Score] = []
            for case in cases:
                result, _, _, _ = client.score_conversation(
                    conversation=case,
                    transcript_turns=transcripts[case.id]["turns"],
                    evaluators=evaluators,
                    system_prompt=system_prompt,
                    tool_specs=TOOL_SPECS,
                )
                scores.extend(result)
            repeats.append(scores)
            print(f"  repeat {index + 1}/{args.repeats} done (spend ${guard.spent_usd:.2f})")
    except BudgetExceeded as exc:
        print(f"\nSTOPPED MID-RUN: {exc}", file=sys.stderr)
        return 3

    print()
    print(render_stability(spreads(repeats, evaluators, case_ids), args.repeats))
    print(f"\nSpend this run: ${guard.spent_usd:.2f}")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    """Create or update the repo-defined evaluators in this account and region."""
    client = EvaluationClient(args.region)
    judges = [e for e in cfg.select(args.evaluators) if e.judge_model]
    if not judges:
        print("No red-line judges in this profile; nothing to set up.")
        return 0
    for judge in judges:
        evaluator_id, action = client.ensure_custom(judge, cfg.redline_judge_config(judge.judge_model))
        print(f"{action}: {evaluator_id} ({judge.level}, judge model {judge.judge_model})")
    print(f"{len(judges)} judge(s); the scorecard row is their majority vote, a tie counts as a breach.")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Check config, dataset and evaluator availability without spending on judges."""
    env_file = settings.ENV_FILE
    if env_file.exists():
        print(f"Config:     loaded {env_file.name}")
    else:
        print(f"Config:     no .env found (using built-in defaults). "
              f"Try: cp .env.example .env")
    for name in ("AWS_PROFILE", "AWS_REGION", "CS_EVAL_MODEL_ID", "CS_EVAL_JUDGE_MODEL_IDS",
                 "CS_EVAL_EVALUATORS", "CS_EVAL_MAX_USD", "CS_EVAL_DATA"):
        print(f"  {name:<23} {settings.get(name):<45} [{settings.describe_source(name)}]")

    identity = _whoami(args.region)
    print(f"\nIdentity:   {identity}")

    golden = _load_golden(args.data, None)
    conversations = golden.conversations
    print(f"Golden set OK: {golden.label} {summarise(conversations)}")
    for warning in dataset_warnings(conversations):
        print(f"  ! {warning}")

    evaluators = cfg.select(args.evaluators)
    client = EvaluationClient(args.region)
    available = client.available_evaluator_ids()
    if available is None:
        print("\nListEvaluators: permission denied -- cannot pre-validate evaluator IDs.")
        print("This is not fatal; Evaluate is authorised separately.")
    else:
        print(f"\nListEvaluators: {len(available)} evaluators visible in this account.")
        missing = [
            e.evaluator_id for e in evaluators if not e.voted
            and (client.custom_evaluator_id(e) is None if e.is_custom else e.evaluator_id not in available)
        ]
        if missing:
            print(f"  ! NOT AVAILABLE: {', '.join(missing)}")
            if any(e.is_custom and e.evaluator_id in missing for e in evaluators):
                print("    Repo-defined evaluators are created with `cs-eval setup`.")
            return 1
        print(f"  all {len(evaluators)} configured evaluators are available")
        for e in (e for e in evaluators if e.judge_model):
            actual = client.judge_model(e)
            if actual != e.judge_model:
                print(f"  ! {e.short_name} judges with {actual}, but CS_EVAL_JUDGE_MODEL_IDS says "
                      f"{e.judge_model}; run `make evaluators`")

    calls, projected = project_cost(conversations, evaluators, sides=2)
    print(f"\nA full before/after run would be ~{calls} judge calls, ~${projected:.2f} "
          f"(ceiling ${settings.get_float('CS_EVAL_MAX_USD'):.2f}).")
    return 0


def _whoami(region: str) -> str:
    """Report the resolved caller, so a wrong profile is obvious before spending."""
    try:
        import boto3

        identity = boto3.client("sts", region_name=region).get_caller_identity()
        return f"account {identity['Account']} as {identity['Arn'].split('/')[-1]}"
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return f"could not resolve credentials ({type(exc).__name__})"


def build_parser() -> argparse.ArgumentParser:
    """Flags default to the resolved setting, so an explicit flag always wins."""
    parser = argparse.ArgumentParser(
        prog="cs_eval",
        description="Measure whether a customer-service agent got better or worse.",
    )
    parser.add_argument("--region", default=settings.get("AWS_REGION"))
    parser.add_argument("--data", default=settings.get("CS_EVAL_DATA"),
                        help="golden set YAML: cases plus ground truth")
    parser.add_argument("--evaluators", default=settings.get("CS_EVAL_EVALUATORS"),
                        choices=["core", "full", "minimal"])
    parser.add_argument("--baseline", default=settings.get("CS_EVAL_BASELINE"))
    parser.add_argument("--improved", default=settings.get("CS_EVAL_IMPROVED"))

    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the whole loop and print the report")
    demo.add_argument("--mode", default=settings.get("CS_EVAL_MODE"),
                      choices=["live", "replay", "deployed"])
    demo.add_argument("--limit", type=int, default=settings.get_int("CS_EVAL_LIMIT"),
                      help="use only the first N cases (0 = all)")
    demo.add_argument("--max-usd", type=float,
                      default=settings.get_float("CS_EVAL_MAX_USD"),
                      help="hard spend ceiling")
    demo.add_argument("-y", "--yes", action="store_true", help="skip the spend prompt")
    demo.set_defaults(func=cmd_demo)

    run = subparsers.add_parser(
        "run", help="run and score ONE variant, saving a fixture"
    )
    run.add_argument("--variant", required=True, help="prompt name, e.g. baseline")
    run.add_argument("--model", default="", help="agent model ID (default: CS_EVAL_MODEL_ID)")
    run.add_argument("--name", default="", help="fixture name (default: the variant)")
    run.add_argument("--limit", type=int, default=settings.get_int("CS_EVAL_LIMIT"))
    run.add_argument("--max-usd", type=float,
                     default=settings.get_float("CS_EVAL_MAX_USD"))
    run.add_argument("-y", "--yes", action="store_true")
    run.set_defaults(func=cmd_run)

    report = subparsers.add_parser(
        "report", help="compare two saved runs without any AWS call"
    )
    report.add_argument("--before", required=True, help="fixture name, e.g. baseline-sonnet-4-5")
    report.add_argument("--after", required=True, help="fixture name, e.g. baseline")
    report.set_defaults(func=cmd_report)

    propose = subparsers.add_parser(
        "propose", help="derive prompts/improved.md from baseline findings"
    )
    propose.add_argument("--source", choices=["bedrock", "agentcore"], default="bedrock",
                         help="bedrock: one model call over judge explanations; "
                              "agentcore: start-recommendation steered by GoalSuccessRate")
    propose.set_defaults(func=cmd_propose)

    rejudge = subparsers.add_parser(
        "rejudge", help="fill missing red-line judge votes on saved runs and recompute the vote"
    )
    rejudge.add_argument("--fixtures", default="", help="comma-separated fixture names (default: all)")
    rejudge.add_argument("--max-usd", type=float, default=settings.get_float("CS_EVAL_MAX_USD"))
    rejudge.add_argument("-y", "--yes", action="store_true")
    rejudge.set_defaults(func=cmd_rejudge)

    stability = subparsers.add_parser(
        "stability", help="re-score saved transcripts N times to measure judge noise"
    )
    stability.add_argument("--variant", default=settings.get("CS_EVAL_BASELINE"),
                           help="fixture name whose transcripts are re-scored")
    stability.add_argument("--cases", default="",
                           help="comma-separated case IDs (default: first 3)")
    stability.add_argument("--repeats", type=int, default=3)
    stability.add_argument("--max-usd", type=float,
                           default=settings.get_float("CS_EVAL_MAX_USD"))
    stability.add_argument("-y", "--yes", action="store_true")
    stability.set_defaults(func=cmd_stability)

    setup = subparsers.add_parser(
        "setup", help="create or update the repo-defined evaluators in this account"
    )
    setup.set_defaults(func=cmd_setup)

    validate = subparsers.add_parser(
        "validate", help="check dataset and evaluators without spending on judges"
    )
    validate.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    # MUST come before argparse (which reads resolved settings for its defaults)
    # and before any boto3 client is built (which reads AWS_* from os.environ).
    settings.load_env()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

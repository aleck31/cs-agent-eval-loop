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
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cs_eval import config as cfg
from cs_eval import settings
from cs_eval.agent import build_agent, run_conversation
from cs_eval.dataset import GoldenSet, dataset_warnings, load_golden_set, summarise
from cs_eval.evaluate import (
    BudgetExceeded,
    BudgetGuard,
    EvaluationClient,
    RunResult,
    Score,
    project_cost,
)
from cs_eval.redlines import check_all
from cs_eval.improve import collect_findings, machine_proposed_header, propose_prompt
from cs_eval.report import (
    build_case_matrix,
    build_rows,
    render_markdown,
    render_terminal,
)
from cs_eval.tools import TOOL_SPECS

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "prompts"
FIXTURES = ROOT / "fixtures"
REPORTS = ROOT / "reports"


# --- persistence ------------------------------------------------------------
def _fixture_path(variant: str) -> Path:
    return FIXTURES / f"{variant}.json"


def save_run(result: RunResult) -> None:
    FIXTURES.mkdir(exist_ok=True)
    payload = {
        "variant": result.variant,
        "golden_set_fingerprint": result.golden_set_fingerprint,
        "generated": datetime.now(timezone.utc).isoformat(),
        "judge_calls": result.judge_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "failures": result.failures,
        "scores": [asdict(s) for s in result.scores],
        "transcripts": result.transcripts,
    }
    _fixture_path(result.variant).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_run(variant: str) -> RunResult:
    path = _fixture_path(variant)
    if not path.exists():
        raise FileNotFoundError(
            f"no fixture for '{variant}' at {path}. Run in --mode live first, "
            f"or check the committed fixtures."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunResult(
        variant=payload["variant"],
        golden_set_fingerprint=payload.get("golden_set_fingerprint", ""),
        scores=[Score(**s) for s in payload["scores"]],
        judge_calls=payload.get("judge_calls", 0),
        input_tokens=payload.get("input_tokens", 0),
        output_tokens=payload.get("output_tokens", 0),
        failures=payload.get("failures", []),
        transcripts=payload.get("transcripts", []),
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
        f"fixtures/{run.variant}.json was scored on golden set "
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
    verbose: bool = True,
) -> RunResult:
    """Generate transcripts for one prompt variant, then score them."""
    prompt_path = PROMPTS / f"{variant}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"prompt not found: {prompt_path}")
    system_prompt = prompt_path.read_text(encoding="utf-8")

    agent = build_agent(system_prompt, region)
    client = EvaluationClient(region, guard)
    result = RunResult(variant=variant, golden_set_fingerprint=golden.fingerprint)
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
    rows = build_rows(before.scores, after.scores, evaluators)
    cells = build_case_matrix(before.scores, after.scores, evaluators, conversations)
    diff = prompt_diff(before.variant, after.variant)
    failures = before.failures + after.failures
    judge_calls = before.judge_calls + after.judge_calls

    shared = {
        "rows": rows,
        "cells": cells,
        "conversations": conversations,
        "golden_set_label": golden.label,
        "baseline_name": before.variant,
        "improved_name": after.variant,
        "judge_calls": judge_calls,
        "cost_usd": cost_usd,
        "duration_s": duration_s,
        "prompt_diff": diff,
        "failures": failures,
        "before_breaches": check_all(before.transcripts),
        "after_breaches": check_all(after.transcripts),
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
    print(f"Variant:    {args.variant}")
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
            variant=args.variant, golden=golden,
            evaluators=evaluators, region=args.region, guard=guard,
        )
    except BudgetExceeded as exc:
        print(f"\nSTOPPED MID-RUN: {exc}", file=sys.stderr)
        return 3

    ok = len([s for s in result.scores if s.ok])
    print(f"\nSaved fixtures/{args.variant}.json  "
          f"({ok} scored results, {len(result.failures)} failures)")
    print(f"Spend this run: ${guard.spent_usd:.2f}")
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    """Derive prompts/improved.md from the baseline run's judge explanations."""
    evaluators = cfg.select(args.evaluators)
    baseline = load_run(args.baseline)
    findings = collect_findings(baseline.scores, evaluators)
    if not findings:
        print("Baseline scored clean; nothing to improve.", file=sys.stderr)
        return 1

    print(f"Learning from {len(findings)} low-scoring results "
          f"(judge explanations only, no ground truth).")
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


def cmd_validate(args: argparse.Namespace) -> int:
    """Check config, dataset and evaluator availability without spending on judges."""
    env_file = settings.ENV_FILE
    if env_file.exists():
        print(f"Config:     loaded {env_file.name}")
    else:
        print(f"Config:     no .env found (using built-in defaults). "
              f"Try: cp .env.example .env")
    for name in ("AWS_PROFILE", "AWS_REGION", "CS_EVAL_MODEL_ID",
                 "CS_EVAL_EVALUATORS", "CS_EVAL_MAX_USD", "CS_EVAL_DATA"):
        print(f"  {name:<20} {settings.get(name):<48} [{settings.describe_source(name)}]")

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
        missing = [e.evaluator_id for e in evaluators if e.evaluator_id not in available]
        if missing:
            print(f"  ! NOT AVAILABLE: {', '.join(missing)}")
            return 1
        print(f"  all {len(evaluators)} configured evaluators are available")

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
    run.add_argument("--limit", type=int, default=settings.get_int("CS_EVAL_LIMIT"))
    run.add_argument("--max-usd", type=float,
                     default=settings.get_float("CS_EVAL_MAX_USD"))
    run.add_argument("-y", "--yes", action="store_true")
    run.set_defaults(func=cmd_run)

    propose = subparsers.add_parser(
        "propose", help="derive prompts/improved.md from baseline findings"
    )
    propose.set_defaults(func=cmd_propose)

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

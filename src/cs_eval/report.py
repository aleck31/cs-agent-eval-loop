"""Turns raw evaluation results into something a non-coder can act on.

Two renderers, one aggregation. The terminal view is what gets screenshotted; the
Markdown view is the same content plus the full per-conversation matrix.

Presentation decisions that matter:

* Every metric is normalised so UP IS ALWAYS BETTER. Harmfulness is inverted
  (and footnoted), because a scorecard where some arrows mean the opposite of
  others is a scorecard that gets misread in a meeting.
* Deltas below the noise floor are shown as "~" rather than as wins. Judges are
  themselves LLMs and are not deterministic: during development, the identical
  transcript scored 0.83 and 1.00 on Helpfulness across runs. Reporting a 2-point
  move as an improvement would be dishonest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean

from cs_eval.config import Evaluator
from cs_eval.dataset import Conversation
from cs_eval.evaluate import Score
from cs_eval.redlines import RED_LINES, Breach, RedLine, cases_breaching

# Below this absolute change, we call it noise rather than movement. Derived from
# observed judge variance on identical input (~0.17 on a 0-1 scale).
NOISE_FLOOR = 0.10

# A conversation/metric at or below this is "weak" and worth a human looking.
WEAK_THRESHOLD = 0.60


@dataclass
class MetricRow:
    metric: str
    evaluator: Evaluator
    before: float | None
    after: float | None
    blurb: str

    @property
    def delta(self) -> float | None:
        if self.before is None or self.after is None:
            return None
        return self.after - self.before

    @property
    def verdict(self) -> str:
        delta = self.delta
        if delta is None:
            return "n/a"
        if abs(delta) < NOISE_FLOOR:
            return "flat"
        return "better" if delta > 0 else "worse"


def _normalise(value: float | None, evaluator: Evaluator) -> float | None:
    """Flip metrics where a low raw score is the good outcome."""
    if value is None:
        return None
    return value if evaluator.higher_is_better else 1.0 - value


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def aggregate_metric(
    scores: list[Score], evaluator: Evaluator, conversation_id: str | None = None
) -> float | None:
    """Mean normalised score for one evaluator, optionally within one conversation."""
    values = [
        normalised
        for score in scores
        if score.evaluator_id == evaluator.evaluator_id
        and score.ok
        and (conversation_id is None or score.conversation_id == conversation_id)
        and (normalised := _normalise(score.value, evaluator)) is not None
    ]
    return _mean_or_none(values)


def build_rows(
    before: list[Score], after: list[Score], evaluators: tuple[Evaluator, ...]
) -> list[MetricRow]:
    return [
        MetricRow(
            metric=evaluator.metric,
            evaluator=evaluator,
            before=aggregate_metric(before, evaluator),
            after=aggregate_metric(after, evaluator),
            blurb=evaluator.blurb,
        )
        for evaluator in evaluators
    ]


@dataclass
class CaseCell:
    conversation_id: str
    metric: str
    before: float | None
    after: float | None
    explanation: str = ""

    @property
    def delta(self) -> float | None:
        if self.before is None or self.after is None:
            return None
        return self.after - self.before


def build_case_matrix(
    before: list[Score],
    after: list[Score],
    evaluators: tuple[Evaluator, ...],
    conversations: list[Conversation],
) -> list[CaseCell]:
    """Per-conversation, per-metric scores, with the 'after' judge explanation."""
    cells: list[CaseCell] = []
    for conversation in conversations:
        for evaluator in evaluators:
            after_value = aggregate_metric(after, evaluator, conversation.id)
            explanation = ""
            if after_value is not None and after_value <= WEAK_THRESHOLD:
                matching = [
                    score
                    for score in after
                    if score.evaluator_id == evaluator.evaluator_id
                    and score.conversation_id == conversation.id
                    and score.ok
                    and score.explanation
                ]
                if matching:
                    worst = min(
                        matching, key=lambda s: _normalise(s.value, evaluator) or 0.0
                    )
                    explanation = worst.explanation
            cells.append(
                CaseCell(
                    conversation_id=conversation.id,
                    metric=evaluator.metric,
                    before=aggregate_metric(before, evaluator, conversation.id),
                    after=after_value,
                    explanation=explanation,
                )
            )
    return cells


def red_line_rows(
    before: list[Breach], after: list[Breach]
) -> list[tuple[RedLine, set[str], set[str]]]:
    """Per red line: which cases breached before and after."""
    return [(rl, cases_breaching(before, rl), cases_breaching(after, rl)) for rl in RED_LINES]


def _pct(value: float | None) -> str:
    return "  --" if value is None else f"{value * 100:4.0f}%"


def _delta_cell(row: MetricRow) -> str:
    delta = row.delta
    if delta is None:
        return "   n/a"
    points = delta * 100
    if abs(delta) < NOISE_FLOOR:
        return f"  ~{points:+.0f}"
    arrow = "^" if delta > 0 else "v"
    return f"{arrow} {points:+.0f}"


def _verdict_mark(row: MetricRow) -> str:
    return {"better": "[+]", "worse": "[-]", "flat": "[=]", "n/a": "[?]"}[row.verdict]


def render_terminal(
    *,
    rows: list[MetricRow],
    cells: list[CaseCell],
    conversations: list[Conversation],
    golden_set_label: str,
    baseline_name: str,
    improved_name: str,
    judge_calls: int,
    cost_usd: float,
    duration_s: float,
    prompt_diff: list[str],
    failures: list[str],
    before_breaches: list[Breach],
    after_breaches: list[Breach],
    width: int = 78,
) -> str:
    """The screenshot. Business metrics first, weak cases second, cause last."""
    out: list[str] = []
    line = "=" * width

    turns = sum(c.turn_count for c in conversations)
    languages = sorted({c.lang for c in conversations})

    out.append(line)
    out.append("CUSTOMER SERVICE AGENT - QUALITY REPORT".center(width))
    out.append(line)
    out.append(f"Golden set: {golden_set_label}")
    out.append(f"            {len(conversations)} conversations, {turns} turns, "
               f"{'/'.join(languages)}; same set scored before and after")
    out.append(f"Change:     prompts/{baseline_name}.md -> prompts/{improved_name}.md")
    out.append(f"Cost:       {judge_calls} judge calls | ~${cost_usd:.2f} | {duration_s:.0f}s")
    out.append("")

    # --- headline scorecard -------------------------------------------------
    out.append(f"{'':34}{'BEFORE':>8}{'AFTER':>8}{'CHANGE':>9}")
    for row in rows:
        out.append(
            f"{row.metric:<34}{_pct(row.before):>8}{_pct(row.after):>8}"
            f"{_delta_cell(row):>9}  {_verdict_mark(row)}"
        )
    out.append("")
    out.append("  [+] improved   [-] regressed   [=] unchanged (within noise)")
    out.append(f"  Higher is always better. Changes under {NOISE_FLOOR * 100:.0f} points are")
    out.append("  shown as ~ because LLM judges are not deterministic.")
    out.append("")

    # --- red lines ----------------------------------------------------------
    out.append(f"{'RED LINES (regex, no judge)':<34}{'BEFORE':>8}{'AFTER':>8}")
    for red_line, before_ids, after_ids in red_line_rows(before_breaches, after_breaches):
        mark = "[+]" if len(after_ids) < len(before_ids) else "[-]" if len(after_ids) > len(before_ids) else "[=]"
        out.append(f"{red_line.label:<34}{len(before_ids):>8}{len(after_ids):>8}{'':>9}  {mark}")
    out.append("  Cases breaching; lower is better. Deterministic, so any change is real.")
    out.append("")

    # --- weak cases --------------------------------------------------------
    weak = [
        cell
        for cell in cells
        if cell.after is not None and cell.after <= WEAK_THRESHOLD
    ]
    weak.sort(key=lambda c: (c.after or 0.0))
    titles = {c.id: c.title for c in conversations}

    out.append("-" * width)
    out.append("WEAKEST CASES AFTER THE CHANGE")
    out.append("-" * width)
    if not weak:
        out.append("  None below the weak threshold. Every case scored above "
                   f"{WEAK_THRESHOLD * 100:.0f}%.")
    else:
        for cell in weak[:6]:
            title = titles.get(cell.conversation_id, cell.conversation_id)
            trend = ""
            if cell.delta is not None and cell.delta > NOISE_FLOOR:
                trend = " (improved, still weak)"
            elif cell.delta is not None and cell.delta < -NOISE_FLOOR:
                trend = " (regressed)"
            out.append(
                f"  {cell.conversation_id}  {title[:30]:<30} {cell.metric[:20]:<20} "
                f"{_pct(cell.after)}{trend}"
            )
        first = weak[0]
        if first.explanation:
            out.append("")
            out.append(f"  Why {first.conversation_id} is still weak - the judge's own words:")
            for chunk in _wrap(first.explanation, width - 6, limit=5):
                out.append(f"    {chunk}")
    out.append("")

    # --- movement ----------------------------------------------------------
    fixed = [
        c for c in cells
        if c.before is not None and c.after is not None
        and c.before <= WEAK_THRESHOLD and c.after > WEAK_THRESHOLD
    ]
    regressed = [
        c for c in cells
        if c.delta is not None and c.delta < -NOISE_FLOOR
    ]
    out.append("-" * width)
    out.append(f"FIXED BY THIS CHANGE ({len(fixed)}){'':<8}REGRESSED ({len(regressed)})")
    out.append("-" * width)
    for index in range(max(len(fixed), len(regressed), 1)):
        left = ""
        if index < len(fixed):
            cell = fixed[index]
            left = f"{cell.conversation_id} {cell.metric[:22]}"
        right = ""
        if index < len(regressed):
            cell = regressed[index]
            right = (f"{cell.conversation_id} {cell.metric[:18]} "
                     f"{_pct(cell.before)}->{_pct(cell.after)}")
        if not left and not right:
            left = "(none)"
        out.append(f"  {left:<36}{right}")
    out.append("")

    # --- what changed ------------------------------------------------------
    out.append("-" * width)
    out.append("WHAT CHANGED IN THE PROMPT")
    out.append("-" * width)
    for entry in prompt_diff[:26]:
        out.append(f"  {entry}")
    out.append("")

    if failures:
        out.append("-" * width)
        out.append(f"EVALUATION FAILURES ({len(failures)}) - scores above exclude these")
        out.append("-" * width)
        for failure in failures[:6]:
            out.append(f"  {failure}")
        out.append("")

    out.append(line)
    return "\n".join(out)


def _wrap(text: str, width: int, limit: int = 6) -> list[str]:
    """Naive word wrap, collapsing newlines. Enough for quoting explanations."""
    words = " ".join(text.split()).split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
            if len(lines) >= limit:
                lines[-1] = lines[-1].rstrip(".,") + " ..."
                return lines
        else:
            current = f"{current} {word}".strip()
    if current and len(lines) < limit:
        lines.append(current)
    return lines


def render_markdown(
    *,
    rows: list[MetricRow],
    cells: list[CaseCell],
    conversations: list[Conversation],
    golden_set_label: str,
    baseline_name: str,
    improved_name: str,
    judge_calls: int,
    cost_usd: float,
    duration_s: float,
    prompt_diff: list[str],
    failures: list[str],
    before_breaches: list[Breach],
    after_breaches: list[Breach],
    evaluators: tuple[Evaluator, ...],
) -> str:
    """Same story as the terminal view, plus the full matrix for whoever digs in."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out: list[str] = []

    out.append("# Customer service agent - quality report")
    out.append("")
    out.append(f"_Generated {stamp}_")
    out.append("")
    out.append(f"- **Golden set:** `{golden_set_label}` - {len(conversations)} conversations, "
               f"{sum(c.turn_count for c in conversations)} turns, same set scored before and after")
    out.append(f"- **Change:** `prompts/{baseline_name}.md` -> `prompts/{improved_name}.md`")
    out.append(f"- **Cost of this run:** {judge_calls} judge calls, ~${cost_usd:.2f}, "
               f"{duration_s:.0f}s")
    out.append("")

    out.append("## What changed")
    out.append("")
    out.append("| Metric | Before | After | Change | |")
    out.append("|---|---|---|---|---|")
    for row in rows:
        out.append(
            f"| {row.metric} | {_pct(row.before).strip()} | {_pct(row.after).strip()} "
            f"| {_delta_cell(row).strip()} | {_verdict_mark(row)} |"
        )
    out.append("")
    out.append(f"Higher is always better. Changes smaller than {NOISE_FLOOR * 100:.0f} "
               "points are marked `~`: LLM judges are not deterministic, so small "
               "moves are noise, not progress.")
    out.append("")

    out.append("## Red lines")
    out.append("")
    out.append("Deterministic regex checks over the replies; no judge, so any change is real. "
               "Counts are cases breaching; lower is better.")
    out.append("")
    out.append("| Red line | Before | After | Cases after |")
    out.append("|---|---|---|---|")
    for red_line, before_ids, after_ids in red_line_rows(before_breaches, after_breaches):
        out.append(f"| {red_line.label} | {len(before_ids)} | {len(after_ids)} | "
                   f"{', '.join(sorted(after_ids)) or '-'} |")
    out.append("")
    for breach in after_breaches:
        out.append(f"- after, {breach.conversation_id} turn {breach.turn}, {breach.red_line.label}: "
                   f"`{breach.evidence}`")
    if after_breaches:
        out.append("")

    out.append("### What each metric means")
    out.append("")
    for evaluator in evaluators:
        out.append(f"- **{evaluator.metric}** ({evaluator.evaluator_id}, "
                   f"{evaluator.level.lower()}-level) - {evaluator.blurb}")
    out.append("")

    out.append("## Per-conversation breakdown")
    out.append("")
    metrics = [e.metric for e in evaluators]
    out.append("| Case | Title | " + " | ".join(metrics) + " |")
    out.append("|---|---|" + "---|" * len(metrics))
    by_case: dict[str, dict[str, CaseCell]] = {}
    for cell in cells:
        by_case.setdefault(cell.conversation_id, {})[cell.metric] = cell
    for conversation in conversations:
        row_cells = by_case.get(conversation.id, {})
        parts = []
        for metric in metrics:
            cell = row_cells.get(metric)
            if cell is None or cell.after is None:
                parts.append("--")
            else:
                mark = ""
                if cell.delta is not None and abs(cell.delta) >= NOISE_FLOOR:
                    mark = " ^" if cell.delta > 0 else " v"
                parts.append(
                    f"{_pct(cell.before).strip()}->{_pct(cell.after).strip()}{mark}"
                )
        out.append(f"| {conversation.id} | {conversation.title} | " + " | ".join(parts) + " |")
    out.append("")

    weak = [c for c in cells if c.after is not None and c.after <= WEAK_THRESHOLD]
    weak.sort(key=lambda c: (c.after or 0.0))
    out.append("## Cases still weak after the change")
    out.append("")
    if not weak:
        out.append(f"None. Every case scored above {WEAK_THRESHOLD * 100:.0f}%.")
    else:
        titles = {c.id: c.title for c in conversations}
        for cell in weak[:8]:
            out.append(f"### {cell.conversation_id} - {titles.get(cell.conversation_id, '')}"
                       f" ({cell.metric}: {_pct(cell.after).strip()})")
            out.append("")
            if cell.explanation:
                out.append("> " + " ".join(cell.explanation.split())[:1200])
            else:
                out.append("> (no explanation returned)")
            out.append("")

    out.append("## Prompt change")
    out.append("")
    out.append("```diff")
    out.extend(prompt_diff)
    out.append("```")
    out.append("")

    if failures:
        out.append("## Evaluation failures")
        out.append("")
        out.append(f"{len(failures)} evaluation(s) failed and are excluded from the "
                   "averages above:")
        out.append("")
        for failure in failures:
            out.append(f"- `{failure}`")
        out.append("")

    return "\n".join(out)

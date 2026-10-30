"""Judge stability: score the same transcript N times and measure the spread.

This tests repeatability, not correctness: a judge can agree with itself and still be
wrong, which only human-labelled calibration catches. The point here is to replace
the assumed NOISE_FLOOR with a measured one.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from cs_eval.config import Evaluator
from cs_eval.evaluate import Score
from cs_eval.report import NOISE_FLOOR, aggregate_metric


@dataclass(frozen=True)
class Spread:
    evaluator: Evaluator
    conversation_id: str
    values: list[float]  # one conversation-level aggregate per repeat

    @property
    def range(self) -> float:
        return max(self.values) - min(self.values) if self.values else 0.0

    @property
    def stdev(self) -> float:
        return pstdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def within_noise_floor(self) -> bool:
        return self.range < NOISE_FLOOR


def spreads(
    repeats: list[list[Score]], evaluators: tuple[Evaluator, ...], conversation_ids: list[str]
) -> list[Spread]:
    """One Spread per (evaluator, conversation) across the repeated scorings."""
    out: list[Spread] = []
    for evaluator in evaluators:
        for cid in conversation_ids:
            values = [
                value
                for scores in repeats
                if (value := aggregate_metric(scores, evaluator, cid)) is not None
            ]
            out.append(Spread(evaluator, cid, values))
    return out


def render(results: list[Spread], repeats: int) -> str:
    lines = [f"JUDGE STABILITY ({repeats} repeats per case; repeatability, not correctness)", ""]
    lines.append(f"{'Metric':<24}{'Case':<9}{'Mean':>6}{'Range':>7}{'Stdev':>7}  Scores")
    for spread in results:
        scores = " ".join(f"{v:.2f}" for v in spread.values)
        flag = "" if spread.within_noise_floor else "  > noise floor"
        lines.append(
            f"{spread.evaluator.metric[:23]:<24}{spread.conversation_id:<9}"
            f"{mean(spread.values) if spread.values else 0:>6.2f}{spread.range:>7.2f}"
            f"{spread.stdev:>7.2f}  {scores}{flag}"
        )
    worst = max((s.range for s in results), default=0.0)
    lines.append("")
    lines.append(
        f"Largest range: {worst:.2f}. Report NOISE_FLOOR: {NOISE_FLOOR:.2f}. "
        + ("Deltas at the floor can still be judge noise." if worst >= NOISE_FLOOR
           else "Observed spread sits under the floor.")
    )
    return "\n".join(lines)

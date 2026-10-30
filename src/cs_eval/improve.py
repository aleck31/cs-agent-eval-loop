"""Derives an improved system prompt from the baseline run's judge explanations.

This exists so the "after" prompt is not hand-tuned to beat the test. The model
that writes it sees only:

  * the current prompt
  * the low-scoring cases, with the judge's stated reason for each

It never sees the golden set's ground-truth assertions or any holdout case, and it is told not to
reference specific order IDs. Otherwise we would be teaching to the test, and the
comparison would prove nothing.

The output is written to prompts/improved.md with a header marking it
machine-proposed, so nobody mistakes it for hand-authored guidance.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import boto3
from botocore.config import Config

from cs_eval import settings
from cs_eval.config import Evaluator
from cs_eval.evaluate import Score
from cs_eval.report import WEAK_THRESHOLD, _normalise

INSTRUCTION = """\
You are improving the system prompt of a customer-service AI agent.

Below is the agent's CURRENT system prompt, followed by evaluation findings from an
automated quality review. Each finding is a case where the agent scored poorly, with
the evaluator's explanation of what went wrong.

Write a REPLACEMENT system prompt that fixes the root causes.

Rules:
- Fix the general behaviour, not the specific examples. Do not mention any order
  number, customer name, or case ID from the findings.
- Keep it concise and operational: rules an agent can follow, not essays.
- Preserve what the current prompt gets right, including replying in the customer's
  own language.
- Do not invent company policies or specific figures that are not evidenced in the
  findings. If a rule depends on policy detail, instruct the agent to retrieve it
  rather than stating a value.
- Output ONLY the replacement prompt text. No preamble, no code fences, no commentary.
"""


def collect_findings(
    scores: list[Score], evaluators: tuple[Evaluator, ...], limit: int = 24
) -> list[dict[str, str]]:
    """Pick the weakest scored results and their explanations."""
    by_id = {e.evaluator_id: e for e in evaluators}
    candidates = []
    for score in scores:
        evaluator = by_id.get(score.evaluator_id)
        if evaluator is None or not score.ok or not score.explanation:
            continue
        normalised = _normalise(score.value, evaluator)
        if normalised is None or normalised > WEAK_THRESHOLD:
            continue
        candidates.append((normalised, score, evaluator))

    candidates.sort(key=lambda item: item[0])
    return [
        {
            "metric": evaluator.metric,
            "evaluator": evaluator.evaluator_id,
            "score": f"{normalised:.2f}",
            "judge_explanation": " ".join(score.explanation.split())[:1400],
        }
        for normalised, score, evaluator in candidates[:limit]
    ]


def propose_prompt(
    *,
    current_prompt: str,
    findings: list[dict[str, str]],
    region: str,
    max_tokens: int = 8000,
) -> tuple[str, int, int]:
    """Ask Bedrock for a replacement prompt. Returns (text, in_tokens, out_tokens)."""
    if not findings:
        raise ValueError("no findings to learn from; baseline scored clean")

    client = boto3.client(
        "bedrock-runtime", region_name=region, config=Config(read_timeout=180)
    )
    payload = (
        f"{INSTRUCTION}\n\n"
        f"=== CURRENT SYSTEM PROMPT ===\n{current_prompt}\n\n"
        f"=== EVALUATION FINDINGS ({len(findings)} weak results) ===\n"
        f"{json.dumps(findings, ensure_ascii=False, indent=2)}\n"
    )

    response = client.converse(
        modelId=settings.get("CS_EVAL_MODEL_ID"),
        messages=[{"role": "user", "content": [{"text": payload}]}],
        inferenceConfig={"maxTokens": max_tokens},
    )
    # Adaptive thinking puts a reasoning block before the text, so join text blocks only.
    blocks = response["output"]["message"]["content"]
    text = "".join(b["text"] for b in blocks if "text" in b).strip()
    usage = response.get("usage", {})
    return text, usage.get("inputTokens", 0), usage.get("outputTokens", 0)


def machine_proposed_header(finding_count: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        "<!-- MACHINE-PROPOSED PROMPT - NOT HAND-AUTHORED\n"
        f"     Generated {stamp} by `make propose`.\n"
        f"     Derived from {finding_count} low-scoring results in the baseline run,\n"
        "     using only the judges' explanations. The generating model did not see\n"
        "     the dataset's ground-truth assertions, so this prompt was not tuned to\n"
        "     the test. Review it before using anything like it in production. -->\n"
    )

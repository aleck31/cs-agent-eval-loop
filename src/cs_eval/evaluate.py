"""Calls the AgentCore Evaluate API and normalises the results.

Design notes worth knowing before you change anything here:

* We call `Evaluate` once per (conversation, evaluator). A trace-level evaluator
  then returns one result per turn in a single response. This keeps every call
  comfortably under the API's 10-result cap.
* `Evaluate` is on the DATA plane client (`bedrock-agentcore`). `ListEvaluators`
  is on the CONTROL plane client (`bedrock-agentcore-control`). They are different
  clients; mixing them up produces a confusing "operation not found".
* Ground truth goes in `evaluationReferenceInputs`: session-scoped assertions and
  expected tool trajectory. This is what lifts GoalSuccessRate from a vibe check
  to a test against your SOP.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from cs_eval.config import SESSION, TOOL_CALL, TRACE, Evaluator, estimate_cost, judge_cost
from cs_eval.dataset import Conversation
from cs_eval.spans import build_session_spans

RETRYABLE = {"ThrottlingException", "InternalServerException", "ServiceQuotaExceededException"}


class BudgetExceeded(RuntimeError):
    """Raised when a run would cross, or has crossed, the spend ceiling."""


class MissingEvaluator(RuntimeError):
    """A repo-defined evaluator has not been created in this account and region."""


@dataclass
class Score:
    """One evaluation result, normalised."""

    conversation_id: str
    evaluator_id: str
    metric: str
    level: str
    value: float | None
    label: str
    explanation: str
    trace_id: str | None = None
    span_id: str | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and self.value is not None


@dataclass
class RunResult:
    """All scores for one prompt variant."""

    variant: str
    golden_set_fingerprint: str = ""
    scores: list[Score] = field(default_factory=list)
    judge_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: list[str] = field(default_factory=list)
    transcripts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return judge_cost(self.input_tokens, self.output_tokens)


class BudgetGuard:
    """Tracks spend against a hard ceiling, across every phase of the run."""

    def __init__(self, ceiling_usd: float) -> None:
        self.ceiling_usd = ceiling_usd
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def spent_usd(self) -> float:
        return judge_cost(self.input_tokens, self.output_tokens)

    @property
    def remaining_usd(self) -> float:
        return self.ceiling_usd - self.spent_usd

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def check_before(self, projected_usd: float) -> None:
        if self.spent_usd + projected_usd > self.ceiling_usd:
            raise BudgetExceeded(
                f"projected spend ${self.spent_usd + projected_usd:.2f} would exceed "
                f"the ${self.ceiling_usd:.2f} ceiling (already spent ${self.spent_usd:.2f}). "
                f"Reduce --limit or use --evaluators minimal."
            )

    def check_during(self) -> None:
        if self.spent_usd > self.ceiling_usd:
            raise BudgetExceeded(
                f"spend ${self.spent_usd:.2f} crossed the ${self.ceiling_usd:.2f} "
                f"ceiling mid-run; stopping"
            )


class EvaluationClient:
    """Thin wrapper over Evaluate, with retries and result normalisation."""

    def __init__(self, region: str, guard: BudgetGuard | None = None) -> None:
        self.region = region
        self.guard = guard
        config = Config(retries={"max_attempts": 3, "mode": "standard"}, read_timeout=120)
        self._data = boto3.client("bedrock-agentcore", region_name=region, config=config)
        self._control = boto3.client("bedrock-agentcore-control", region_name=region)
        self._custom_ids: dict[str, str] | None = None

    # -- startup validation --------------------------------------------------
    def list_evaluators(self) -> list[dict[str, Any]] | None:
        """Ask the account which evaluators actually exist.

        Returns None when we lack `ListEvaluators` permission -- which happens,
        and is not fatal, since Evaluate is authorised separately.

        We ask rather than hardcode, because every published count we checked was
        wrong for this account: FAST's guide says 15 and lists two that do not
        exist here (Builtin.Maliciousness, Builtin.ContextRelevance); the AWS CDK
        docs say 13. This account returns 31 (18 Builtin + 13 ThirdParty).

        Note the response key is `evaluators` and the level field is `level` --
        FAST's guide says `evaluatorSummaries`, which yields a silent empty list.
        """
        try:
            found: list[dict[str, Any]] = []
            token = None
            while True:
                kwargs: dict[str, Any] = {"maxResults": 100}
                if token:
                    kwargs["nextToken"] = token
                response = self._control.list_evaluators(**kwargs)
                found.extend(response.get("evaluators", []))
                token = response.get("nextToken")
                if not token:
                    return found
        except ClientError:
            return None

    def custom_evaluator_id(self, evaluator: Evaluator) -> str | None:
        """Account-specific ID (`<name>-<10 chars>`) of a repo-defined evaluator, if created."""
        if self._custom_ids is None:
            found = self.list_evaluators() or []
            self._custom_ids = {}
            for e in found:
                eid = e.get("evaluatorId", "")
                name, _, suffix = eid.rpartition("-")
                if name and len(suffix) == 10:
                    self._custom_ids[name] = eid
        return self._custom_ids.get(evaluator.short_name)

    def resolve(self, evaluator: Evaluator) -> str:
        """The ID to send to Evaluate; Evaluate takes IDs, not ARNs (verified live)."""
        if not evaluator.is_custom:
            return evaluator.evaluator_id
        resolved = self.custom_evaluator_id(evaluator)
        if resolved is None:
            raise MissingEvaluator(
                f"{evaluator.short_name} is not created in this account/region; run `cs-eval setup`"
            )
        return resolved

    def evaluator_arn(self, evaluator: Evaluator) -> str:
        """ARN form, which start-recommendation requires where Evaluate takes an ID."""
        if not evaluator.is_custom:
            return f"arn:aws:bedrock-agentcore:::evaluator/{evaluator.evaluator_id}"
        return self._control.get_evaluator(evaluatorId=self.resolve(evaluator))["evaluatorArn"]

    def ensure_custom(self, evaluator: Evaluator, config: dict[str, Any]) -> tuple[str, str]:
        """Create the evaluator, or update it in place so the code stays the source of truth."""
        existing = self.custom_evaluator_id(evaluator)
        if existing:
            self._control.update_evaluator(
                evaluatorId=existing, evaluatorConfig=config, description=evaluator.blurb
            )
            return existing, "updated"
        response = self._control.create_evaluator(
            evaluatorName=evaluator.short_name,
            level=evaluator.level,
            evaluatorConfig=config,
            description=evaluator.blurb,
        )
        created = response["evaluatorId"]
        if self._custom_ids is not None:
            self._custom_ids[evaluator.short_name] = created
        return created, "created"

    def available_evaluator_ids(self) -> set[str] | None:
        evaluators = self.list_evaluators()
        if evaluators is None:
            return None
        return {e["evaluatorId"] for e in evaluators if e.get("evaluatorId")}

    # -- the call ------------------------------------------------------------
    def _evaluate_with_retry(self, request: dict[str, Any], attempts: int = 4) -> dict[str, Any]:
        delay = 2.0
        last: Exception | None = None
        for _ in range(attempts):
            try:
                return self._data.evaluate(**request)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                last = exc
                if code not in RETRYABLE:
                    raise
                time.sleep(delay)
                delay *= 2
        raise last  # type: ignore[misc]

    def score_conversation(
        self,
        *,
        conversation: Conversation,
        transcript_turns: list[dict[str, Any]],
        evaluators: tuple[Evaluator, ...],
        system_prompt: str,
        tool_specs: list[dict[str, Any]],
    ) -> tuple[list[Score], int, int, list[str]]:
        """Run every evaluator against one conversation.

        Returns (scores, input_tokens, output_tokens, failure_messages).
        """
        session_id = f"{conversation.id}-{uuid.uuid4().hex[:8]}"
        spans, trace_ids = build_session_spans(
            session_id=session_id,
            turns=transcript_turns,
            system_prompt=system_prompt,
            tool_specs=tool_specs,
        )

        reference_inputs = self._reference_inputs(conversation, session_id)

        scores: list[Score] = []
        failures: list[str] = []
        input_tokens = output_tokens = 0

        for evaluator in evaluators:
            try:
                request: dict[str, Any] = {
                    "evaluatorId": self.resolve(evaluator),
                    "evaluationInput": {"sessionSpans": spans},
                }
                if reference_inputs:
                    request["evaluationReferenceInputs"] = reference_inputs
                response = self._evaluate_with_retry(request)
            except (ClientError, MissingEvaluator) as exc:
                code = (
                    exc.response.get("Error", {}).get("Code", "Error")
                    if isinstance(exc, ClientError) else "MissingEvaluator"
                )
                message = f"{conversation.id}/{evaluator.short_name}: {code}"
                failures.append(message)
                scores.append(
                    Score(
                        conversation_id=conversation.id,
                        evaluator_id=evaluator.evaluator_id,
                        metric=evaluator.metric,
                        level=evaluator.level,
                        value=None,
                        label="",
                        explanation="",
                        error=str(exc),
                    )
                )
                continue

            for result in response.get("evaluationResults", []):
                usage = result.get("tokenUsage") or {}
                input_tokens += usage.get("inputTokens", 0)
                output_tokens += usage.get("outputTokens", 0)
                context = (result.get("context") or {}).get("spanContext") or {}

                if result.get("errorCode"):
                    failures.append(
                        f"{conversation.id}/{evaluator.short_name}: {result['errorCode']}"
                    )

                scores.append(
                    Score(
                        conversation_id=conversation.id,
                        evaluator_id=evaluator.evaluator_id,
                        metric=evaluator.metric,
                        level=evaluator.level,
                        value=result.get("value"),
                        label=result.get("label", ""),
                        explanation=result.get("explanation", ""),
                        trace_id=context.get("traceId"),
                        span_id=context.get("spanId"),
                        error=result.get("errorCode"),
                        input_tokens=usage.get("inputTokens", 0),
                        output_tokens=usage.get("outputTokens", 0),
                    )
                )

            if self.guard:
                self.guard.record(input_tokens, output_tokens)
                input_tokens = output_tokens = 0
                self.guard.check_during()

        return scores, input_tokens, output_tokens, failures

    @staticmethod
    def _reference_inputs(
        conversation: Conversation, session_id: str
    ) -> list[dict[str, Any]]:
        """Attach ground truth at session scope."""
        truth = conversation.ground_truth
        if not truth.assertions and not truth.expected_trajectory:
            return []

        entry: dict[str, Any] = {
            "context": {"spanContext": {"sessionId": session_id}}
        }
        if truth.assertions:
            entry["assertions"] = [{"text": a} for a in truth.assertions]
        if truth.expected_trajectory:
            entry["expectedTrajectory"] = {"toolNames": list(truth.expected_trajectory)}
        return [entry]


def project_cost(
    conversations: list[Conversation], evaluators: tuple[Evaluator, ...], sides: int = 2
) -> tuple[int, float]:
    """Estimate judge calls and USD for a full before/after run.

    Tool calls are unknown until the agent runs, so we assume one per turn, which
    is close for this dataset and errs slightly high.
    """
    calls = 0
    for conversation in conversations:
        turns = conversation.turn_count
        assumed_tool_calls = turns
        for evaluator in evaluators:
            if evaluator.level == SESSION:
                calls += 1
            elif evaluator.level == TRACE:
                calls += turns
            elif evaluator.level == TOOL_CALL:
                calls += assumed_tool_calls
    calls *= sides
    return calls, estimate_cost(calls)

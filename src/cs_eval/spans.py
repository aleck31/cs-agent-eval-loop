"""Transcript -> OpenInference spans for the AgentCore Evaluate API.

This is the load-bearing module of the offline path. AgentCore Evaluations reads
telemetry through OpenTelemetry or OpenInference semantic conventions ("generic
framework support"). We emit OpenInference, because that is the convention the
OpenInference LangChain/LangGraph instrumentation emits in production -- so the
spans built here have the same shape the customer's real agent will produce.

Span layout, per conversation turn:

    AGENT span   (root of one trace)   <- what TRACE-level evaluators score
      +- LLM span                      <- carries system prompt + available tools
      +- TOOL span (one per tool call) <- what TOOL_CALL-level evaluators score

All turns of a conversation share one `session.id`, which is what SESSION-level
evaluators (GoalSuccessRate) aggregate over.

Verified against the live Evaluate API on 2026-08-20; see docs in README.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

# The service picks its reader from scope.name. This prefix selects the
# OpenInference convention. Do not change it without re-probing the API.
SCOPE_NAME = "openinference.instrumentation.langchain"

SERVICE_NAME = "cs-agent-eval-loop"


def _trace_id() -> str:
    """32 hex chars, per W3C trace context."""
    return uuid.uuid4().hex


def _span_id() -> str:
    """16 hex chars, per W3C trace context."""
    return uuid.uuid4().hex[:16]


def _as_text(value: Any) -> str:
    """Stringify a value. The service stringifies extracted values anyway."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _span(
    *,
    name: str,
    kind: str,
    trace_id: str,
    span_id: str,
    session_id: str,
    start_ns: int,
    end_ns: int,
    attributes: dict[str, Any],
    parent_span_id: str | None = None,
) -> dict[str, Any]:
    """Build one span envelope with the attributes the service looks for."""
    attrs: dict[str, Any] = {
        "session.id": session_id,
        "openinference.span.kind": kind,
        **attributes,
    }
    span: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": start_ns,
        "endTimeUnixNano": end_ns,
        "attributes": attrs,
        "status": {"code": "OK"},
        "scope": {"name": SCOPE_NAME},
        "resource": {"attributes": {"service.name": SERVICE_NAME}},
    }
    if parent_span_id:
        span["parentSpanId"] = parent_span_id
    return span


def _tool_schema_attributes(tool_specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Advertise the tools that were available to the model.

    ToolSelectionAccuracy needs to know what the agent *could* have called in
    order to judge whether what it *did* call was right.
    """
    attrs: dict[str, Any] = {}
    for index, spec in enumerate(tool_specs):
        attrs[f"llm.tools.{index}.tool.json_schema"] = json.dumps(spec, ensure_ascii=False)
    return attrs


def build_session_spans(
    *,
    session_id: str,
    turns: list[dict[str, Any]],
    system_prompt: str,
    tool_specs: list[dict[str, Any]],
    start_ns: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert one conversation transcript into spans.

    Each turn is ``{"user": str, "assistant": str, "tool_calls": [...]}`` where a
    tool call is ``{"name": str, "args": dict, "result": Any}``.

    Returns ``(spans, trace_ids)``; ``trace_ids[i]`` is the trace for turn i, used
    to attach per-turn ground truth via ``evaluationReferenceInputs``.
    """
    clock = start_ns if start_ns is not None else time.time_ns()
    spans: list[dict[str, Any]] = []
    trace_ids: list[str] = []

    for turn in turns:
        trace_id = _trace_id()
        agent_span_id = _span_id()
        trace_ids.append(trace_id)

        turn_start = clock
        turn_end = clock + 2_000_000_000  # nominal 2s turn

        # --- AGENT span: the unit a TRACE-level evaluator scores ---------------
        spans.append(
            _span(
                name="AgentExecutor",
                kind="AGENT",
                trace_id=trace_id,
                span_id=agent_span_id,
                session_id=session_id,
                start_ns=turn_start,
                end_ns=turn_end,
                attributes={
                    "input.value": _as_text(turn["user"]),
                    "input.mime_type": "text/plain",
                    "output.value": _as_text(turn["assistant"]),
                    "output.mime_type": "text/plain",
                },
            )
        )

        # --- LLM span: carries system prompt and the available-tool catalogue --
        llm_attributes: dict[str, Any] = {
            "llm.model_name": turn.get("model", "unknown"),
            "llm.system": "aws.bedrock",
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": system_prompt,
            "llm.input_messages.1.message.role": "user",
            "llm.input_messages.1.message.content": _as_text(turn["user"]),
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": _as_text(turn["assistant"]),
            "input.value": _as_text(turn["user"]),
            "output.value": _as_text(turn["assistant"]),
        }
        llm_attributes.update(_tool_schema_attributes(tool_specs))
        spans.append(
            _span(
                name="ChatBedrock",
                kind="LLM",
                trace_id=trace_id,
                span_id=_span_id(),
                session_id=session_id,
                start_ns=turn_start + 100_000_000,
                end_ns=turn_end - 100_000_000,
                attributes=llm_attributes,
                parent_span_id=agent_span_id,
            )
        )

        # --- TOOL spans: the unit a TOOL_CALL-level evaluator scores -----------
        for offset, call in enumerate(turn.get("tool_calls", [])):
            tool_name = call["name"]
            spans.append(
                _span(
                    name=tool_name,
                    kind="TOOL",
                    trace_id=trace_id,
                    span_id=_span_id(),
                    session_id=session_id,
                    start_ns=turn_start + 200_000_000 + offset * 50_000_000,
                    end_ns=turn_start + 400_000_000 + offset * 50_000_000,
                    attributes={
                        # Name is read from several places depending on
                        # convention; set all of them.
                        "tool.name": tool_name,
                        "gen_ai.tool.name": tool_name,
                        "tool.description": call.get("description", ""),
                        "input.value": _as_text(call.get("args", {})),
                        "input.mime_type": "application/json",
                        "output.value": _as_text(call.get("result", "")),
                        "output.mime_type": "application/json",
                    },
                    parent_span_id=agent_span_id,
                )
            )

        clock = turn_end + 500_000_000

    return spans, trace_ids

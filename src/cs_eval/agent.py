"""The LangGraph customer-service agent under test.

Shape borrowed from FAST's `patterns/langgraph-single-agent`: LangChain v1
`create_agent` over `ChatBedrock`, same model ID. Differences, deliberate:

  * tools are local fakes, not Gateway MCP  -> runs with no deployed infrastructure
  * no AgentCore Memory checkpointer        -> each conversation is independent
  * system prompt is injected per run       -> that is the variable we A/B

Running this needs only `bedrock:InvokeModel`. Nothing is deployed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cs_eval import settings
from cs_eval.tools import TOOLS


def model_id() -> str:
    """Resolved lazily: a module constant would freeze before `.env` is loaded."""
    return settings.get("CS_EVAL_MODEL_ID")


@dataclass
class Turn:
    """One user message and everything the agent did in response."""

    user: str
    assistant: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = field(default_factory=model_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "user": self.user,
            "assistant": self.assistant,
            "tool_calls": self.tool_calls,
            "model": self.model,
        }


@dataclass
class Transcript:
    """A full conversation, plus which prompt produced it."""

    conversation_id: str
    variant: str
    turns: list[Turn]

    @property
    def tool_call_count(self) -> int:
        return sum(len(t.tool_calls) for t in self.turns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "variant": self.variant,
            "turns": [t.as_dict() for t in self.turns],
        }


def build_agent(system_prompt: str, region: str, model: str | None = None):
    """Compile the LangGraph agent. Same construction as the reference pattern."""
    from langchain.agents import create_agent

    # No temperature: current Claude models reject sampling parameters.
    bedrock = ChatBedrock(
        model_id=model or model_id(), region_name=region, beta_use_converse_api=True
    )
    return create_agent(model=bedrock, tools=TOOLS, system_prompt=system_prompt)


def _extract_turn(user_text: str, messages: list[Any]) -> Turn:
    """Pull the assistant reply and tool calls out of one turn's new messages.

    Tool call args come from the AIMessage that requested them; results come from
    the matching ToolMessage, correlated by tool_call_id.
    """
    requested: dict[str, dict[str, Any]] = {}
    results: dict[str, str] = {}
    final_text = ""

    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                requested[call["id"]] = {
                    "name": call["name"],
                    "args": call.get("args", {}),
                }
            text = getattr(message, "text", "") or ""
            if text and text.strip():
                final_text = text.strip()
        elif isinstance(message, ToolMessage):
            results[message.tool_call_id] = str(message.content)

    tool_calls = [
        {
            "name": info["name"],
            "args": info["args"],
            "result": results.get(call_id, ""),
        }
        for call_id, info in requested.items()
    ]
    return Turn(user=user_text, assistant=final_text, tool_calls=tool_calls)


def run_conversation(
    agent, conversation_id: str, variant: str, user_turns: list[str]
) -> Transcript:
    """Replay scripted user turns against the agent, keeping history between them."""
    history: list[Any] = []
    turns: list[Turn] = []

    for user_text in user_turns:
        history.append(HumanMessage(content=user_text))
        before = len(history)

        result = agent.invoke({"messages": history})
        history = list(result["messages"])

        turns.append(_extract_turn(user_text, history[before:]))

    return Transcript(conversation_id=conversation_id, variant=variant, turns=turns)

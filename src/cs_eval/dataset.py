"""Dataset loading and validation.

The schema is intentionally small so you can swap this file for an export of your
own tickets without touching code. Validation is strict and fails loudly:
a malformed dataset should not become a mysterious evaluation result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Evaluate returns at most 10 results per call and keeps only the last 10, so a
# trace-level evaluator would silently drop early turns beyond this.
MAX_TURNS = 10
RECOMMENDED_MAX_TURNS = 5


@dataclass(frozen=True)
class GroundTruth:
    assertions: list[str] = field(default_factory=list)
    expected_trajectory: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Conversation:
    id: str
    lang: str
    title: str
    category: str
    user_turns: list[str]
    ground_truth: GroundTruth
    notes: str = ""

    @property
    def turn_count(self) -> int:
        return len(self.user_turns)


class DatasetError(ValueError):
    """Raised when the dataset file is malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetError(message)


def load_dataset(path: str | Path) -> list[Conversation]:
    """Load and validate the conversation dataset."""
    file_path = Path(path)
    _require(file_path.exists(), f"dataset not found: {file_path}")

    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    entries = raw.get("conversations")
    _require(isinstance(entries, list) and bool(entries), "dataset has no 'conversations' list")

    conversations: list[Conversation] = []
    seen: set[str] = set()

    for index, entry in enumerate(entries):
        where = f"conversations[{index}]"
        _require(isinstance(entry, dict), f"{where} is not a mapping")

        conversation_id = entry.get("id")
        _require(bool(conversation_id), f"{where} is missing 'id'")
        _require(conversation_id not in seen, f"duplicate conversation id: {conversation_id}")
        seen.add(conversation_id)

        turns = entry.get("user_turns")
        _require(
            isinstance(turns, list) and bool(turns),
            f"{conversation_id}: 'user_turns' must be a non-empty list",
        )
        _require(
            all(isinstance(t, str) and t.strip() for t in turns),
            f"{conversation_id}: every user turn must be a non-empty string",
        )
        _require(
            len(turns) <= MAX_TURNS,
            f"{conversation_id}: {len(turns)} turns exceeds the {MAX_TURNS}-result "
            f"Evaluate cap; split the conversation",
        )

        truth = entry.get("ground_truth") or {}
        _require(isinstance(truth, dict), f"{conversation_id}: 'ground_truth' must be a mapping")

        conversations.append(
            Conversation(
                id=conversation_id,
                lang=entry.get("lang", "en"),
                title=entry.get("title", conversation_id),
                category=entry.get("category", "uncategorised"),
                user_turns=list(turns),
                ground_truth=GroundTruth(
                    assertions=list(truth.get("assertions") or []),
                    expected_trajectory=list(truth.get("expected_trajectory") or []),
                ),
                notes=entry.get("notes", ""),
            )
        )

    return conversations


def dataset_warnings(conversations: list[Conversation]) -> list[str]:
    """Non-fatal advice about the dataset."""
    warnings: list[str] = []
    for conversation in conversations:
        if conversation.turn_count > RECOMMENDED_MAX_TURNS:
            warnings.append(
                f"{conversation.id}: {conversation.turn_count} turns is above the "
                f"recommended {RECOMMENDED_MAX_TURNS}; trace-level cost scales with turns"
            )
        if not conversation.ground_truth.assertions:
            warnings.append(
                f"{conversation.id}: no assertions -- GoalSuccessRate will score "
                f"without ground truth and be far less meaningful"
            )
    return warnings


def summarise(conversations: list[Conversation]) -> dict[str, Any]:
    languages: dict[str, int] = {}
    for conversation in conversations:
        languages[conversation.lang] = languages.get(conversation.lang, 0) + 1
    return {
        "conversations": len(conversations),
        "turns": sum(c.turn_count for c in conversations),
        "languages": languages,
    }

"""Chat template helpers tailored for the THCE medical coding tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Sequence


class Stage(Enum):
    """Training stages supported by the THCE project."""

    STAGE_1 = "stage_1"
    STAGE_2 = "stage_2"
    STAGE_3 = "stage_3"

    @classmethod
    def from_string(cls, value: str) -> "Stage":
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        msg = f"Unsupported stage '{value}'. Expected one of {[m.value for m in cls]}."
        raise ValueError(msg)


@dataclass(frozen=True)
class ChatTemplate:
    """Stores a system prompt and metadata used when building chat records."""

    stage: Stage
    system_prompt: str
    assistant_prefix: str
    reasoning_required: bool

    def to_dict(self) -> Dict[str, str]:
        return {
            "stage": self.stage.value,
            "system_prompt": self.system_prompt,
            "assistant_prefix": self.assistant_prefix,
            "reasoning_required": str(self.reasoning_required),
        }


TEMPLATES: Dict[Stage, ChatTemplate] = {
    Stage.STAGE_1: ChatTemplate(
        stage=Stage.STAGE_1,
        system_prompt=(
            "You are a medical coding assistant. Convert clinical narratives to billing codes "
            "without showing your reasoning."
        ),
        assistant_prefix="",
        reasoning_required=False,
    ),
    Stage.STAGE_2: ChatTemplate(
        stage=Stage.STAGE_2,
        system_prompt=(
            "You are a medical coding expert. Use <think> tags to show concise reasoning before "
            "presenting the selected codes."
        ),
        assistant_prefix="<think>\n",
        reasoning_required=True,
    ),
    Stage.STAGE_3: ChatTemplate(
        stage=Stage.STAGE_3,
        system_prompt=(
            "You review medical coding responses. Rank completions by correctness and reasoning quality."
        ),
        assistant_prefix="",
        reasoning_required=True,
    ),
}


def get_template(stage: str | Stage) -> ChatTemplate:
    """Return the template for a given stage string or enum."""
    if isinstance(stage, Stage):
        key = stage
    else:
        key = Stage.from_string(stage)
    return TEMPLATES[key]


def apply_template(
    template: ChatTemplate,
    user_message: str,
    reasoning_lines: Sequence[str] | None,
    completion: str,
) -> List[Dict[str, str]]:
    """Build a chat record from discrete components."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": template.system_prompt},
        {"role": "user", "content": user_message},
    ]

    if template.reasoning_required and reasoning_lines:
        reasoning = "\n".join(reasoning_lines)
        assistant_content = f"<think>\n{reasoning}\n</think>\n{completion}".strip()
    else:
        assistant_content = completion

    messages.append({"role": "assistant", "content": assistant_content})
    return messages


def list_templates() -> Iterable[ChatTemplate]:
    """Iterate over all registered templates."""
    return TEMPLATES.values()

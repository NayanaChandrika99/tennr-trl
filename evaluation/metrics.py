"""Metric helpers for evaluating THCE model outputs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

from utils.code_validator import CodeValidator


@dataclass(frozen=True)
class MetricResult:
    """Container for metric name/value pairs."""

    name: str
    value: float


def normalize_code(code: str | None) -> str:
    return (code or "").strip().upper()


def compute_accuracy(expected: Sequence[str], predicted: Sequence[str]) -> float:
    """Simple exact-match accuracy on normalized codes."""
    if not expected:
        return 0.0
    matches = sum(
        1 for gold, pred in zip(expected, predicted) if normalize_code(gold) == normalize_code(pred)
    )
    return matches / len(expected)


def compute_code_validity(predicted: Sequence[str], validator: CodeValidator) -> float:
    """Share of predictions that exist in the curated code databases."""
    if not predicted:
        return 0.0
    valid = 0
    for code in predicted:
        inferred_type = validator.infer_code_type(code or "")
        if inferred_type and validator.validate(code, inferred_type):
            valid += 1
    return valid / len(predicted)


def extract_reasoning_blocks(responses: Iterable[str]) -> List[str]:
    """Extract `<think>` blocks from assistant responses."""
    blocks: List[str] = []
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    for response in responses:
        match = pattern.search(response or "")
        if match:
            blocks.append(match.group(1).strip())
        else:
            blocks.append("")
    return blocks


def compute_reasoning_coverage(responses: Sequence[str]) -> float:
    """Percentage of responses that contain non-empty `<think>` blocks."""
    if not responses:
        return 0.0
    blocks = extract_reasoning_blocks(responses)
    covered = sum(1 for block in blocks if block)
    return covered / len(responses)


def compute_average_reasoning_length(responses: Sequence[str]) -> float:
    """Average number of lines inside `<think>` blocks."""
    blocks = extract_reasoning_blocks(responses)
    lengths = []
    for block in blocks:
        if not block:
            continue
        lines = [line for line in block.splitlines() if line.strip()]
        lengths.append(len(lines))
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


METRIC_REGISTRY: Dict[str, Callable[..., float]] = {
    "accuracy": compute_accuracy,
    "code_validity": compute_code_validity,
    "reasoning_coverage": compute_reasoning_coverage,
    "average_reasoning_length": compute_average_reasoning_length,
}


def available_metrics() -> List[str]:
    return list(METRIC_REGISTRY.keys())

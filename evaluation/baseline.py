"""Simple baseline predictors for THCE evaluation."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

from utils.code_validator import CodeValidator


def format_prediction(code_type: str, code: str, include_reasoning: bool) -> str:
    code_line = f"{code_type}: {code}"
    if not include_reasoning:
        return code_line
    return f"<think>\nBaseline reasoning placeholder.\n</think>\n{code_line}"


class BaselinePredictor:
    """Generate deterministic or random predictions without running a trained model."""

    def __init__(
        self,
        validator: CodeValidator,
        strategy: str = "echo_ground_truth",
        seed: int | None = None,
    ) -> None:
        self.validator = validator
        self.strategy = strategy
        self.random = random.Random(seed)

    def predict(self, records: Iterable[Dict[str, object]]) -> List[str]:
        strategies = {
            "echo_ground_truth": self._echo_ground_truth,
            "random_valid_code": self._random_valid_code,
            "majority_code": self._majority_code,
        }
        if self.strategy not in strategies:
            msg = f"Unsupported baseline strategy '{self.strategy}'."
            raise ValueError(msg)
        return strategies[self.strategy](list(records))

    def _echo_ground_truth(self, records: List[Dict[str, object]]) -> List[str]:
        predictions: List[str] = []
        for record in records:
            metadata = record.get("_metadata", {}) or {}
            code = str(metadata.get("code", ""))
            code_type = str(metadata.get("code_type", ""))
            reasoning_required = bool(metadata.get("reasoning_present", False))
            predictions.append(format_prediction(code_type, code, reasoning_required))
        return predictions

    def _random_valid_code(self, records: List[Dict[str, object]]) -> List[str]:
        predictions: List[str] = []
        cache: Dict[str, List[str]] = {}
        for record in records:
            metadata = record.get("_metadata", {}) or {}
            code_type = str(metadata.get("code_type", ""))
            reasoning_required = bool(metadata.get("reasoning_present", False))
            if code_type not in cache:
                cache[code_type] = sorted(self.validator.list_codes(code_type))
            codes = cache.get(code_type, [])
            code = self.random.choice(codes) if codes else ""
            predictions.append(format_prediction(code_type, code, reasoning_required))
        return predictions

    def _majority_code(self, records: List[Dict[str, object]]) -> List[str]:
        majority_per_type: Dict[str, str] = {}
        counts: Dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            metadata = record.get("_metadata", {}) or {}
            code_type = str(metadata.get("code_type", ""))
            code = str(metadata.get("code", ""))
            if code:
                counts[code_type][code] += 1
        for code_type, counter in counts.items():
            if counter:
                majority_per_type[code_type] = counter.most_common(1)[0][0]

        predictions: List[str] = []
        for record in records:
            metadata = record.get("_metadata", {}) or {}
            code_type = str(metadata.get("code_type", ""))
            reasoning_required = bool(metadata.get("reasoning_present", False))
            code = majority_per_type.get(code_type, "")
            predictions.append(format_prediction(code_type, code, reasoning_required))
        return predictions

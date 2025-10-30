"""Tokenizer utilities for preparing models in the THCE project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from transformers import PreTrainedTokenizerBase

SPECIAL_TOKENS = {
    "additional_special_tokens": ["<think>", "</think>", "ICD10:", "CPT:", "HCPCS:"],
}


@dataclass(slots=True)
class TokenizerPreparer:
    """Adds project-specific tokens and provides helpers for checking readiness."""

    tokenizer: PreTrainedTokenizerBase

    def ensure_special_tokens(self, extra_tokens: Sequence[str] | None = None) -> int:
        """Add the THCE special tokens to the tokenizer if they are missing."""
        tokens_to_add = list(SPECIAL_TOKENS["additional_special_tokens"])
        if extra_tokens:
            for token in extra_tokens:
                if token not in tokens_to_add:
                    tokens_to_add.append(token)

        new_tokens: List[str] = [
            token for token in tokens_to_add if token not in self.tokenizer.get_vocab()
        ]
        if not new_tokens:
            return 0

        added_count = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": new_tokens},
        )
        return added_count

    def configure_padding(self, side: str = "left") -> None:
        """Set padding side in a consistent way for causal decoders."""
        self.tokenizer.padding_side = side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


def list_missing_tokens(tokenizer: PreTrainedTokenizerBase) -> Iterable[str]:
    """Yield tokens that are not yet part of the tokenizer vocabulary."""
    vocab = tokenizer.get_vocab()
    for token in SPECIAL_TOKENS["additional_special_tokens"]:
        if token not in vocab:
            yield token

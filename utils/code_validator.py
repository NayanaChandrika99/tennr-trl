"""Utilities for validating medical billing codes used in the THCE project."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, TypedDict

CODE_TYPE_ALIASES = {
    "icd10": "ICD10",
    "icd-10": "ICD10",
    "icd": "ICD10",
    "cpt": "CPT",
    "hcpcs": "HCPCS",
}


class CodeRecord(TypedDict, total=False):
    """Dictionary structure for entries in the code databases."""

    code: str
    short_description: str
    long_description: str
    chapter: str
    category: str


@dataclass(slots=True)
class CodeValidator:
    """Validates codes against the curated sample databases stored in ``data/code_databases``."""

    database_dir: Path
    _cache: Dict[str, Set[str]] = field(default_factory=dict, init=False)
    _records: Dict[str, Dict[str, CodeRecord]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.database_dir = self.database_dir.expanduser().resolve()
        if not self.database_dir.exists():
            msg = f"Database directory {self.database_dir} does not exist."
            raise FileNotFoundError(msg)

    def validate(self, code: str, code_type: str) -> bool:
        """Return True if the code exists for the provided type."""
        normalized_type = self._normalize_type(code_type)
        normalized_code = code.strip().upper()
        codes = self._load_codes(normalized_type)
        return normalized_code in codes

    def get(self, code: str, code_type: str) -> Optional[CodeRecord]:
        """Return the record for a given code if available."""
        normalized_type = self._normalize_type(code_type)
        normalized_code = code.strip().upper()
        self._load_codes(normalized_type)
        return self._records[normalized_type].get(normalized_code)

    def list_codes(self, code_type: str) -> Set[str]:
        """Return a set of known codes for the requested type."""
        normalized_type = self._normalize_type(code_type)
        return set(self._load_codes(normalized_type))

    def _load_codes(self, normalized_type: str) -> Set[str]:
        if normalized_type in self._cache:
            return self._cache[normalized_type]

        filename = {
            "ICD10": "icd10_codes.json",
            "CPT": "cpt_codes.json",
            "HCPCS": "hcpcs_codes.json",
        }.get(normalized_type)

        if filename is None:
            msg = f"Unsupported code type: {normalized_type}"
            raise ValueError(msg)

        file_path = self.database_dir / filename
        if not file_path.exists():
            msg = f"Code database file not found: {file_path}"
            raise FileNotFoundError(msg)

        with file_path.open("r", encoding="utf-8") as file:
            data: Iterable[CodeRecord] = json.load(file)

        self._records[normalized_type] = {item["code"].upper(): item for item in data}
        self._cache[normalized_type] = set(self._records[normalized_type])
        return self._cache[normalized_type]

    @staticmethod
    def _normalize_type(code_type: str) -> str:
        key = code_type.strip().lower()
        normalized = CODE_TYPE_ALIASES.get(key, code_type.strip().upper())
        if normalized not in {"ICD10", "CPT", "HCPCS"}:
            msg = f"Unsupported code type: {code_type}"
            raise ValueError(msg)
        return normalized

    @staticmethod
    def infer_code_type(code: str) -> Optional[str]:
        """Infer a likely code type from the code pattern."""
        code = code.strip().upper()
        if re.fullmatch(r"[A-Z]\d{2}(\.\d{1,2}[A-Z]?)?", code):
            return "ICD10"
        if re.fullmatch(r"\d{5}", code):
            return "CPT"
        if re.fullmatch(r"[A-Z]\d{3,4}", code) or re.fullmatch(r"[A-Z]\d{4}[A-Z]", code):
            return "HCPCS"
        return None


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate medical billing codes.")
    parser.add_argument("--code", required=True, help="Code to validate, e.g. E11.9")
    parser.add_argument(
        "--type",
        required=False,
        help="Code type (ICD10, CPT, HCPCS). If omitted the type will be inferred.",
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=Path("data/code_databases"),
        help="Path to the code database directory.",
    )
    parser.add_argument(
        "--show-record",
        action="store_true",
        help="Print the stored record when the code is valid.",
    )
    return parser


def main(args: Optional[argparse.Namespace] = None) -> int:
    parser = build_cli()
    parsed = args or parser.parse_args()

    validator = CodeValidator(parsed.database_dir)
    code_type = parsed.type or validator.infer_code_type(parsed.code)
    if not code_type:
        print("Unable to infer the code type. Provide --type explicitly.")
        return 2

    if validator.validate(parsed.code, code_type):
        print(f"✓ {parsed.code} is a valid {code_type} code.")
        if parsed.show_record:
            record = validator.get(parsed.code, code_type) or {}
            print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    print(f"✗ {parsed.code} is not valid for {code_type}.")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

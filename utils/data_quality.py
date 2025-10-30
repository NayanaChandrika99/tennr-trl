"""Utility for performing lightweight quality checks on prepared datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from utils.code_validator import CodeValidator


@dataclass(slots=True)
class DataQualityIssue:
    """Represents a single data quality problem."""

    index: int
    message: str
    payload: Optional[Dict[str, object]] = None


@dataclass(slots=True)
class DataQualityReport:
    """Aggregated results from a dataset validation pass."""

    total_records: int = 0
    invalid_codes: List[DataQualityIssue] = field(default_factory=list)
    missing_reasoning: List[DataQualityIssue] = field(default_factory=list)
    malformed_messages: List[DataQualityIssue] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return not (self.invalid_codes or self.missing_reasoning or self.malformed_messages)

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_records": self.total_records,
            "invalid_codes": [issue.__dict__ for issue in self.invalid_codes],
            "missing_reasoning": [issue.__dict__ for issue in self.missing_reasoning],
            "malformed_messages": [issue.__dict__ for issue in self.malformed_messages],
            "status": "ok" if self.is_healthy else "issues_found",
        }


def load_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def validate_dataset(
    dataset_path: Path,
    stage: str,
    validator: CodeValidator,
) -> DataQualityReport:
    report = DataQualityReport()
    reasoning_required = stage.lower() in {"stage_2", "stage_3"}

    for index, record in enumerate(load_jsonl(dataset_path)):
        report.total_records += 1
        metadata = record.get("_metadata", {})
        messages = record.get("messages")
        code_type = str(metadata.get("code_type", ""))
        code = str(metadata.get("code", ""))

        if messages is None and stage.lower() == "stage_3":
            # Stage 3 uses preference format
            prompt = record.get("prompt")
            if not isinstance(prompt, list):
                report.malformed_messages.append(
                    DataQualityIssue(index=index, message="Preference prompt missing list format.", payload=record),
                )
            continue

        if not isinstance(messages, list) or len(messages) < 3:
            report.malformed_messages.append(
                DataQualityIssue(
                    index=index,
                    message="Message structure invalid or too short.",
                    payload={"messages": messages},
                ),
            )
        if code_type and code and not validator.validate(code, code_type):
            report.invalid_codes.append(
                DataQualityIssue(
                    index=index,
                    message=f"Code {code} not found in database for type {code_type}.",
                    payload={"code": code, "code_type": code_type},
                ),
            )
        if reasoning_required and stage.lower() != "stage_3":
            assistant_message = messages[-1] if isinstance(messages, list) else {}
            assistant_content = str(assistant_message.get("content", ""))
            if "<think>" not in assistant_content or "</think>" not in assistant_content:
                report.missing_reasoning.append(
                    DataQualityIssue(
                        index=index,
                        message="Stage 2 record missing <think> reasoning block.",
                    ),
                )

    return report


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run quality checks on dataset files.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset jsonl file.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["stage_1", "stage_2", "stage_3"],
        help="Training stage associated with the dataset.",
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=Path("data/code_databases"),
        help="Directory containing code databases.",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser


def main(args: Optional[argparse.Namespace] = None) -> int:
    parser = build_cli()
    parsed = args or parser.parse_args()

    validator = CodeValidator(parsed.database_dir)
    report = validate_dataset(parsed.dataset, parsed.stage, validator)

    if parsed.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"Records checked: {report.total_records}")
        print(f"Invalid codes: {len(report.invalid_codes)}")
        print(f"Missing reasoning: {len(report.missing_reasoning)}")
        print(f"Malformed messages: {len(report.malformed_messages)}")

        if not report.is_healthy:
            if report.invalid_codes:
                print("\nInvalid codes:")
                for issue in report.invalid_codes[:5]:
                    print(f" - #{issue.index}: {issue.message}")
            if report.missing_reasoning:
                print("\nMissing reasoning:")
                for issue in report.missing_reasoning[:5]:
                    print(f" - #{issue.index}: {issue.message}")
            if report.malformed_messages:
                print("\nMalformed messages:")
                for issue in report.malformed_messages[:5]:
                    print(f" - #{issue.index}: {issue.message}")

    return 0 if report.is_healthy else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

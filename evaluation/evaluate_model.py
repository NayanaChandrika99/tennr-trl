"""Lightweight evaluation runner for THCE."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from evaluation.baseline import BaselinePredictor
from evaluation.metrics import (
    MetricResult,
    available_metrics,
    compute_accuracy,
    compute_average_reasoning_length,
    compute_code_validity,
    compute_reasoning_coverage,
)
from utils.code_validator import CodeValidator


CODE_PATTERN = re.compile(r"(ICD10|CPT|HCPCS)\s*:\s*([A-Z0-9.\-]+)")


def load_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if limit is not None and index >= limit:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_code(response: str, fallback: str = "") -> str:
    match = CODE_PATTERN.search(response or "")
    if match:
        return match.group(2)
    return fallback.strip().upper()


def build_predictions_from_file(path: Path) -> List[str]:
    predictions: List[str] = []
    records = load_jsonl(path)
    for record in records:
        if isinstance(record, str):
            predictions.append(record)
        elif isinstance(record, dict):
            if "prediction" in record:
                predictions.append(str(record["prediction"]))
            elif "content" in record:
                predictions.append(str(record["content"]))
            else:
                predictions.append(json.dumps(record))
        else:
            predictions.append(str(record))
    return predictions


def compute_metric_results(
    metric_names: Sequence[str],
    expected_codes: Sequence[str],
    predicted_codes: Sequence[str],
    responses: Sequence[str],
    validator: CodeValidator,
) -> List[MetricResult]:
    results: List[MetricResult] = []
    for metric in metric_names:
        if metric == "accuracy":
            value = compute_accuracy(expected_codes, predicted_codes)
        elif metric == "code_validity":
            value = compute_code_validity(predicted_codes, validator)
        elif metric == "reasoning_coverage":
            value = compute_reasoning_coverage(responses)
        elif metric == "average_reasoning_length":
            value = compute_average_reasoning_length(responses)
        else:
            raise ValueError(f"Unsupported metric '{metric}'.")
        results.append(MetricResult(metric, value))
    return results


def evaluate(
    dataset: List[Dict[str, object]],
    predictions: List[str],
    metric_names: Sequence[str],
    validator: CodeValidator,
) -> Dict[str, object]:
    expected_codes = [
        str(record.get("_metadata", {}).get("code", "")).strip().upper() for record in dataset
    ]
    predicted_codes = [
        extract_code(response, fallback=str(record.get("_metadata", {}).get("code", "")))
        for response, record in zip(predictions, dataset)
    ]

    metric_results = compute_metric_results(
        metric_names=metric_names,
        expected_codes=expected_codes,
        predicted_codes=predicted_codes,
        responses=predictions,
        validator=validator,
    )

    failures: List[Dict[str, object]] = []
    for record, pred_code, response in zip(dataset, predicted_codes, predictions):
        metadata = record.get("_metadata", {}) or {}
        expected_code = str(metadata.get("code", "")).strip().upper()
        if expected_code and pred_code != expected_code:
            failures.append(
                {
                    "narrative": record.get("messages", [{"content": ""}])[1]["content"]
                    if "messages" in record
                    else record.get("prompt"),
                    "expected": expected_code,
                    "predicted": pred_code,
                    "response": response,
                }
            )

    return {
        "metrics": {result.name: result.value for result in metric_results},
        "failures": failures[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate THCE model outputs.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset jsonl file.")
    parser.add_argument(
        "--baseline",
        default="echo_ground_truth",
        choices=["echo_ground_truth", "random_valid_code", "majority_code"],
        help="Baseline strategy to generate predictions when --predictions is not provided.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional jsonl file containing predictions. Each line should have a 'prediction' field.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "code_validity", "reasoning_coverage"],
        choices=available_metrics(),
        help="Metrics to compute.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Optional path to save the evaluation report as JSON.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit evaluation to the first N samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for baseline strategies that use randomness.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_jsonl(args.dataset, limit=args.max_samples)
    if not dataset:
        raise ValueError("Dataset is empty or could not be loaded.")

    validator = CodeValidator(Path("data/code_databases"))

    if args.predictions:
        predictions = build_predictions_from_file(args.predictions)
    else:
        predictor = BaselinePredictor(validator, strategy=args.baseline, seed=args.seed)
        predictions = predictor.predict(dataset)

    if len(predictions) != len(dataset):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) does not match dataset length ({len(dataset)}).",
        )

    report = evaluate(dataset, predictions, args.metrics, validator)
    summary = {
        "dataset": str(args.dataset),
        "num_samples": len(dataset),
        "strategy": args.baseline if not args.predictions else "custom_predictions",
        **report,
    }

    for name, value in summary["metrics"].items():
        print(f"{name}: {value:.3f}")

    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        with args.report_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        print(f"Saved report to {args.report_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

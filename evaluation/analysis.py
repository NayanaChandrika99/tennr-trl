"""Report analysis helper for THCE evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect THCE evaluation reports.")
    parser.add_argument("--report-path", type=Path, required=True, help="Path to JSON report.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of failure examples to print.",
    )
    return parser.parse_args()


def load_report(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def print_summary(report: Dict[str, object], top_k: int) -> None:
    print(f"Dataset: {report.get('dataset')}")
    print(f"Samples: {report.get('num_samples')}")
    strategy = report.get("strategy")
    if strategy:
        print(f"Strategy: {strategy}")

    metrics = report.get("metrics", {})
    if metrics:
        print("\nMetrics:")
        for name, value in metrics.items():
            print(f"  {name:<24} {value:.3f}")

    failures: List[Dict[str, object]] = report.get("failures", [])
    if not failures:
        print("\nNo failure cases recorded.")
        return

    print(f"\nTop {min(top_k, len(failures))} failures:")
    for idx, failure in enumerate(failures[:top_k], start=1):
        print(f"\n[{idx}] Expected: {failure.get('expected')} | Predicted: {failure.get('predicted')}")
        narrative = failure.get("narrative")
        if isinstance(narrative, list):
            print("Prompt:")
            for message in narrative:
                role = message.get("role")
                content = message.get("content")
                print(f"  {role}: {content}")
        else:
            print(f"Narrative: {narrative}")
        response = failure.get("response")
        if response:
            print("Response:")
            print(response)


def main() -> int:
    args = parse_args()
    report = load_report(args.report_path)
    print_summary(report, args.top_k)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

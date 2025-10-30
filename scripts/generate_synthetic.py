"""
Batch generator for THCE synthetic examples using hosted LLM APIs.

This script is intentionally lightweight and avoids direct API dependencies so that
teams can plug in their preferred provider (OpenAI, Anthropic, Azure, etc.). It reads
prompt seeds from `data/prompts/thce_synthetic_prompt.txt`, dispatches requests, and
writes JSONL outputs compatible with `data/preprocess_medical.py`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Sequence

import dotenv

DEFAULT_PROMPT_PATH = Path("data/prompts/thce_synthetic_prompt.txt")


def load_prompt_template(path: Path) -> str:
    if not path.exists():
        msg = f"Prompt template not found at {path}."
        raise FileNotFoundError(msg)
    return path.read_text(encoding="utf-8")


def batched(iterable: Sequence[str], batch_size: int) -> Iterator[Sequence[str]]:
    for index in range(0, len(iterable), batch_size):
        yield iterable[index : index + batch_size]


def write_records(records: Iterable[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def default_request_fn(prompt: str) -> str:
    """Placeholder request function. Replace with real API call."""
    raise NotImplementedError(
        "Plug in a provider-specific request function (OpenAI, Anthropic, etc.).",
    )


def generate_examples(
    prompt_template: str,
    topics: Sequence[str],
    request_fn: Callable[[str], str],
    throttle_seconds: float,
) -> List[dict]:
    """Dispatch prompts and parse responses."""
    outputs: List[dict] = []
    for topic in topics:
        prompt = f"{prompt_template}\n\nFocus specialty: {topic}"
        raw = request_fn(prompt)
        try:
            record = json.loads(raw)
            outputs.append(record)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Response was not valid JSON:\n{raw}") from exc
        time.sleep(throttle_seconds)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic THCE samples via LLM API.")
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument(
        "--topics",
        nargs="+",
        default=[
            "cardiology",
            "endocrinology",
            "pulmonology",
            "orthopedics",
            "oncology",
            "neurology",
            "rehabilitation",
            "durable medical equipment",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Prompts per batch.")
    parser.add_argument("--throttle", type=float, default=1.0, help="Seconds between calls.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL path.")
    return parser.parse_args()


def main() -> int:
    dotenv.load_dotenv()
    args = parse_args()
    prompt_template = load_prompt_template(args.prompt_path)

    # TODO: replace default_request_fn with an actual API client wrapper. For instance:
    # from scripts.providers import anthropic_request
    # request_fn = anthropic_request
    request_fn = default_request_fn

    collected = []
    topics = args.topics
    for group in batched(topics, args.batch_size):
        collected.extend(
            generate_examples(prompt_template=prompt_template, topics=group, request_fn=request_fn, throttle_seconds=args.throttle),
        )

    write_records(collected, args.output)
    print(f"Wrote {len(collected)} synthetic records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

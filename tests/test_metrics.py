from evaluation.metrics import (
    compute_accuracy,
    compute_average_reasoning_length,
    compute_reasoning_coverage,
    extract_reasoning_blocks,
)


def test_compute_accuracy() -> None:
    gold = ["E11.9", "I10", "A4253"]
    predicted = ["E11.9", "J45.909", "A4253"]
    assert compute_accuracy(gold, predicted) == 2 / 3


def test_reasoning_metrics() -> None:
    responses = [
        "<think>\nstep 1\nstep 2\n</think>\nICD10: E11.9",
        "ICD10: I10",
    ]
    assert compute_reasoning_coverage(responses) == 0.5
    assert compute_average_reasoning_length(responses) == 2.0


def test_extract_reasoning_blocks_handles_missing_tags() -> None:
    blocks = extract_reasoning_blocks(["No reasoning", "<think>Only one tag"])
    assert blocks == ["", ""]

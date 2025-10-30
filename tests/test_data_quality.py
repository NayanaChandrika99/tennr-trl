from pathlib import Path

from utils.code_validator import CodeValidator
from utils.data_quality import validate_dataset


def test_stage_1_dataset_is_healthy() -> None:
    dataset = Path("data/examples/sample_stage_1.jsonl")
    validator = CodeValidator(Path("data/code_databases"))
    report = validate_dataset(dataset, "stage_1", validator)
    assert report.is_healthy


def test_stage_2_dataset_is_healthy() -> None:
    dataset = Path("data/examples/sample_stage_2.jsonl")
    validator = CodeValidator(Path("data/code_databases"))
    report = validate_dataset(dataset, "stage_2", validator)
    assert report.is_healthy


def test_detects_missing_reasoning(tmp_path) -> None:
    dataset_path = tmp_path / "stage_2_invalid.jsonl"
    dataset_path.write_text(
        '{"messages": [{"role": "system", "content": "X"}, {"role": "user", "content": "Y"}, {"role": "assistant", "content": "ICD10: I10"}], "_metadata": {"code_type": "ICD10", "code": "I10"}}\n',
        encoding="utf-8",
    )
    validator = CodeValidator(Path("data/code_databases"))
    report = validate_dataset(dataset_path, "stage_2", validator)
    assert not report.is_healthy
    assert report.missing_reasoning

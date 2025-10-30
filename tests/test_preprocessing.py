from pathlib import Path

from data.preprocess_medical import convert_records, load_raw_records
from utils.code_validator import CodeValidator
from utils.medical_chat_templates import Stage


def test_convert_records_stage_1() -> None:
    raw_path = Path("tests/fixtures/raw_records.jsonl")
    records = load_raw_records(raw_path)
    validator = CodeValidator(Path("data/code_databases"))
    converted = convert_records(records, Stage.STAGE_1, validator)
    assert len(converted) == 2
    first = converted[0]
    assert first["_metadata"]["reasoning_present"] is False
    assert first["messages"][-1]["content"].startswith("ICD10:")


def test_convert_records_stage_3_includes_preference_structure() -> None:
    raw_path = Path("tests/fixtures/raw_records.jsonl")
    records = load_raw_records(raw_path)
    validator = CodeValidator(Path("data/code_databases"))
    converted = convert_records(records, Stage.STAGE_3, validator)
    assert len(converted) == 2
    item = converted[0]
    assert "prompt" in item and "chosen" in item and "rejected" in item

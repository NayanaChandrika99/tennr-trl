from pathlib import Path

import pytest

from utils.code_validator import CodeValidator


@pytest.fixture(scope="module")
def validator() -> CodeValidator:
    return CodeValidator(Path("data/code_databases"))


def test_validate_known_icd10_code(validator: CodeValidator) -> None:
    assert validator.validate("E11.9", "ICD10")


def test_validate_known_cpt_code(validator: CodeValidator) -> None:
    assert validator.validate("99213", "CPT")


def test_infer_code_type() -> None:
    assert CodeValidator.infer_code_type("J20.9") == "ICD10"
    assert CodeValidator.infer_code_type("90686") == "CPT"
    assert CodeValidator.infer_code_type("E0130") == "HCPCS"


def test_validate_unknown_code(validator: CodeValidator) -> None:
    assert not validator.validate("E99.999", "ICD10")

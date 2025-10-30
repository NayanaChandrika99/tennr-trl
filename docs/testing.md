# THCE Testing Guide
**Test System, Fixtures, and Testing Patterns**

---

## Table of Contents
1. [Testing Philosophy](#testing-philosophy)
2. [Test Structure](#test-structure)
3. [Unit Tests](#unit-tests)
4. [Integration Tests](#integration-tests)
5. [Fixtures and Test Data](#fixtures-and-test-data)
6. [Running Tests](#running-tests)
7. [Test Coverage](#test-coverage)
8. [Testing Patterns](#testing-patterns)

---

## Testing Philosophy

### Testing Pyramid

```
         /\
        /  \
       / E2E \          Few (End-to-end tests)
      /______\
     /        \
    /Integration\       Some (Integration tests)
   /____________\
  /              \
 /   Unit Tests   \     Many (Unit tests)
/__________________\
```

### Testing Goals

1. **Correctness**: Ensure code works as intended
2. **Confidence**: Catch bugs before they reach production
3. **Documentation**: Tests serve as usage examples
4. **Refactoring Safety**: Change code without fear
5. **Fast Feedback**: Quick test execution

### Test Coverage Targets

```
Overall:         80%+
Core utilities:  90%+
Data pipeline:   85%+
Training:        70%+
Evaluation:      85%+
```

---

## Test Structure

### Directory Layout

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── fixtures/                      # Test data
│   ├── valid_codes.json          # Valid ICD/CPT/HCPCS codes
│   ├── invalid_codes.json        # Invalid codes for testing
│   ├── sample_narratives.json    # Sample medical narratives
│   ├── sample_stage_1.jsonl      # Stage 1 format examples
│   ├── sample_stage_2.jsonl      # Stage 2 format examples
│   └── sample_stage_3.jsonl      # Stage 3 format examples
│
├── unit/                          # Unit tests
│   ├── test_code_validator.py    # Code validation tests
│   ├── test_data_quality.py      # Data quality tests
│   ├── test_chat_templates.py    # Chat template tests
│   ├── test_tokenization.py      # Tokenization tests
│   └── test_metrics.py           # Metrics calculation tests
│
├── integration/                   # Integration tests
│   ├── test_data_pipeline.py     # Full data pipeline
│   ├── test_training_setup.py    # Training setup tests
│   └── test_evaluation.py        # Evaluation pipeline tests
│
└── e2e/                          # End-to-end tests
    ├── test_full_workflow.py     # Complete workflow tests
    └── test_minimal_training.py  # Minimal training tests
```

---

## Unit Tests

### Testing Code Validator

**File:** `tests/unit/test_code_validator.py`

```python
import pytest
from utils.code_validator import CodeValidator

@pytest.fixture
def validator():
    """Create a code validator instance."""
    return CodeValidator()

class TestCodeValidator:
    """Test suite for code validation."""

    def test_validate_valid_icd10_code(self, validator):
        """Test validation of a valid ICD-10 code."""
        result = validator.validate("E11.9", code_type="ICD10")
        assert result.is_valid is True
        assert result.code == "E11.9"
        assert result.description is not None

    def test_validate_invalid_icd10_code(self, validator):
        """Test validation of an invalid ICD-10 code."""
        result = validator.validate("INVALID", code_type="ICD10")
        assert result.is_valid is False
        assert result.error_message is not None

    def test_validate_cpt_code(self, validator):
        """Test validation of CPT codes."""
        result = validator.validate("99213", code_type="CPT")
        assert result.is_valid is True
        assert result.code_type == "CPT"

    def test_validate_hcpcs_code(self, validator):
        """Test validation of HCPCS codes."""
        result = validator.validate("E0130", code_type="HCPCS")
        assert result.is_valid is True
        assert result.code_type == "HCPCS"

    def test_validate_empty_code(self, validator):
        """Test validation of empty code."""
        with pytest.raises(ValueError):
            validator.validate("", code_type="ICD10")

    def test_validate_invalid_code_type(self, validator):
        """Test validation with invalid code type."""
        with pytest.raises(ValueError):
            validator.validate("E11.9", code_type="INVALID_TYPE")

    @pytest.mark.parametrize("code,expected", [
        ("E11.9", True),
        ("I10", True),
        ("J20.9", True),
        ("INVALID", False),
        ("", False),
    ])
    def test_validate_multiple_icd10_codes(self, validator, code, expected):
        """Test validation of multiple ICD-10 codes."""
        if code == "":
            with pytest.raises(ValueError):
                validator.validate(code, code_type="ICD10")
        else:
            result = validator.validate(code, code_type="ICD10")
            assert result.is_valid == expected
```

### Testing Data Quality

**File:** `tests/unit/test_data_quality.py`

```python
import pytest
from utils.data_quality import DataQualityChecker

@pytest.fixture
def quality_checker():
    """Create a data quality checker instance."""
    return DataQualityChecker()

@pytest.fixture
def valid_stage_1_example():
    """Valid Stage 1 example."""
    return {
        "messages": [
            {"role": "system", "content": "You are a medical coding assistant."},
            {"role": "user", "content": "Patient has diabetes."},
            {"role": "assistant", "content": "ICD10: E11.9"}
        ]
    }

@pytest.fixture
def valid_stage_2_example():
    """Valid Stage 2 example with reasoning."""
    return {
        "messages": [
            {"role": "system", "content": "You are a medical coding expert."},
            {"role": "user", "content": "Patient has diabetes."},
            {"role": "assistant", "content": "<think>Type 2 diabetes, no complications</think>ICD10: E11.9"}
        ]
    }

class TestDataQualityChecker:
    """Test suite for data quality checking."""

    def test_check_valid_stage_1_example(self, quality_checker, valid_stage_1_example):
        """Test quality check on valid Stage 1 example."""
        result = quality_checker.check_example(valid_stage_1_example, stage=1)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_check_valid_stage_2_example(self, quality_checker, valid_stage_2_example):
        """Test quality check on valid Stage 2 example."""
        result = quality_checker.check_example(valid_stage_2_example, stage=2)
        assert result.is_valid is True
        assert result.has_reasoning is True

    def test_check_missing_messages_field(self, quality_checker):
        """Test quality check on example missing messages."""
        invalid_example = {"data": "test"}
        result = quality_checker.check_example(invalid_example, stage=1)
        assert result.is_valid is False
        assert "missing 'messages'" in result.errors[0].lower()

    def test_check_empty_content(self, quality_checker):
        """Test quality check on example with empty content."""
        invalid_example = {
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "ICD10: E11.9"}
            ]
        }
        result = quality_checker.check_example(invalid_example, stage=1)
        assert result.is_valid is False

    def test_check_missing_reasoning_stage_2(self, quality_checker):
        """Test Stage 2 example missing reasoning tags."""
        invalid_example = {
            "messages": [
                {"role": "user", "content": "Patient has diabetes."},
                {"role": "assistant", "content": "ICD10: E11.9"}  # No <think> tags
            ]
        }
        result = quality_checker.check_example(invalid_example, stage=2)
        assert result.has_reasoning is False
```

### Testing Chat Templates

**File:** `tests/unit/test_chat_templates.py`

```python
import pytest
from utils.medical_chat_templates import (
    get_stage_1_template,
    get_stage_2_template,
    format_medical_coding_prompt
)

class TestChatTemplates:
    """Test suite for chat templates."""

    def test_stage_1_template_format(self):
        """Test Stage 1 template formatting."""
        template = get_stage_1_template()
        assert "<|im_start|>" in template
        assert "<|im_end|>" in template
        assert "medical coding assistant" in template.lower()

    def test_stage_2_template_format(self):
        """Test Stage 2 template formatting."""
        template = get_stage_2_template()
        assert "<think>" in template or "reasoning" in template.lower()

    def test_format_medical_coding_prompt(self):
        """Test formatting a medical coding prompt."""
        narrative = "Patient has diabetes."
        code = "E11.9"

        prompt = format_medical_coding_prompt(
            narrative=narrative,
            code=code,
            stage=1
        )

        assert narrative in prompt
        assert code in prompt

    def test_format_with_reasoning(self):
        """Test formatting with reasoning."""
        narrative = "Patient has diabetes."
        code = "E11.9"
        reasoning = "Type 2 diabetes without complications"

        prompt = format_medical_coding_prompt(
            narrative=narrative,
            code=code,
            reasoning=reasoning,
            stage=2
        )

        assert "<think>" in prompt
        assert reasoning in prompt
        assert code in prompt
```

### Testing Metrics

**File:** `tests/unit/test_metrics.py`

```python
import pytest
from evaluation.metrics import (
    calculate_exact_match,
    calculate_top_k_accuracy,
    calculate_precision_recall_f1,
    calculate_code_type_breakdown
)

@pytest.fixture
def predictions():
    """Sample predictions."""
    return [
        {"predicted_code": "E11.9", "confidence": 0.95},
        {"predicted_code": "I10", "confidence": 0.88},
        {"predicted_code": "J20.9", "confidence": 0.92},
    ]

@pytest.fixture
def ground_truth():
    """Sample ground truth."""
    return [
        {"code": "E11.9", "code_type": "ICD10"},
        {"code": "I10", "code_type": "ICD10"},
        {"code": "J20.8", "code_type": "ICD10"},  # Different from prediction
    ]

class TestMetrics:
    """Test suite for metrics calculation."""

    def test_exact_match_all_correct(self):
        """Test exact match with all correct predictions."""
        predictions = ["E11.9", "I10", "J20.9"]
        ground_truth = ["E11.9", "I10", "J20.9"]

        accuracy = calculate_exact_match(predictions, ground_truth)
        assert accuracy == 1.0

    def test_exact_match_partial_correct(self):
        """Test exact match with partial correct predictions."""
        predictions = ["E11.9", "I10", "J20.9"]
        ground_truth = ["E11.9", "I10", "J20.8"]

        accuracy = calculate_exact_match(predictions, ground_truth)
        assert accuracy == pytest.approx(0.667, abs=0.01)

    def test_top_k_accuracy(self):
        """Test top-k accuracy calculation."""
        predictions = [
            ["E11.9", "E11.8", "E11.65"],  # Top 3 predictions
            ["I10", "I11", "I12"],
            ["J20.9", "J20.8", "J20.0"],
        ]
        ground_truth = ["E11.9", "I10", "J20.8"]

        top_1 = calculate_top_k_accuracy(predictions, ground_truth, k=1)
        top_3 = calculate_top_k_accuracy(predictions, ground_truth, k=3)

        assert top_1 == pytest.approx(0.667, abs=0.01)
        assert top_3 == 1.0

    def test_precision_recall_f1(self):
        """Test precision, recall, and F1 score calculation."""
        predictions = ["E11.9", "I10", "J20.9"]
        ground_truth = ["E11.9", "I10", "J20.8"]

        precision, recall, f1 = calculate_precision_recall_f1(
            predictions, ground_truth
        )

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1

    def test_code_type_breakdown(self):
        """Test code type breakdown calculation."""
        predictions = [
            {"code": "E11.9", "type": "ICD10"},
            {"code": "99213", "type": "CPT"},
            {"code": "E0130", "type": "HCPCS"},
        ]
        ground_truth = [
            {"code": "E11.9", "type": "ICD10"},
            {"code": "99213", "type": "CPT"},
            {"code": "E0130", "type": "HCPCS"},
        ]

        breakdown = calculate_code_type_breakdown(predictions, ground_truth)

        assert breakdown["ICD10"]["accuracy"] == 1.0
        assert breakdown["CPT"]["accuracy"] == 1.0
        assert breakdown["HCPCS"]["accuracy"] == 1.0
```

---

## Integration Tests

### Testing Data Pipeline

**File:** `tests/integration/test_data_pipeline.py`

```python
import pytest
import tempfile
import shutil
from pathlib import Path
from data.data_collection import save_dataset_streaming

@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def minimal_config():
    """Minimal data collection config for testing."""
    return {
        "name": "test-dataset",
        "description": "Test dataset",
        "datasets": [
            {
                "name": "test_source",
                "subset": None,
                "split": "train",
                "entries": 10
            }
        ]
    }

class TestDataPipeline:
    """Integration tests for data pipeline."""

    def test_data_collection_creates_output(self, temp_output_dir, minimal_config):
        """Test that data collection creates output files."""
        save_dataset_streaming(
            config=minimal_config,
            output_dir=temp_output_dir,
            upload_to_hub=False
        )

        output_path = Path(temp_output_dir)
        assert output_path.exists()
        assert len(list(output_path.glob("*.parquet"))) > 0

    def test_data_collection_metadata(self, temp_output_dir, minimal_config):
        """Test that metadata is created."""
        save_dataset_streaming(
            config=minimal_config,
            output_dir=temp_output_dir,
            upload_to_hub=False
        )

        metadata_path = Path(temp_output_dir) / "dataset_metadata.json"
        assert metadata_path.exists()

    def test_data_validation_pipeline(self, temp_output_dir):
        """Test complete data validation pipeline."""
        # This would test the full flow:
        # Raw data → Validation → Preprocessing → Final dataset
        pass
```

### Testing Training Setup

**File:** `tests/integration/test_training_setup.py`

```python
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from post_training.sft import load_model_and_tokenizer

class TestTrainingSetup:
    """Integration tests for training setup."""

    @pytest.mark.slow
    def test_load_base_model(self):
        """Test loading base model."""
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct"
        )
        assert model is not None
        assert model.config.vocab_size > 0

    @pytest.mark.slow
    def test_load_tokenizer(self):
        """Test loading tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct"
        )
        assert tokenizer is not None

    def test_add_special_tokens(self):
        """Test adding special tokens to tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct"
        )

        original_vocab_size = len(tokenizer)

        tokenizer.add_special_tokens({
            "additional_special_tokens": ["<think>", "</think>"]
        })

        new_vocab_size = len(tokenizer)
        assert new_vocab_size > original_vocab_size

    def test_chat_template_application(self):
        """Test applying chat template."""
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct"
        )

        messages = [
            {"role": "system", "content": "You are a medical coding assistant."},
            {"role": "user", "content": "Patient has diabetes."},
            {"role": "assistant", "content": "ICD10: E11.9"}
        ]

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        assert formatted is not None
        assert "diabetes" in formatted.lower()
```

---

## Fixtures and Test Data

### Shared Fixtures

**File:** `tests/conftest.py`

```python
import pytest
import json
from pathlib import Path

@pytest.fixture
def fixtures_dir():
    """Path to fixtures directory."""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def valid_codes(fixtures_dir):
    """Load valid codes fixture."""
    with open(fixtures_dir / "valid_codes.json") as f:
        return json.load(f)

@pytest.fixture
def invalid_codes(fixtures_dir):
    """Load invalid codes fixture."""
    with open(fixtures_dir / "invalid_codes.json") as f:
        return json.load(f)

@pytest.fixture
def sample_narratives(fixtures_dir):
    """Load sample medical narratives."""
    with open(fixtures_dir / "sample_narratives.json") as f:
        return json.load(f)

@pytest.fixture
def stage_1_examples(fixtures_dir):
    """Load Stage 1 format examples."""
    examples = []
    with open(fixtures_dir / "sample_stage_1.jsonl") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

@pytest.fixture
def stage_2_examples(fixtures_dir):
    """Load Stage 2 format examples."""
    examples = []
    with open(fixtures_dir / "sample_stage_2.jsonl") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

@pytest.fixture(scope="session")
def code_validator():
    """Create a code validator instance (session scope)."""
    from utils.code_validator import CodeValidator
    return CodeValidator()
```

### Fixture Data Files

**File:** `tests/fixtures/valid_codes.json`

```json
{
  "ICD10": [
    {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications"},
    {"code": "I10", "description": "Essential (primary) hypertension"},
    {"code": "J20.9", "description": "Acute bronchitis, unspecified"}
  ],
  "CPT": [
    {"code": "99213", "description": "Office visit, established patient"},
    {"code": "99214", "description": "Office visit, established patient"},
    {"code": "80053", "description": "Comprehensive metabolic panel"}
  ],
  "HCPCS": [
    {"code": "E0130", "description": "Walker, rigid, adjustable or fixed height"},
    {"code": "E0143", "description": "Walker, folding, wheeled, adjustable"}
  ]
}
```

**File:** `tests/fixtures/sample_stage_1.jsonl`

```jsonl
{"messages": [{"role": "system", "content": "You are a medical coding assistant."}, {"role": "user", "content": "Patient diagnosed with Type 2 diabetes mellitus without complications."}, {"role": "assistant", "content": "ICD10: E11.9 (Type 2 diabetes mellitus without complications)"}]}
{"messages": [{"role": "system", "content": "You are a medical coding assistant."}, {"role": "user", "content": "Patient presents with essential hypertension."}, {"role": "assistant", "content": "ICD10: I10 (Essential (primary) hypertension)"}]}
```

---

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_code_validator.py

# Run specific test function
pytest tests/unit/test_code_validator.py::test_validate_valid_icd10_code

# Run specific test class
pytest tests/unit/test_code_validator.py::TestCodeValidator
```

### Test Selection

```bash
# Run tests matching a pattern
pytest -k "validator"

# Run tests by marker
pytest -m "unit"
pytest -m "integration"
pytest -m "slow"

# Skip slow tests
pytest -m "not slow"

# Run only failed tests from last run
pytest --lf

# Run failed tests first
pytest --ff
```

### Parallel Execution

```bash
# Run tests in parallel (4 workers)
pytest -n 4

# Run tests in parallel (auto-detect CPU count)
pytest -n auto
```

### Output Control

```bash
# Show print statements
pytest -s

# Show local variables on failure
pytest -l

# Stop on first failure
pytest -x

# Stop after 3 failures
pytest --maxfail=3

# Quiet mode (less verbose)
pytest -q
```

---

## Test Coverage

### Measuring Coverage

```bash
# Run tests with coverage
pytest --cov=.

# Generate HTML coverage report
pytest --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Coverage Configuration

**File:** `.coveragerc`

```ini
[run]
source = .
omit =
    .venv/*
    tests/*
    setup.py
    */__pycache__/*
    */site-packages/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
```

### Coverage Targets

```bash
# Fail if coverage below 80%
pytest --cov=. --cov-fail-under=80

# Show missing lines
pytest --cov=. --cov-report=term-missing
```

---

## Testing Patterns

### Pattern 1: Parametrized Tests

```python
@pytest.mark.parametrize("code,code_type,expected", [
    ("E11.9", "ICD10", True),
    ("99213", "CPT", True),
    ("E0130", "HCPCS", True),
    ("INVALID", "ICD10", False),
])
def test_validate_codes(validator, code, code_type, expected):
    """Test validation with multiple inputs."""
    result = validator.validate(code, code_type=code_type)
    assert result.is_valid == expected
```

### Pattern 2: Fixture Factories

```python
@pytest.fixture
def make_medical_example():
    """Factory fixture for creating medical examples."""
    def _make(narrative, code, code_type="ICD10", stage=1):
        example = {
            "messages": [
                {"role": "system", "content": "You are a medical coding assistant."},
                {"role": "user", "content": narrative},
                {"role": "assistant", "content": f"{code_type}: {code}"}
            ]
        }
        return example
    return _make

def test_with_factory(make_medical_example):
    """Test using fixture factory."""
    example = make_medical_example(
        narrative="Patient has diabetes",
        code="E11.9"
    )
    assert "diabetes" in example["messages"][1]["content"]
```

### Pattern 3: Mocking External Services

```python
from unittest.mock import patch, MagicMock

@patch('openai.ChatCompletion.create')
def test_synthetic_generation(mock_openai):
    """Test synthetic data generation with mocked API."""
    mock_openai.return_value = MagicMock(
        choices=[MagicMock(message={"content": "ICD10: E11.9"})]
    )

    result = generate_synthetic_example("Patient has diabetes")
    assert "E11.9" in result
    mock_openai.assert_called_once()
```

### Pattern 4: Temporary Files

```python
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_save_dataset(temp_dir):
    """Test saving dataset to temporary directory."""
    dataset = create_test_dataset()
    save_path = Path(temp_dir) / "dataset.parquet"
    dataset.to_parquet(save_path)
    assert save_path.exists()
```

### Pattern 5: Context Managers

```python
@pytest.fixture
def mock_environment():
    """Mock environment variables."""
    import os
    original_env = os.environ.copy()
    os.environ["TEST_MODE"] = "true"
    yield
    os.environ.clear()
    os.environ.update(original_env)

def test_with_environment(mock_environment):
    """Test with mocked environment."""
    import os
    assert os.getenv("TEST_MODE") == "true"
```

---

## Continuous Integration

### GitHub Actions Workflow

**File:** `.github/workflows/tests.yaml`

```yaml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12"]

    steps:
      - uses: actions/checkout@v3

      - name: Install UV
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          uv sync --extra cpu

      - name: Run linter
        run: |
          source .venv/bin/activate
          ruff check .

      - name: Run tests
        run: |
          source .venv/bin/activate
          pytest tests/ -v --cov=. --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

---

**Last Updated:** 2025-10-30
**Next:** See [spec.md](spec.md) for feature specifications

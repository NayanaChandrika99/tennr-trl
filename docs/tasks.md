# THCE Implementation Tasks
**Detailed Task Breakdown and Tracking**

---

## Table of Contents
1. [Task Overview](#task-overview)
2. [Week 1-2: Core Utilities](#week-1-2-core-utilities)
3. [Week 3: Data Pipeline](#week-3-data-pipeline)
4. [Week 4: Training Configuration](#week-4-training-configuration)
5. [Week 5: Evaluation System](#week-5-evaluation-system)
6. [Week 6: Documentation & Testing](#week-6-documentation--testing)
7. [Task Dependencies](#task-dependencies)
8. [Task Priorities](#task-priorities)

---

## Task Overview

### Project Phase: Repository Engineering
**Duration:** 6 weeks
**Goal:** Complete, production-ready codebase (no training execution)

### Task Status Legend
- ⬜ Not Started
- 🟨 In Progress
- ✅ Complete
- 🔴 Blocked
- ⏸️ Deferred

### Overall Progress
```
Progress: [███░░░░░░░] 30% (18/60 tasks)

By Category:
- Core Utilities:    [██░░░░] 25% (5/20)
- Data Pipeline:     [█░░░░░] 15% (3/20)
- Training:          [█░░░░░] 10% (1/10)
- Evaluation:        [░░░░░░]  0% (0/15)
- Documentation:     [████░░] 60% (9/15)
- Testing:           [░░░░░░]  0% (0/10)
```

---

## Week 1-2: Core Utilities

### Milestone: Core Infrastructure Complete
**Deliverables:** Code validator, data quality checker, chat templates, test fixtures

---

### Task 1.1: Code Validator Implementation
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 6 hours

**File:** `utils/code_validator.py`

**Requirements:**
- [ ] Create `CodeValidator` class
- [ ] Implement ICD-10 validation
- [ ] Implement CPT validation
- [ ] Implement HCPCS validation
- [ ] Add code database loader
- [ ] Add batch validation support
- [ ] Add caching mechanism
- [ ] Write docstrings

**Subtasks:**
```python
# 1. Define data structures (1 hour)
class ValidationResult:
    is_valid: bool
    code: str
    code_type: str
    description: str | None
    category: str | None
    error_message: str | None

# 2. Load code databases (2 hours)
def load_code_database(code_type: str) -> dict:
    """Load code database from JSON file."""
    pass

# 3. Implement validators (2 hours)
def validate_icd10(code: str) -> ValidationResult:
    """Validate ICD-10 code."""
    pass

def validate_cpt(code: str) -> ValidationResult:
    """Validate CPT code."""
    pass

def validate_hcpcs(code: str) -> ValidationResult:
    """Validate HCPCS code."""
    pass

# 4. Main interface (1 hour)
def validate(code: str, code_type: str) -> ValidationResult:
    """Validate medical code."""
    pass
```

**Testing:**
- [ ] Unit tests for each code type
- [ ] Test with valid codes
- [ ] Test with invalid codes
- [ ] Test error handling

**Dependencies:** Code database files (Task 1.2)

---

### Task 1.2: Code Database Creation
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 4 hours

**Directory:** `data/code_databases/`

**Requirements:**
- [ ] Create ICD-10 database (top 500 codes)
- [ ] Create CPT database (top 200 codes)
- [ ] Create HCPCS database (top 100 codes)
- [ ] Add code descriptions
- [ ] Add code categories
- [ ] Create database loader script

**Files to Create:**
```
data/code_databases/
├── icd10_codes.json
├── cpt_codes.json
├── hcpcs_codes.json
├── load_codes.py
└── README.md
```

**Database Schema:**
```json
{
  "codes": [
    {
      "code": "E11.9",
      "description": "Type 2 diabetes mellitus without complications",
      "category": "Endocrine",
      "billable": true,
      "parent_code": "E11"
    }
  ],
  "metadata": {
    "version": "2024",
    "total_codes": 500,
    "last_updated": "2025-10-30"
  }
}
```

**Data Sources:**
- ICD-10: CMS website or public datasets
- CPT: Sample from available resources
- HCPCS: CMS HCPCS files

**Testing:**
- [ ] Validate JSON schema
- [ ] Check code uniqueness
- [ ] Verify descriptions present

---

### Task 1.3: Data Quality Checker
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 5 hours

**File:** `utils/data_quality.py`

**Requirements:**
- [ ] Create `DataQualityChecker` class
- [ ] Implement schema validation
- [ ] Check message structure
- [ ] Validate code formats
- [ ] Detect reasoning tags (Stage 2+)
- [ ] Check for empty content
- [ ] PHI detection patterns
- [ ] Generate quality report

**Interface:**
```python
class DataQualityChecker:
    def check_example(
        self,
        example: dict,
        stage: int
    ) -> QualityResult:
        """Check single example quality."""
        pass

    def check_file(
        self,
        file_path: str,
        stage: int
    ) -> QualityReport:
        """Check entire file quality."""
        pass

    def generate_report(
        self,
        results: list[QualityResult]
    ) -> str:
        """Generate quality report."""
        pass
```

**Quality Checks:**
1. Schema validation (has 'messages')
2. Message structure (role, content fields)
3. Content not empty
4. Valid code format (ICD10/CPT/HCPCS pattern)
5. Reasoning tags present (Stage 2+)
6. Narrative length (10-2000 chars)
7. No obvious PHI patterns

**Testing:**
- [ ] Test with valid examples
- [ ] Test with invalid examples
- [ ] Test all quality checks
- [ ] Test report generation

---

### Task 1.4: Medical Chat Templates
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 4 hours

**File:** `utils/medical_chat_templates.py`

**Requirements:**
- [ ] Define Stage 1 template (no reasoning)
- [ ] Define Stage 2 template (with reasoning)
- [ ] Define Stage 3 template (preferences)
- [ ] Create formatting functions
- [ ] Add template validation
- [ ] Write usage examples

**Templates to Create:**
```python
# Stage 1: No reasoning
STAGE_1_TEMPLATE = """<|im_start|>system
You are a medical coding assistant...<|im_end|>
<|im_start|>user
{narrative}<|im_end|>
<|im_start|>assistant
{code_type}: {code} ({description})<|im_end|>"""

# Stage 2: With reasoning
STAGE_2_TEMPLATE = """<|im_start|>system
You are a medical coding expert. Use <think> tags...<|im_end|>
<|im_start|>user
{narrative}<|im_end|>
<|im_start|>assistant
<think>
{reasoning}
</think>
{code_type}: {code} ({description})<|im_end|>"""

# Functions
def get_stage_1_template() -> str: ...
def get_stage_2_template() -> str: ...
def format_conversation(...) -> str: ...
```

**Testing:**
- [ ] Test template retrieval
- [ ] Test formatting functions
- [ ] Validate output format

---

### Task 1.5: Test Fixtures Creation
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 6 hours

**Directory:** `tests/fixtures/`

**Requirements:**
- [ ] Create valid codes fixture (100 codes)
- [ ] Create invalid codes fixture (50 codes)
- [ ] Create sample narratives (100 narratives)
- [ ] Create Stage 1 examples (50 examples)
- [ ] Create Stage 2 examples (50 examples)
- [ ] Create Stage 3 examples (50 examples)
- [ ] Create README documenting fixtures

**Files to Create:**
```
tests/fixtures/
├── valid_codes.json           # 100 valid codes
├── invalid_codes.json         # 50 invalid codes
├── sample_narratives.json     # 100 narratives
├── sample_stage_1.jsonl       # 50 Stage 1 examples
├── sample_stage_2.jsonl       # 50 Stage 2 examples
├── sample_stage_3.jsonl       # 50 Stage 3 examples
└── README.md                  # Documentation
```

**Example Fixture:**
```json
// valid_codes.json
{
  "ICD10": [
    {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications"},
    {"code": "I10", "description": "Essential (primary) hypertension"}
  ],
  "CPT": [
    {"code": "99213", "description": "Office visit, established patient"}
  ]
}
```

**Testing:**
- [ ] Validate all JSON/JSONL files
- [ ] Check data completeness
- [ ] Verify correct formats

---

### Task 1.6: Unit Tests for Validators
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 4 hours

**File:** `tests/unit/test_code_validator.py`

**Requirements:**
- [ ] Test ICD-10 validation
- [ ] Test CPT validation
- [ ] Test HCPCS validation
- [ ] Test invalid codes
- [ ] Test error handling
- [ ] Parametrized tests
- [ ] Achieve >90% coverage

**Test Cases:**
```python
# Basic tests
def test_validate_valid_icd10_code()
def test_validate_invalid_icd10_code()
def test_validate_cpt_code()
def test_validate_hcpcs_code()

# Edge cases
def test_validate_empty_code()
def test_validate_invalid_code_type()
def test_validate_with_invalid_format()

# Parametrized tests
@pytest.mark.parametrize("code,expected", [...])
def test_validate_multiple_codes()
```

**Run Tests:**
```bash
pytest tests/unit/test_code_validator.py -v
```

---

### Task 1.7: Unit Tests for Data Quality
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 4 hours

**File:** `tests/unit/test_data_quality.py`

**Requirements:**
- [ ] Test schema validation
- [ ] Test message structure checks
- [ ] Test content validation
- [ ] Test reasoning detection
- [ ] Test PHI detection
- [ ] Test report generation
- [ ] Achieve >85% coverage

---

### Task 1.8: Setup Validation Script
**Priority:** MEDIUM | **Status:** ⬜ Not Started | **Estimate:** 3 hours

**File:** `scripts/validate_setup.py`

**Requirements:**
- [ ] Check Python version (≥3.12)
- [ ] Check dependencies installed
- [ ] Test GPU availability
- [ ] Validate environment variables
- [ ] Check disk space
- [ ] Test HuggingFace connectivity
- [ ] Generate setup report

**Example Output:**
```
THCE Setup Validation
=====================

✓ Python 3.12.1
✓ PyTorch 2.7.0
✓ Transformers 4.56.2
✓ CUDA available: True
  GPU 0: NVIDIA RTX 3090 (24GB)
✓ Disk space: 156 GB free
✓ HuggingFace Hub accessible
✗ Warning: WANDB_API_KEY not set
✗ Warning: OPENAI_API_KEY not set

Summary: 6/8 checks passed
Status: Ready for development (warnings can be ignored for now)
```

---

### Task 1.9: Tokenization Utilities
**Priority:** MEDIUM | **Status:** ⬜ Not Started | **Estimate:** 3 hours

**File:** `utils/tokenization.py`

**Requirements:**
- [ ] Add special tokens to tokenizer
- [ ] Resize model embeddings
- [ ] Test tokenization with chat template
- [ ] Count tokens in narratives
- [ ] Truncation handling

---

### Task 1.10: Logging Configuration
**Priority:** LOW | **Status:** ⬜ Not Started | **Estimate:** 2 hours

**File:** `utils/logging_config.py`

**Requirements:**
- [ ] Configure loguru logger
- [ ] Set up log levels
- [ ] Add file logging
- [ ] Add colored console output
- [ ] Create log rotation

---

## Week 3: Data Pipeline

### Milestone: Data Collection and Processing Complete
**Deliverables:** Medical data configs, preprocessing scripts, example datasets

---

### Task 3.1: Medical Data Preprocessing
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 6 hours

**File:** `data/preprocess_medical.py`

**Requirements:**
- [ ] Text normalization
- [ ] PHI removal patterns
- [ ] Code format standardization
- [ ] Narrative length filtering
- [ ] Language detection
- [ ] Quality filtering

**Preprocessing Steps:**
```python
1. Remove PHI (regex patterns for names, dates, IDs)
2. Normalize text (lowercase, whitespace, punctuation)
3. Standardize codes (uppercase, format validation)
4. Filter by length (10-2000 characters)
5. Detect language (English only)
6. Quality score calculation
```

---

### Task 3.2: Stage 1 Data Configuration
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 2 hours

**File:** `data/config/thce_stage_1.yaml`

**Requirements:**
- [ ] Define data sources
- [ ] Set entry counts per source
- [ ] Configure column transformations
- [ ] Add data source documentation
- [ ] Validate YAML syntax

**Configuration:**
```yaml
name: "thce-sft-stage-1"
description: "Medical coding training data without reasoning"

datasets:
  # Use example data for development
  - name: "local"
    source_path: "data/examples/sample_stage_1.jsonl"
    split: "train"
    entries: 50

  # Future: Real data sources (commented out)
  # - name: "user/mimic-medical-codes"
  #   subset: "icd10_codes"
  #   split: "train"
  #   entries: 50000
```

---

### Task 3.3: Stage 2 Data Configuration
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 2 hours

**File:** `data/config/thce_stage_2.yaml`

**Similar to Task 3.2 but for Stage 2 format with reasoning**

---

### Task 3.4: Stage 3 Data Configuration
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 2 hours

**File:** `data/config/thce_stage_3.yaml`

**Similar to Task 3.2 but for Stage 3 preference pairs**

---

### Task 3.5: Example Dataset Creation
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 8 hours

**Files:**
- `data/examples/sample_stage_1.jsonl` (50 examples)
- `data/examples/sample_stage_2.jsonl` (50 examples)
- `data/examples/sample_stage_3.jsonl` (50 examples)

**Requirements:**
- [ ] Create 50 Stage 1 examples
- [ ] Create 50 Stage 2 examples (with reasoning)
- [ ] Create 50 Stage 3 preference pairs
- [ ] Validate all examples
- [ ] Use diverse medical specialties
- [ ] Cover all 3 code types
- [ ] Add README documenting examples

**Distribution:**
- ICD-10: 40% (20 examples)
- CPT: 30% (15 examples)
- HCPCS: 30% (15 examples)

**Specialties:**
- Endocrinology (diabetes, thyroid)
- Cardiology (hypertension, heart disease)
- Pulmonology (asthma, bronchitis)
- Orthopedics (fractures, mobility aids)
- General medicine

---

### Task 3.6: Synthetic Data Generation Script
**Priority:** MEDIUM | **Status:** ⬜ Not Started | **Estimate:** 10 hours

**File:** `data/synthetic_generation.py`

**Requirements:**
- [ ] OpenAI API integration
- [ ] Anthropic API integration
- [ ] Stage 1 generation
- [ ] Stage 2 generation (with reasoning)
- [ ] Stage 3 generation (preferences)
- [ ] Cost tracking
- [ ] Dry-run mode
- [ ] Batch processing
- [ ] Progress tracking
- [ ] Error handling and retries
- [ ] Quality filtering
- [ ] CLI interface

**CLI Interface:**
```bash
python data/synthetic_generation.py \
  --stage 1 \
  --count 100 \
  --model gpt-4 \
  --output data/synthetic/stage_1_batch_1.jsonl \
  --dry-run \
  --estimate-cost
```

**Output Format:**
```
Synthetic Data Generation
=========================

Configuration:
- Stage: 1
- Count: 100
- Model: gpt-4
- Output: data/synthetic/stage_1_batch_1.jsonl

Cost Estimate:
- Estimated tokens per example: 200
- Cost per example: $0.005
- Total cost: $0.50

Dry run mode - no API calls made.
Run without --dry-run to generate data.
```

---

### Task 3.7: Data Collection Testing
**Priority:** MEDIUM | **Status:** ⬜ Not Started | **Estimate:** 4 hours

**File:** `tests/integration/test_data_pipeline.py`

**Requirements:**
- [ ] Test data collection with example data
- [ ] Test preprocessing pipeline
- [ ] Test configuration loading
- [ ] Test output format
- [ ] Test metadata generation

---

### Task 3.8: Data Pipeline Documentation
**Priority:** MEDIUM | **Status:** ⬜ Not Started | **Estimate:** 3 hours

**File:** `docs/DATA_PREPARATION.md`

**Requirements:**
- [ ] Data sources guide
- [ ] Configuration format documentation
- [ ] Preprocessing steps explanation
- [ ] Example data creation guide
- [ ] Synthetic generation guide
- [ ] Troubleshooting section

---

## Week 4: Training Configuration

### Milestone: Training Configs and Validation Complete
**Deliverables:** Training configs, config validation, dry-run tests

---

### Task 4.1: Stage 1 Training Configuration
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 3 hours

**File:** `post_training/config/thce_stage_1.yaml`

**Requirements:**
- [ ] Set model path
- [ ] Configure dataset
- [ ] Set training hyperparameters
- [ ] Configure logging
- [ ] Set checkpoint strategy
- [ ] Add comments explaining params

**Configuration:**
```yaml
model: "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer: "HuggingFaceTB/SmolLM2-135M-Instruct"

dataset:
  name: "data/examples"  # Local path for development
  split: "train"
  num_proc: 4
  streaming: false

chat_template: |
  <|im_start|>system
  You are a medical coding assistant.<|im_end|>
  <|im_start|>user
  {user_message}<|im_end|>
  <|im_start|>assistant
  {assistant_message}<|im_end|>

trainer:
  output_dir: "outputs/stage_1"
  num_train_epochs: 3
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 3e-4
  # ... (full config)
```

---

### Task 4.2: Stage 2 Training Configuration
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 3 hours

**File:** `post_training/config/thce_stage_2.yaml`

**Similar to Task 4.1 but with:**
- Stage 1 checkpoint as base model
- Reasoning-focused chat template
- Special tokens for `<think>` tags

---

### Task 4.3: Stage 3 DPO Configuration
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 3 hours

**File:** `post_training/config/thce_stage_3.yaml`

**DPO-specific parameters:**
- beta: 0.1
- loss_type: apo_zero
- max_grad_norm: 0.2
- Lower learning rate: 1e-5

---

### Task 4.4: Config Validation Script
**Priority:** MEDIUM | **Status:** ⬜ Not Started | **Estimate:** 4 hours

**File:** `utils/validate_config.py`

**Requirements:**
- [ ] Load and parse YAML
- [ ] Validate required fields
- [ ] Check parameter types
- [ ] Validate parameter ranges
- [ ] Check file paths exist
- [ ] Validate model availability
- [ ] Generate validation report

**CLI Interface:**
```bash
python utils/validate_config.py \
  --config post_training/config/thce_stage_1.yaml
```

---

### Task 4.5: Training Dry-Run Tests
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 4 hours

**File:** `tests/integration/test_training_setup.py`

**Requirements:**
- [ ] Test loading base model
- [ ] Test loading tokenizer
- [ ] Test adding special tokens
- [ ] Test chat template application
- [ ] Test dataset loading
- [ ] Test training config parsing
- [ ] Dry-run training (no actual training)

---

### Task 4.6: Training Documentation
**Priority:** MEDIUM | **Status:** ⬜ Not Started | **Estimate:** 3 hours

**File:** `docs/TRAINING.md`

**Requirements:**
- [ ] Training overview
- [ ] Configuration guide
- [ ] Hyperparameter explanations
- [ ] Training monitoring guide
- [ ] Checkpoint management
- [ ] Troubleshooting guide
- [ ] Expected training times
- [ ] Resource requirements

---

## Week 5: Evaluation System

### Milestone: Complete Evaluation Pipeline
**Deliverables:** Metrics calculator, evaluation pipeline, baseline implementations

---

### Task 5.1: Metrics Implementation
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 8 hours

**File:** `evaluation/metrics.py`

**Requirements:**
- [ ] Exact match accuracy
- [ ] Top-K accuracy (K=1,3,5)
- [ ] Precision, Recall, F1
- [ ] Code type breakdown
- [ ] Confidence metrics
- [ ] Reasoning quality score (Stage 2+)
- [ ] Statistical significance tests

**Functions to Implement:**
```python
def calculate_exact_match(
    predictions: list[str],
    ground_truth: list[str]
) -> float:
    """Calculate exact match accuracy."""
    pass

def calculate_top_k_accuracy(
    predictions: list[list[str]],  # Top-K predictions
    ground_truth: list[str],
    k: int = 3
) -> float:
    """Calculate top-K accuracy."""
    pass

def calculate_precision_recall_f1(
    predictions: list[str],
    ground_truth: list[str]
) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 score."""
    pass

def calculate_code_type_breakdown(
    predictions: list[dict],
    ground_truth: list[dict]
) -> dict:
    """Calculate metrics by code type."""
    pass

def calculate_all_metrics(
    predictions_file: str,
    ground_truth_file: str
) -> dict:
    """Calculate all metrics."""
    pass
```

---

### Task 5.2: Evaluation Pipeline
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 10 hours

**File:** `evaluation/evaluate_model.py`

**Requirements:**
- [ ] Load trained model
- [ ] Load test dataset
- [ ] Run batch inference
- [ ] Extract predicted codes
- [ ] Parse reasoning (Stage 2+)
- [ ] Calculate metrics
- [ ] Compare against baselines
- [ ] Generate predictions file
- [ ] Generate evaluation report
- [ ] CLI interface

**CLI Interface:**
```bash
python evaluation/evaluate_model.py \
  --model-path outputs/stage_3_final \
  --test-data data/test/test_set.jsonl \
  --output results/evaluation_report.json \
  --compare-baseline \
  --max-samples 100  # Optional: limit for quick testing
```

---

### Task 5.3: Baseline Implementations
**Priority:** MEDIUM | **Status:** ⬜ Not Started | **Estimate:** 6 hours

**File:** `evaluation/baseline.py`

**Requirements:**
- [ ] Most frequent code baseline
- [ ] Random code baseline
- [ ] Keyword matching baseline
- [ ] Rule-based baseline
- [ ] Baseline comparison reports

**Baselines:**
```python
def most_frequent_baseline(
    train_data: list[dict],
    test_data: list[dict]
) -> list[str]:
    """Always predict most common code."""
    pass

def keyword_baseline(
    test_data: list[dict],
    keyword_map: dict[str, str]
) -> list[str]:
    """Match keywords to codes."""
    pass

def random_baseline(
    test_data: list[dict],
    code_vocabulary: list[str]
) -> list[str]:
    """Random code selection."""
    pass
```

---

### Task 5.4: Error Analysis Tools
**Priority:** MEDIUM | **Status:** ⬜ Not Started | **Estimate:** 6 hours

**File:** `evaluation/analysis.py`

**Requirements:**
- [ ] Categorize error types
- [ ] Identify challenging examples
- [ ] Analyze reasoning failures
- [ ] Generate error report
- [ ] Export error examples

**Error Categories:**
1. Wrong code entirely
2. Wrong specificity (E11 vs E11.9)
3. Wrong code type (ICD10 vs CPT)
4. Invalid code format
5. Missing reasoning (Stage 2+)
6. Incorrect reasoning (Stage 2+)

---

### Task 5.5: Metrics Unit Tests
**Priority:** MEDIUM | **Status:** ⬜ Not Started | **Estimate:** 4 hours

**File:** `tests/unit/test_metrics.py`

**Requirements:**
- [ ] Test exact match calculation
- [ ] Test top-K accuracy
- [ ] Test precision/recall/F1
- [ ] Test code type breakdown
- [ ] Test edge cases
- [ ] Achieve >85% coverage

---

### Task 5.6: Evaluation Documentation
**Priority:** MEDIUM | **Status:** ⬜ Not Started | **Estimate:** 3 hours

**File:** `docs/EVALUATION.md`

**Requirements:**
- [ ] Metrics explanations
- [ ] Evaluation pipeline guide
- [ ] Baseline descriptions
- [ ] How to interpret results
- [ ] Error analysis guide
- [ ] Examples

---

## Week 6: Documentation & Testing

### Milestone: Complete Documentation and CI/CD
**Deliverables:** All docs complete, tests passing, CI/CD working

---

### Task 6.1: README Update
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 4 hours

**File:** `README.md`

**Requirements:**
- [ ] Project overview
- [ ] Key features
- [ ] Quick start guide
- [ ] Installation instructions
- [ ] Usage examples
- [ ] Project structure
- [ ] Links to detailed docs
- [ ] Contributing guidelines
- [ ] License information

---

### Task 6.2: Setup Guide
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 3 hours

**File:** `docs/SETUP.md`

**Requirements:**
- [ ] Prerequisites
- [ ] Installation steps
- [ ] Environment configuration
- [ ] Credential setup
- [ ] Verification steps
- [ ] Troubleshooting
- [ ] Platform-specific notes

---

### Task 6.3: Complete Test Suite
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 8 hours

**Requirements:**
- [ ] All unit tests complete
- [ ] Integration tests complete
- [ ] Test fixtures complete
- [ ] Achieve 80%+ coverage
- [ ] All tests passing
- [ ] Fast test execution (<2 min)

---

### Task 6.4: CI/CD Pipeline
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 4 hours

**File:** `.github/workflows/tests.yaml`

**Requirements:**
- [ ] Run on push and PR
- [ ] Test on Python 3.12
- [ ] Install dependencies
- [ ] Run linter (ruff)
- [ ] Run tests with coverage
- [ ] Upload coverage to Codecov
- [ ] Fail on linting errors
- [ ] Fail on test failures

---

### Task 6.5: Pre-commit Hooks Update
**Priority:** MEDIUM | **Status:** ⬜ Not Started | **Estimate:** 2 hours

**File:** `.pre-commit-config.yaml`

**Requirements:**
- [ ] Ruff linting
- [ ] Ruff formatting
- [ ] YAML validation
- [ ] JSON validation
- [ ] Trailing whitespace removal
- [ ] End-of-file fixer

---

### Task 6.6: Example Notebooks
**Priority:** LOW | **Status:** ⬜ Not Started | **Estimate:** 6 hours

**Files:**
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_code_analysis.ipynb`
- `notebooks/03_model_evaluation.ipynb`

**Requirements:**
- [ ] Data exploration notebook
- [ ] Code statistics notebook
- [ ] Model evaluation notebook
- [ ] Clear documentation
- [ ] Reproducible outputs

---

### Task 6.7: Contribution Guide
**Priority:** LOW | **Status:** ⬜ Not Started | **Estimate:** 2 hours

**File:** `CONTRIBUTING.md`

**Requirements:**
- [ ] How to contribute
- [ ] Code style guidelines
- [ ] Testing requirements
- [ ] PR process
- [ ] Issue reporting

---

### Task 6.8: License and Legal
**Priority:** LOW | **Status:** ⬜ Not Started | **Estimate:** 1 hour

**Files:**
- `LICENSE`
- `PRIVACY.md`

**Requirements:**
- [ ] Choose appropriate license
- [ ] Add license file
- [ ] Privacy considerations
- [ ] HIPAA compliance notes

---

### Task 6.9: Environment Template
**Priority:** MEDIUM | **Status:** ⬜ Not Started | **Estimate:** 1 hour

**File:** `.env.example`

**Requirements:**
- [ ] All required env variables
- [ ] Descriptions for each
- [ ] Example values
- [ ] Security notes

```bash
# .env.example

# Hugging Face
HF_TOKEN=hf_xxxxx  # Get from https://huggingface.co/settings/tokens

# Weights & Biases
WANDB_API_KEY=xxxxx  # Get from https://wandb.ai/authorize
WANDB_PROJECT=thce-training
WANDB_ENTITY=your-username

# OpenAI (for synthetic data generation)
OPENAI_API_KEY=sk-xxxxx  # Get from https://platform.openai.com/api-keys

# Anthropic (alternative for synthetic data)
ANTHROPIC_API_KEY=sk-ant-xxxxx  # Get from https://console.anthropic.com/

# Optional: Custom paths
DATA_DIR=./data
OUTPUT_DIR=./outputs
CACHE_DIR=~/.cache/huggingface
```

---

### Task 6.10: Final Integration Test
**Priority:** HIGH | **Status:** ⬜ Not Started | **Estimate:** 4 hours

**File:** `tests/e2e/test_full_workflow.py`

**Requirements:**
- [ ] Test complete workflow with example data
- [ ] Data collection → Training setup → Evaluation
- [ ] Verify all scripts run
- [ ] Check output formats
- [ ] Validate end-to-end flow

---

## Task Dependencies

### Dependency Graph

```
1.2 (Code DB) → 1.1 (Validator) → 1.6 (Tests)
                       ↓
1.5 (Fixtures) → 1.3 (Quality) → 1.7 (Tests)
       ↓              ↓
1.4 (Templates) → 3.5 (Examples) → 3.1 (Preprocessing)
                       ↓              ↓
                  3.2-3.4 (Configs) → 3.7 (Tests)
                       ↓
                  4.1-4.3 (Training Configs) → 4.5 (Dry-run)
                       ↓
                  5.1 (Metrics) → 5.2 (Evaluation) → 5.5 (Tests)
                       ↓              ↓
                  5.3 (Baselines) → 5.4 (Analysis)
                       ↓
                  6.1-6.10 (Documentation & Final Testing)
```

### Critical Path (Longest chain)
```
1.5 → 1.3 → 3.5 → 3.1 → 3.2 → 4.1 → 4.5 → 5.1 → 5.2 → 5.5 → 6.3
Total: ~65 hours (13 days at 5 hours/day)
```

---

## Task Priorities

### Must Have (P0) - Week 1-4
1. Code validator + databases
2. Data quality checker
3. Chat templates
4. Example datasets
5. Data + training configs
6. Basic documentation

### Should Have (P1) - Week 5
7. Metrics implementation
8. Evaluation pipeline
9. Baseline implementations
10. Unit tests

### Nice to Have (P2) - Week 6
11. Synthetic generation script
12. Error analysis tools
13. Notebooks
14. Advanced documentation
15. CI/CD polish

---

## Progress Tracking

### Weekly Checkpoints

**Week 1-2 Goal:** Core utilities complete
- [ ] Tasks 1.1-1.10 complete
- [ ] All unit tests passing
- [ ] Code coverage >85%

**Week 3 Goal:** Data pipeline ready
- [ ] Tasks 3.1-3.8 complete
- [ ] Example datasets created
- [ ] Data collection tested

**Week 4 Goal:** Training configs validated
- [ ] Tasks 4.1-4.6 complete
- [ ] All configs validated
- [ ] Dry-run tests passing

**Week 5 Goal:** Evaluation system working
- [ ] Tasks 5.1-5.6 complete
- [ ] Metrics tested
- [ ] Baseline comparisons working

**Week 6 Goal:** Documentation complete
- [ ] Tasks 6.1-6.10 complete
- [ ] All docs written
- [ ] CI/CD passing
- [ ] Repository ready for Phase 2

---

## Time Estimates Summary

```
Week 1-2: 41 hours (Core Utilities)
Week 3:   29 hours (Data Pipeline)
Week 4:   17 hours (Training Config)
Week 5:   37 hours (Evaluation)
Week 6:   35 hours (Documentation)
─────────────────────────────────
Total:   159 hours (~26 days at 6 hours/day)
```

**Recommended Pace:**
- 15-20 hours/week = 8-11 weeks
- 20-25 hours/week = 6-8 weeks
- 25+ hours/week = 5-6 weeks

---

**Last Updated:** 2025-10-30
**Phase:** Engineering (Script Development)
**Status:** Planning Complete, Ready to Implement

# THCE Feature Specification
**Detailed Feature Requirements and Specifications**

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Data Features](#data-features)
3. [Training Features](#training-features)
4. [Evaluation Features](#evaluation-features)
5. [Utility Features](#utility-features)
6. [Future Features](#future-features)

---

## System Requirements

### Functional Requirements

#### FR-1: Code Validation System
**Priority:** HIGH
**Status:** COMPLETED (Phase 1)

**Description:**
System must validate medical billing codes against official code databases.

**Acceptance Criteria:**
- [x] Support ICD-10-CM code validation (alpha-numeric format)
- [x] Support CPT code validation (5-digit numeric format)
- [x] Support HCPCS code validation (alphanumeric format)
- [x] Load code databases from JSON files
- [x] Cache code databases for performance
- [x] Expose metadata lookup via `CodeValidator.get`
- [ ] Batch validation helper (not yet prioritised)
- [ ] CLI flag for exporting validation summaries

**Input:**
```python
code: str                  # e.g., "E11.9"
code_type: str            # "ICD10", "CPT", or "HCPCS"
```

**Output:**
```python
bool  # True if code exists in the requested database
```

**Example Usage:**
```python
from pathlib import Path
from utils.code_validator import CodeValidator

validator = CodeValidator(Path("data/code_databases"))

if validator.validate("E11.9", code_type="ICD10"):
    record = validator.get("E11.9", "ICD10")
    print("Valid:", record["short_description"])
else:
    print("Code not found.")
```

**Error Handling:**
- Raises `ValueError` for unsupported code types.
- Returns `False` when codes are missing (no exceptions).
- Raises `FileNotFoundError` when databases are not available.

---

#### FR-2: Data Quality Checking
**Priority:** HIGH
**Status:** IMPLEMENTED (Phase 1 scope)

**Description:**
System must validate training data format and quality before training.

**Acceptance Criteria:**
- [x] Validate JSONL structure and message format
- [x] Ensure required fields (`messages`, `_metadata`) exist
- [x] Confirm `<think>` reasoning for Stage 2
- [x] Validate codes using `CodeValidator`
- [x] Produce machine-readable report (JSON)
- [ ] Check narrative length constraints
- [ ] Detect potential PHI in narratives
- [ ] Attach severity levels to issues

**Quality Checks:**

| Check | Description | Stage |
|-------|-------------|-------|
| Schema validation | Valid JSON structure | All |
| Required fields | Has `messages` or preference pair | All |
| Code validity | Checked against curated databases | All |
| Reasoning tags | Requires `<think>` in Stage 2 | Stage 2 |
| Preference format | Ensures prompt/chosen/rejected keys | Stage 3 |

**Example Usage:**
```python
from pathlib import Path
from utils.code_validator import CodeValidator
from utils.data_quality import validate_dataset

validator = CodeValidator(Path("data/code_databases"))
report = validate_dataset(Path("data/examples/sample_stage_2.jsonl"), "stage_2", validator)

print(report.total_records)
print(report.to_dict())
```

---

#### FR-3: Data Collection Pipeline
**Priority:** HIGH
**Status:** PARTIALLY IMPLEMENTED (needs medical adaptation)

**Description:**
System must collect, merge, and prepare datasets from multiple sources.

**Acceptance Criteria:**
- [ ] Load datasets from Hugging Face Hub
- [ ] Support local file loading (JSON, JSONL, CSV)
- [ ] Stream large datasets without loading into memory
- [ ] Merge multiple datasets into single dataset
- [ ] Apply column transformations (drop, rename)
- [ ] Add source tracking metadata
- [ ] Shuffle dataset with fixed seed
- [ ] Save in efficient format (Parquet/Arrow)
- [ ] Generate dataset card/README
- [ ] Support uploading to Hugging Face Hub
- [ ] Track dataset statistics and composition

**Configuration Format:**
```yaml
name: "thce-stage-1-data"
description: "Medical coding training data"

datasets:
  - name: "user/medical-notes"
    subset: "icd10_codes"
    split: "train"
    entries: 50000
    drop_columns: ["id", "metadata"]
    rename_columns:
      note: "narrative"
      dx_code: "code"

  - name: "mtsamples/transcriptions"
    subset: null
    split: "train"
    entries: 5000
```

**Example Usage:**
```bash
python data/data_collection.py \
  --config-path data/config/thce_stage_1.yaml \
  --output-dir data/artefacts/stage_1 \
  --memory-monitor
```

---

#### FR-4: Medical Chat Templates
**Priority:** HIGH
**Status:** TO BE IMPLEMENTED

**Description:**
System must provide chat templates for formatting medical coding conversations.

**Acceptance Criteria:**
- [ ] Stage 1 template (no reasoning)
- [ ] Stage 2 template (with reasoning tags)
- [ ] Stage 3 template (preference format)
- [ ] System message variants for different stages
- [ ] Support for custom system messages
- [ ] Template validation function
- [ ] Documentation for each template

**Templates:**

**Stage 1 (No Reasoning):**
```
<|im_start|>system
You are a medical coding assistant. Convert clinical narratives to standard billing codes (ICD-10, CPT, HCPCS).<|im_end|>
<|im_start|>user
{narrative}<|im_end|>
<|im_start|>assistant
{code_type}: {code} ({description})<|im_end|>
```

**Stage 2 (With Reasoning):**
```
<|im_start|>system
You are a medical coding expert. Use <think> tags to show your step-by-step reasoning process.<|im_end|>
<|im_start|>user
{narrative}<|im_end|>
<|im_start|>assistant
<think>
{reasoning_steps}
</think>
{code_type}: {code} ({description})<|im_end|>
```

**Example Usage:**
```python
from utils.medical_chat_templates import get_stage_1_template, format_conversation

template = get_stage_1_template()
conversation = format_conversation(
    narrative="Patient has Type 2 diabetes",
    code="E11.9",
    code_type="ICD10",
    description="Type 2 diabetes mellitus without complications"
)
```

---

#### FR-5: Synthetic Data Generation
**Priority:** MEDIUM
**Status:** TO BE IMPLEMENTED

**Description:**
System must generate synthetic medical coding examples using LLM APIs.

**Acceptance Criteria:**
- [ ] Support OpenAI GPT-4 API
- [ ] Support Anthropic Claude API
- [ ] Generate Stage 1 format (no reasoning)
- [ ] Generate Stage 2 format (with reasoning)
- [ ] Generate Stage 3 preference pairs
- [ ] Validate generated codes
- [ ] Track generation costs
- [ ] Support batch generation
- [ ] Dry-run mode for cost estimation
- [ ] Progress tracking for large batches
- [ ] Error handling and retries
- [ ] Quality filtering of generated data

**Generation Parameters:**
```python
{
    "stage": 1 | 2 | 3,
    "count": int,              # Number of examples to generate
    "model": str,              # "gpt-4", "gpt-3.5-turbo", "claude-3-opus"
    "temperature": float,      # 0.0-1.0
    "code_types": list[str],   # ["ICD10", "CPT", "HCPCS"]
    "specialties": list[str],  # ["cardiology", "endocrinology", ...]
    "quality_threshold": float # Min quality score to keep
}
```

**Example Usage:**
```bash
# Generate 100 Stage 1 examples
python data/synthetic_generation.py \
  --stage 1 \
  --count 100 \
  --model gpt-4 \
  --output data/synthetic/stage_1_batch_1.jsonl

# Estimate cost without generating
python data/synthetic_generation.py \
  --stage 2 \
  --count 50000 \
  --model gpt-4 \
  --dry-run \
  --estimate-cost
```

**Cost Estimation:**
```
Stage 1 (no reasoning): ~$0.005 per example
Stage 2 (with reasoning): ~$0.015 per example
Stage 3 (preference pairs): ~$0.020 per pair

Estimated total for 160k examples: $400-600
```

---

#### FR-6: Training Pipeline (SFT)
**Priority:** HIGH
**Status:** IMPLEMENTED (verify compatibility)

**Description:**
System must support supervised fine-tuning for Stages 1 and 2.

**Acceptance Criteria:**
- [ ] Load base model from Hugging Face
- [ ] Load custom tokenizer
- [ ] Add special tokens (`<think>`, `</think>`)
- [ ] Apply chat template
- [ ] Load training dataset
- [ ] Configure training parameters from YAML
- [ ] Support gradient checkpointing
- [ ] Support mixed precision (bf16/fp16)
- [ ] Log metrics to W&B
- [ ] Save checkpoints periodically
- [ ] Resume from checkpoint
- [ ] Push final model to Hub (optional)

**Training Configuration:**
```yaml
model: "HuggingFaceTB/SmolLM2-135M-Instruct"
dataset:
  name: "user/thce-stage-1-data"
  split: "train"

trainer:
  output_dir: "outputs/stage_1"
  num_train_epochs: 3
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 3e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  max_seq_length: 4096
  bf16: true
  gradient_checkpointing: true
  logging_steps: 10
  save_steps: 500
```

**Example Usage:**
```bash
# Stage 1 training
python post_training/sft.py \
  --config-path post_training/config/thce_stage_1.yaml

# Stage 2 training (from Stage 1 checkpoint)
python post_training/sft.py \
  --config-path post_training/config/thce_stage_2.yaml
```

---

#### FR-7: Training Pipeline (DPO)
**Priority:** HIGH
**Status:** IMPLEMENTED (verify compatibility)

**Description:**
System must support Direct Preference Optimization for Stage 3.

**Acceptance Criteria:**
- [ ] Load Stage 2 checkpoint as base
- [ ] Load preference pairs dataset
- [ ] Support DPO loss variants (apo_zero, sigmoid)
- [ ] Configure DPO beta parameter
- [ ] Track reward margins
- [ ] Log chosen vs rejected accuracies
- [ ] Save final aligned model
- [ ] Support gradient clipping
- [ ] Push to Hub (optional)

**DPO Configuration:**
```yaml
model: "user/thce-stage-2-checkpoint"
dataset:
  name: "user/thce-stage-3-preferences"
  split: "train"

trainer:
  output_dir: "outputs/stage_3"
  num_train_epochs: 1
  per_device_train_batch_size: 32
  learning_rate: 1e-5
  beta: 0.1
  loss_type: "apo_zero"
  max_grad_norm: 0.2
```

**Example Usage:**
```bash
python post_training/dpo.py \
  --config-path post_training/config/thce_stage_3.yaml
```

---

#### FR-8: Evaluation Metrics
**Priority:** HIGH
**Status:** TO BE IMPLEMENTED

**Description:**
System must calculate comprehensive metrics for model evaluation.

**Acceptance Criteria:**
- [ ] Exact match accuracy
- [ ] Top-K accuracy (K=1, 3, 5)
- [ ] Precision, Recall, F1 score
- [ ] Code type breakdown (ICD10, CPT, HCPCS)
- [ ] Specialty breakdown (if available)
- [ ] Reasoning quality score (Stage 2+)
- [ ] Confidence calibration metrics
- [ ] Error categorization
- [ ] Statistical significance tests
- [ ] Generate evaluation report (JSON/Markdown)

**Metrics Specification:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Exact Match | Correct / Total | % predictions exactly matching ground truth |
| Top-K Accuracy | In-Top-K / Total | % where correct code is in top K predictions |
| Precision | TP / (TP + FP) | % of predictions that are correct |
| Recall | TP / (TP + FN) | % of ground truth codes predicted |
| F1 Score | 2 * (P * R) / (P + R) | Harmonic mean of precision and recall |

**Example Usage:**
```python
from evaluation.metrics import calculate_all_metrics

results = calculate_all_metrics(
    predictions_file="results/predictions.jsonl",
    ground_truth_file="data/test/test_set.jsonl"
)

print(f"Exact Match: {results['exact_match']:.2%}")
print(f"Top-3 Accuracy: {results['top_3_accuracy']:.2%}")
print(f"F1 Score: {results['f1_score']:.3f}")
```

---

#### FR-9: Model Evaluation Pipeline
**Priority:** HIGH
**Status:** TO BE IMPLEMENTED

**Description:**
System must provide end-to-end evaluation of trained models.

**Acceptance Criteria:**
- [ ] Load trained model checkpoint
- [ ] Load test dataset
- [ ] Run batch inference
- [ ] Extract predicted codes from outputs
- [ ] Parse reasoning traces (Stage 2+)
- [ ] Calculate all metrics
- [ ] Compare against baselines
- [ ] Generate predictions file
- [ ] Generate evaluation report
- [ ] Visualize results (optional)

**Example Usage:**
```bash
python evaluation/evaluate_model.py \
  --model-path outputs/stage_3_final \
  --test-data data/test/test_set.jsonl \
  --output results/evaluation_report.json \
  --compare-baseline
```

**Output:**
```json
{
  "model_path": "outputs/stage_3_final",
  "test_size": 1000,
  "metrics": {
    "exact_match": 0.87,
    "top_3_accuracy": 0.95,
    "precision": 0.88,
    "recall": 0.86,
    "f1_score": 0.87
  },
  "breakdown_by_type": {
    "ICD10": {"exact_match": 0.89, "count": 400},
    "CPT": {"exact_match": 0.85, "count": 300},
    "HCPCS": {"exact_match": 0.84, "count": 300}
  },
  "baseline_comparison": {
    "keyword_matching": 0.42,
    "most_frequent": 0.15,
    "model": 0.87
  }
}
```

---

#### FR-10: Baseline Implementations
**Priority:** MEDIUM
**Status:** TO BE IMPLEMENTED

**Description:**
System must provide simple baseline methods for comparison.

**Acceptance Criteria:**
- [ ] Random code selection baseline
- [ ] Most frequent code baseline
- [ ] Keyword matching baseline
- [ ] Rule-based heuristic baseline
- [ ] Support same evaluation metrics
- [ ] Generate comparison reports

**Baselines:**

**1. Most Frequent Code:**
```python
# Always predict the most common code in training set
def most_frequent_baseline(train_data, test_data):
    most_common = train_data['code'].value_counts().index[0]
    return [most_common] * len(test_data)
```

**2. Keyword Matching:**
```python
# Match keywords to codes
def keyword_baseline(narrative, keyword_map):
    for keyword, code in keyword_map.items():
        if keyword in narrative.lower():
            return code
    return "UNKNOWN"
```

**Example Usage:**
```bash
python evaluation/baseline.py \
  --test-data data/test/test_set.jsonl \
  --output results/baseline_results.json
```

---

## Data Features

### DF-1: Code Database Management

**Requirements:**
- Load ICD-10-CM codes (sample: top 500)
- Load CPT codes (sample: top 200)
- Load HCPCS codes (sample: top 100)
- Efficient lookup by code
- Search by description keywords
- Export database statistics

**Database Schema:**
```json
{
  "code": "E11.9",
  "code_type": "ICD10",
  "description": "Type 2 diabetes mellitus without complications",
  "category": "Endocrine, nutritional and metabolic diseases",
  "parent_code": "E11",
  "billable": true,
  "valid_from": "2015-10-01",
  "valid_until": null
}
```

### DF-2: Data Preprocessing

**Requirements:**
- Remove PHI (names, dates, identifiers)
- Normalize text (lowercase, whitespace)
- Standardize code formats
- Handle missing values
- Validate code existence
- Add metadata fields

**Preprocessing Pipeline:**
```
Raw Data → PHI Removal → Text Normalization → Code Validation → Format Conversion → Output
```

### DF-3: Dataset Statistics

**Requirements:**
- Track dataset composition
- Calculate code distribution
- Measure narrative length statistics
- Detect data imbalances
- Generate summary reports

**Example Report:**
```
Dataset: thce-stage-1-data
Total Examples: 50,000

Code Type Distribution:
- ICD10: 20,000 (40%)
- CPT: 15,000 (30%)
- HCPCS: 15,000 (30%)

Narrative Length:
- Mean: 156 characters
- Median: 142 characters
- Min: 15 characters
- Max: 1,987 characters

Top 10 Codes:
1. E11.9: 1,234 examples (2.5%)
2. I10: 1,102 examples (2.2%)
...
```

---

## Training Features

### TF-1: Training Monitoring

**Requirements:**
- Real-time loss tracking
- Learning rate schedule visualization
- GPU utilization monitoring
- Training speed (samples/sec)
- Estimated time remaining
- W&B integration for metrics

### TF-2: Checkpoint Management

**Requirements:**
- Save checkpoints every N steps
- Keep only last K checkpoints
- Save best checkpoint by metric
- Resume training from checkpoint
- Load checkpoint for inference

### TF-3: Hyperparameter Management

**Requirements:**
- YAML-based configuration
- Config validation
- Default values for optional params
- Config inheritance (base + overrides)
- Document all parameters

---

## Evaluation Features

### EF-1: Error Analysis

**Requirements:**
- Categorize error types
- Identify challenging examples
- Analyze reasoning failures (Stage 2+)
- Generate error report
- Export error examples for review

**Error Categories:**
```
- Wrong code entirely
- Wrong specificity (E11 vs E11.9)
- Wrong code type (ICD10 vs CPT)
- Invalid code format
- Missing reasoning (Stage 2+)
- Incorrect reasoning (Stage 2+)
```

### EF-2: Visualization (Optional)

**Requirements:**
- Confusion matrix heatmap
- Code distribution charts
- Training curves
- Error distribution by type
- Confidence calibration plots

---

## Utility Features

### UF-1: Setup Validation

**Requirements:**
- Check Python version
- Verify dependencies installed
- Test GPU availability
- Validate environment variables
- Check disk space
- Test HuggingFace connectivity

**Example:**
```bash
python scripts/validate_setup.py

✓ Python 3.12.1
✓ PyTorch 2.7.0
✓ CUDA available: True (GPU 0: RTX 3090)
✓ HuggingFace Hub accessible
✓ Environment variables configured
✓ Disk space: 156 GB free
✗ Warning: WANDB_API_KEY not set
```

### UF-2: Documentation Generation

**Requirements:**
- Auto-generate API docs from docstrings
- Generate dataset cards
- Generate model cards
- Update README with metrics
- Create usage examples

---

## Future Features

### Future-1: Multi-Code Outputs

**Description:**
Support narratives that require multiple codes.

**Example:**
```
Input: "Patient has diabetes and hypertension"
Output:
  - ICD10: E11.9 (Diabetes)
  - ICD10: I10 (Hypertension)
```

### Future-2: Code Hierarchy Awareness

**Description:**
Model understands ICD-10 hierarchy (parent-child relationships).

**Example:**
```
E11 (Type 2 diabetes) is parent of:
  - E11.9 (without complications)
  - E11.65 (with hyperglycemia)
  - E11.641 (with hypoglycemia with coma)
```

### Future-3: Real-Time Inference API

**Description:**
FastAPI endpoint for real-time code prediction.

**Endpoints:**
```
POST /predict
{
  "narrative": "Patient has diabetes",
  "return_reasoning": true
}

Response:
{
  "code": "E11.9",
  "code_type": "ICD10",
  "description": "...",
  "confidence": 0.95,
  "reasoning": "<think>...</think>"
}
```

### Future-4: Active Learning

**Description:**
System identifies uncertain predictions for human review.

**Workflow:**
```
1. Model makes predictions with confidence scores
2. Low-confidence examples flagged for review
3. Human labels uncertain examples
4. Add to training set
5. Retrain model
```

### Future-5: Multi-Modal Support

**Description:**
Accept PDFs, images of medical notes.

**Requirements:**
- PDF text extraction
- OCR for handwritten notes
- Image preprocessing
- Multi-modal model architecture

---

## Non-Functional Requirements

### NFR-1: Performance
- Training: Support single GPU (24GB VRAM)
- Inference: <100ms per prediction on GPU
- Data loading: Handle 100k+ examples efficiently
- Memory: Fit model in 8GB RAM for CPU inference

### NFR-2: Scalability
- Support datasets up to 500k examples
- Handle code vocabularies up to 5000 codes
- Scale to larger models (250M-1B parameters)

### NFR-3: Maintainability
- Code coverage >80%
- Documentation for all public APIs
- Type hints throughout
- Consistent code style (PEP 8)
- Clear error messages

### NFR-4: Security
- No PHI in logs or outputs
- Secure credential management
- No hardcoded secrets
- HIPAA compliance considerations

### NFR-5: Usability
- Clear CLI interfaces
- Helpful error messages
- Progress bars for long operations
- Comprehensive documentation
- Usage examples for all features

---

**Last Updated:** 2025-10-30
**Status:** ✅ Specification Complete
**Next:** See [tasks.md](tasks.md) for implementation tasks

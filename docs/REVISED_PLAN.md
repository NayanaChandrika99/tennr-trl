# THCE Project: Revised Implementation Plan
**Tiny Reasoning Language Model for Health Code Extraction**

**Last Updated:** 2025-10-30
**Status:** Engineering Phase - Script Development Only
**Budget:** $0 (Training deferred to future phase)

---

## Executive Summary

**Goal:** Build a complete, production-ready codebase for training a 135M parameter reasoning model that converts medical narratives into billing codes (ICD-10, CPT, HCPCS).

**Key Constraint:** NO budget for training or synthetic data generation in Phase 1.

**Deliverable:** A turnkey repository with all scripts, configs, tests, and documentation ready to execute when resources become available.

---

## Phase 1: Repository Engineering (Current Phase)

**Duration:** 4-6 weeks
**Budget:** $0
**Goal:** Complete codebase infrastructure

### What We're Building

#### 1. Core Infrastructure Scripts
- ✅ Data collection pipeline (adapted from trlm)
- 🔨 Medical code validation system
- 🔨 Medical-specific data preprocessing
- 🔨 Synthetic data generation scripts (ready for future use)
- ✅ Training scripts (SFT + DPO, already exists)
- 🔨 Evaluation and metrics system

#### 2. Configuration System
- 🔨 Medical-specific YAML configs for all 3 stages
- 🔨 Chat templates for medical coding
- 🔨 Code type definitions (ICD-10, CPT, HCPCS)
- 🔨 Hyperparameter configurations

#### 3. Utilities & Tools
- 🔨 Code validators (ensure codes are valid)
- 🔨 Data quality checkers
- 🔨 Medical tokenization handlers
- 🔨 Batch processing utilities

#### 4. Testing Framework
- 🔨 Unit tests for all utilities
- 🔨 Integration tests for pipelines
- 🔨 Sample/fixture data for testing
- 🔨 CI/CD configuration

#### 5. Documentation
- 🔨 Setup guides
- 🔨 Data preparation guides
- 🔨 Training guides
- 🔨 API documentation
- 🔨 Troubleshooting guides

### What We're NOT Doing (Deferred to Phase 2)
- ❌ Acquiring MIMIC dataset
- ❌ Generating synthetic training data
- ❌ Running actual training
- ❌ Model evaluation on real data
- ❌ Deployment infrastructure

---

## Technical Architecture

### Three-Stage Training Pipeline

```
Stage 1 (SFT - No Reasoning)
├── Input: Medical narrative
├── Output: Code only
└── Goal: Learn medical terminology & code formats

Stage 2 (SFT - With Reasoning)
├── Input: Medical narrative
├── Output: <think>reasoning</think> + Code
└── Goal: Learn to show reasoning process

Stage 3 (DPO - Alignment)
├── Input: Preference pairs (good vs bad reasoning)
├── Output: Aligned model
└── Goal: Prefer accurate, concise reasoning
```

### Model Specifications

**Base Model:** SmolLM2-135M-Instruct
- Parameters: 135M
- Context: 4096 tokens
- Size: ~500MB
- Training: Single GPU (RTX 3090 or equivalent)

**Special Tokens:**
```python
SPECIAL_TOKENS = {
    "reasoning_start": "<think>",
    "reasoning_end": "</think>",
    "code_types": ["ICD10:", "CPT:", "HCPCS:"]
}
```

---

## Data Schema Specifications

### Stage 1: Non-Reasoning Format
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a medical coding assistant. Convert clinical narratives to billing codes."
    },
    {
      "role": "user",
      "content": "Patient presents with acute bronchitis with cough and fever."
    },
    {
      "role": "assistant",
      "content": "ICD10: J20.9 (Acute bronchitis, unspecified)"
    }
  ],
  "_metadata": {
    "code_type": "ICD10",
    "code": "J20.9",
    "code_valid": true,
    "source": "synthetic|mimic|mtsamples"
  }
}
```

### Stage 2: With Reasoning Format
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a medical coding expert. Use <think> tags to show your reasoning."
    },
    {
      "role": "user",
      "content": "Patient requires walker for mobility assistance after hip surgery."
    },
    {
      "role": "assistant",
      "content": "<think>\n- Patient needs durable medical equipment (DME)\n- Walker is a standard DME item\n- Post-surgical mobility aid\n- HCPCS E-codes cover DME\n- E0130 is standard walker code\n</think>\nHCPCS: E0130 (Walker, rigid, adjustable or fixed height)"
    }
  ],
  "_metadata": {
    "code_type": "HCPCS",
    "code": "E0130",
    "reasoning_present": true,
    "reasoning_quality": "high"
  }
}
```

### Stage 3: Preference Pairs Format
```json
{
  "prompt": [
    {
      "role": "system",
      "content": "You are a medical coding expert. Use <think> tags to show your reasoning."
    },
    {
      "role": "user",
      "content": "Patient diagnosed with Type 2 diabetes mellitus without complications."
    }
  ],
  "chosen": "<think>\n- Type 2 diabetes is E11 series in ICD-10\n- No complications mentioned\n- E11.9 is unspecified Type 2 DM\n</think>\nICD10: E11.9 (Type 2 diabetes mellitus without complications)",
  "rejected": "<think>\nDiabetes... could be Type 1 or Type 2? Not sure. Maybe E10? Or E11? Let me guess E11.\n</think>\nICD10: E11.0 (Type 2 diabetes mellitus with hyperosmolarity)"
}
```

---

## Repository Structure

```
thce-project/
├── data/
│   ├── code_databases/              # Medical code reference files
│   │   ├── icd10_codes.json        # ICD-10-CM codes
│   │   ├── cpt_codes.json          # CPT procedure codes
│   │   ├── hcpcs_codes.json        # HCPCS supply codes
│   │   ├── load_codes.py           # Database loader utility
│   │   └── README.md               # Code database documentation
│   ├── config/
│   │   ├── thce_stage_1.yaml       # Stage 1 data config
│   │   ├── thce_stage_2.yaml       # Stage 2 data config
│   │   └── thce_stage_3.yaml       # Stage 3 data config
│   ├── examples/
│   │   ├── sample_stage_1.jsonl    # 50 examples for testing
│   │   ├── sample_stage_2.jsonl    # 50 examples for testing
│   │   └── sample_stage_3.jsonl    # 50 examples for testing
│   ├── data_collection.py          # Streaming data builder
│   ├── synthetic_generation.py     # Synthetic data generator (future)
│   └── preprocess_medical.py       # Medical-specific preprocessing
├── post_training/
│   ├── config/
│   │   ├── thce_stage_1.yaml       # Stage 1 training config
│   │   ├── thce_stage_2.yaml       # Stage 2 training config
│   │   └── thce_stage_3.yaml       # Stage 3 training config
│   ├── sft.py                      # Supervised fine-tuning script
│   └── dpo.py                      # DPO training script
├── evaluation/
│   ├── metrics.py                  # Medical coding metrics
│   ├── evaluate_model.py           # Model evaluation pipeline
│   ├── baseline.py                 # Baseline comparisons
│   └── analysis.py                 # Error analysis tools
├── utils/
│   ├── code_validator.py           # Validate ICD/CPT/HCPCS codes
│   ├── medical_chat_templates.py   # Chat templates
│   ├── data_quality.py             # Data quality checks
│   └── tokenization.py             # Medical tokenization utils
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Explore datasets
│   ├── 02_code_analysis.ipynb      # Analyze code distributions
│   └── 03_model_outputs.ipynb      # Analyze model outputs
├── tests/
│   ├── test_code_validator.py      # Test code validation
│   ├── test_data_quality.py        # Test data quality checks
│   ├── test_preprocessing.py       # Test preprocessing
│   └── fixtures/                   # Test fixtures
│       ├── valid_codes.json
│       └── sample_narratives.json
├── docs/
│   ├── overview.md                 # System architecture
│   ├── development.md              # Development guide
│   ├── testing.md                  # Testing guide
│   ├── spec.md                     # Feature specification
│   ├── tasks.md                    # Task breakdown
│   ├── SETUP.md                    # Setup instructions
│   ├── DATA_PREPARATION.md         # Data prep guide
│   ├── TRAINING.md                 # Training guide
│   └── EVALUATION.md               # Evaluation guide
├── .github/
│   └── workflows/
│       └── tests.yaml              # CI/CD pipeline
├── README.md                       # Project overview
├── pyproject.toml                  # Dependencies
└── .env.example                    # Environment variables template
```

---

## Implementation Roadmap

### Week 1-2: Core Utilities & Data Structures

**Deliverables:**
- [ ] Code validation system (`utils/code_validator.py`)
- [ ] Medical code databases (sample subset)
- [ ] Chat templates (`utils/medical_chat_templates.py`)
- [ ] Data quality checks (`utils/data_quality.py`)
- [ ] Unit tests for all utilities

**Key Scripts:**
```bash
# Code validator usage
python utils/code_validator.py --code "E11.9" --type ICD10

# Data quality check
python utils/data_quality.py --input data/examples/sample_stage_1.jsonl
```

### Week 3: Data Pipeline Scripts

**Deliverables:**
- [ ] Medical data configs (3 YAML files for data/)
- [ ] Preprocessing script (`data/preprocess_medical.py`)
- [ ] Synthetic generation script (`data/synthetic_generation.py`)
- [ ] Example datasets (50 examples each stage)

**Key Scripts:**
```bash
# Test data collection with examples
python data/data_collection.py \
  --config-path data/config/thce_stage_1.yaml \
  --output-dir data/artefacts/stage_1_test

# Generate synthetic data (future)
python data/synthetic_generation.py \
  --stage 1 \
  --count 100 \
  --model gpt-4 \
  --output data/synthetic/stage_1.jsonl
```

### Week 4: Training Configurations

**Deliverables:**
- [ ] Training configs (3 YAML files for post_training/)
- [ ] Modified training scripts for medical domain
- [ ] Checkpoint management utilities
- [ ] Training dry-run tests

**Key Scripts:**
```bash
# Dry run Stage 1 training (no actual training)
python post_training/sft.py \
  --config-path post_training/config/thce_stage_1.yaml \
  --dry-run

# Validate training config
python utils/validate_config.py \
  --config post_training/config/thce_stage_1.yaml
```

### Week 5: Evaluation System

**Deliverables:**
- [ ] Metrics calculation (`evaluation/metrics.py`)
- [ ] Evaluation pipeline (`evaluation/evaluate_model.py`)
- [ ] Baseline implementations (`evaluation/baseline.py`)
- [ ] Analysis notebooks

**Key Scripts:**
```bash
# Evaluate model (on example data)
python evaluation/evaluate_model.py \
  --model-path outputs/stage_3_final \
  --test-data data/examples/sample_test.jsonl \
  --output results/evaluation_report.json

# Calculate metrics
python evaluation/metrics.py \
  --predictions results/predictions.jsonl \
  --ground-truth data/examples/sample_test.jsonl
```

### Week 6: Documentation & Testing

**Deliverables:**
- [ ] Complete documentation (all docs/ files)
- [ ] Comprehensive README.md
- [ ] CI/CD pipeline (.github/workflows/)
- [ ] End-to-end testing
- [ ] Setup validation script

**Key Scripts:**
```bash
# Validate entire setup
python scripts/validate_setup.py

# Run all tests
pytest tests/ -v

# Generate documentation
python scripts/generate_docs.py
```

---

## Success Criteria for Phase 1

### Repository Completeness
- ✅ All directories created
- ✅ All scripts implemented and documented
- ✅ All configs created with comments
- ✅ All tests passing
- ✅ CI/CD pipeline functional

### Code Quality
- ✅ Type hints in all Python code
- ✅ Docstrings for all functions/classes
- ✅ PEP 8 compliant
- ✅ 80%+ test coverage
- ✅ No linter errors

### Documentation Quality
- ✅ README is comprehensive
- ✅ All docs/ files complete
- ✅ Inline code comments
- ✅ Usage examples for all scripts
- ✅ Troubleshooting guides

### Executability
- ✅ Scripts run without errors on sample data
- ✅ Configs are valid and tested
- ✅ Dependencies install cleanly
- ✅ Example data works end-to-end

---

## Phase 2: Data Acquisition & Training (Future)

**Prerequisites:**
- Budget: $500-800 for synthetic data + GPU time
- Time: 8-12 weeks
- Resources: MIMIC access OR synthetic data budget

**Phase 2 Tasks (Deferred):**
1. Apply for MIMIC-III dataset access
2. Generate 160k training examples
3. Run Stage 1 training (3 epochs, ~15-20 GPU hours)
4. Run Stage 2 training (1 epoch, ~10-15 GPU hours)
5. Run Stage 3 DPO (1 epoch, ~8-12 GPU hours)
6. Evaluate on test set
7. Iterate and improve

---

## Key Technical Decisions

### 1. Code Scope (Top-N Approach)
**Decision:** Focus on top 800 most frequent codes
- Top 500 ICD-10-CM codes (~70% coverage)
- Top 200 CPT codes (~60% coverage)
- Top 100 HCPCS codes (~50% coverage)

**Rationale:**
- 135M model cannot learn all 87,000+ codes
- Focus on high-frequency codes maximizes impact
- Extensible to more codes with larger models

### 2. Data Volume Targets
**Decision:** 50k / 70k / 40k examples for Stages 1/2/3

**Rationale:**
- Proven in trlm project
- Sufficient for 800 code vocabulary
- Manageable on single GPU

### 3. Synthetic vs Real Data
**Decision:** Hybrid approach
- Use MIMIC if available (preferred)
- Supplement with synthetic data
- All synthetic data validated for code correctness

**Rationale:**
- Real data is better but slow to acquire
- Synthetic data is controllable but expensive
- Hybrid approach balances both

### 4. Reasoning Format
**Decision:** Use `<think>` tags (proven in trlm)

**Rationale:**
- Clear delimiter for reasoning
- Easy to extract and analyze
- Works well with small models

---

## Risk Management

### Technical Risks

| Risk | Mitigation |
|------|------------|
| Scripts don't work when training starts | Dry-run tests, example data validation |
| Configs have errors | Config validation utility, schema checks |
| Code validator incomplete | Start with sample subset, expand later |
| Dependencies break | Lock all versions in pyproject.toml |

### Project Risks

| Risk | Mitigation |
|------|------------|
| Phase 2 never happens | Repository still valuable as reference |
| Budget never materializes | Scripts designed for efficiency |
| Requirements change | Modular design, easy to adapt |
| MIMIC access denied | Synthetic-only path documented |

---

## Deliverable Checklist

### Code Artifacts
- [ ] `data/data_collection.py` (adapted)
- [ ] `data/synthetic_generation.py` (new)
- [x] `data/preprocess_medical.py` (new)
- [x] `utils/code_validator.py` (new)
- [x] `utils/medical_chat_templates.py` (new)
- [x] `utils/data_quality.py` (new)
- [x] `evaluation/metrics.py` (new)
- [x] `evaluation/evaluate_model.py` (new)
- [x] `evaluation/baseline.py` (new)
- [ ] `post_training/sft.py` (exists, verify)
- [ ] `post_training/dpo.py` (exists, verify)

### Configuration Files
- [ ] `data/config/thce_stage_1.yaml`
- [ ] `data/config/thce_stage_2.yaml`
- [ ] `data/config/thce_stage_3.yaml`
- [ ] `post_training/config/thce_stage_1.yaml`
- [ ] `post_training/config/thce_stage_2.yaml`
- [ ] `post_training/config/thce_stage_3.yaml`

### Data Files
- [x] `data/code_databases/icd10_codes.json` (sample)
- [x] `data/code_databases/cpt_codes.json` (sample)
- [x] `data/code_databases/hcpcs_codes.json` (sample)
- [x] `data/examples/sample_stage_1.jsonl` (10 demo examples)
- [x] `data/examples/sample_stage_2.jsonl` (10 demo examples)
- [x] `data/examples/sample_stage_3.jsonl` (10 demo examples)

### Tests
- [x] `tests/test_code_validator.py`
- [x] `tests/test_data_quality.py`
- [x] `tests/test_preprocessing.py`
- [x] `tests/fixtures/` (all fixtures)

### Documentation
- [x] `README.md` (updated for THCE)
- [x] `docs/overview.md`
- [x] `docs/development.md`
- [ ] `docs/testing.md`
- [ ] `docs/spec.md`
- [ ] `docs/tasks.md`
- [x] `docs/SETUP.md`
- [x] `docs/DATA_PREPARATION.md`
- [x] `docs/TRAINING.md`
- [x] `docs/EVALUATION.md`

### CI/CD
- [x] `.github/workflows/uv-ci.yaml`
- [x] `.pre-commit-config.yaml` (update)

---

## Timeline Summary

**Total Duration:** 6 weeks
**Effort:** 15-20 hours/week
**Total Hours:** ~90-120 hours

| Week | Focus Area | Deliverables |
|------|------------|--------------|
| 1-2 | Core utilities & code validation | 4 scripts, tests |
| 3 | Data pipeline | 3 configs, 3 example files |
| 4 | Training configs | 3 configs, validation |
| 5 | Evaluation system | 3 scripts, notebooks |
| 6 | Documentation & final testing | All docs, CI/CD |

---

## Next Steps

1. **Review this plan** - Ensure alignment with goals
2. **Set up development environment** - See `docs/development.md`
3. **Start with Week 1-2 tasks** - Build core utilities first
4. **Use git branches** - Feature branches for each component
5. **Commit frequently** - Small, atomic commits
6. **Document as you go** - Update docs with discoveries

---

## Questions to Resolve

- [ ] Should we include MTSamples parser script (even if not using yet)?
- [ ] What subset of ICD-10 codes to include in sample database?
- [ ] Should evaluation scripts support multiple models for comparison?
- [ ] Include Jupyter notebooks in initial phase?
- [ ] Set up Weights & Biases integration now or later?

---

**Last Updated:** 2025-10-30
**Status:** ✅ Plan approved, ready to implement
**Next Milestone:** Week 1-2 deliverables complete

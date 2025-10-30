<p align="center"><strong>Tiny Health Coding Extractor (THCE)</strong></p>

# THCE – Tiny Reasoning Model for Medical Coding

THCE is a turnkey repository for building a 135M parameter SmolLM2 model that converts free-form clinical narratives into ICD-10, CPT, and HCPCS billing codes while exposing its reasoning with `<think>` tags.  
Phase 1 (this repo) focuses on engineering deliverables: datasets, validation tooling, configuration, documentation, and tests. Training is deferred to Phase 2 when compute budget is available.

---

## Highlights
- **Domain-specific data assets** – curated code databases and stage-aligned JSONL datasets under `data/examples/`.
- **Reusable pipelines** – preprocessing, synthetic generation scaffolding, and quality gates designed for medical coding.
- **Training ready** – SFT/DPO entry points, configs, and tokenizer helpers tuned for SmolLM2-135M.
- **Full documentation** – setup, development, data prep, training, evaluation, and roadmap guides in `docs/`.
- **Confidence checks** – pytest suite and pre-commit hooks keep the repository shippable.

---

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Bootstrap the environment
uv sync --group dev

# Activate the virtual environment if needed
source .venv/bin/activate  # Linux/macOS

# Run automated checks
uv run --group dev pytest
uvx pre-commit run --all-files  # optional once hooks are installed
```

Create a `.env` from `.env.example` to store tokens for Hugging Face, W&B, or model providers when you are ready to push artefacts.

---

## Working With The Toolkit

### Validate a Code
```bash
uv run python utils/code_validator.py --code E11.9 --type ICD10 --show-record
```

### Check Dataset Quality
```bash
uv run python utils/data_quality.py \
  --dataset data/examples/sample_stage_2.jsonl \
  --stage stage_2 \
  --json
```

### Convert Raw Records Into Stage Formats
```bash
uv run python data/preprocess_medical.py \
  --input tests/fixtures/raw_records.jsonl \
  --output /tmp/stage_2_output.jsonl \
  --stage stage_2
```

### Produce Synthetic Scaffolding
```bash
uv run python data/synthetic_generation.py \
  --stage stage_1 \
  --count 20 \
  --output /tmp/thce_stage_1_synth.jsonl
```

### Drive LLM-Based Synthetic Generation
```bash
# Replace default_request_fn in scripts/generate_synthetic.py with your provider client first.
uv run python scripts/generate_synthetic.py \
  --output data/synthetic/stage_1_llm.jsonl
```

### Dry-Run Training
```bash
uv run python post_training/sft.py \
  --config-path post_training/config/stage_1.yaml \
  --dry-run
```

### Run Baseline Evaluation
```bash
uv run python evaluation/evaluate_model.py \
  --dataset data/processed/stage_2_val.jsonl \
  --baseline majority_code \
  --report-path reports/sample_stage_2.json
```

### Generate Predictions from a Trained Model
```bash
uv run python scripts/generate_predictions.py \
  --model-path outputs/stage_3 \
  --dataset '{"type": "local", "path": "data/processed/stage_2_val.jsonl"}' \
  --output reports/stage_2_predictions.jsonl
```

### Inspect Evaluation Reports
```bash
uv run python evaluation/analysis.py \
  --report-path reports/sample_stage_2.json \
  --top-k 3
```

---

## Data Assets

- `data/code_databases/*.json` – trimmed ICD-10, CPT, and HCPCS subsets used for validation.
- `data/examples/sample_stage_*.jsonl` – stage-specific datasets with narratives, reasoning traces, and preference pairs.
- `tests/fixtures/raw_records.jsonl` – compact fixture for preprocessing and integration testing.

Schemas and expectations are documented in `docs/DATA_PREPARATION.md`.

---

## Repository Layout

```
.
├── .github/workflows/uv-ci.yaml        # CI pipeline (lint + tests)
├── data
│   ├── code_databases
│   │   ├── cpt_codes.json
│   │   ├── hcpcs_codes.json
│   │   └── icd10_codes.json
│   ├── config
│   │   ├── stage_1.yaml
│   │   ├── stage_2.yaml
│   │   └── stage_3.yaml
│   ├── data_collection.py
│   ├── preprocess_medical.py
│   └── synthetic_generation.py
├── evaluation
│   ├── analysis.py
│   ├── baseline.py
│   ├── evaluate_model.py
│   └── metrics.py
├── post_training
│   ├── config
│   │   ├── stage_1.yaml
│   │   ├── stage_2.yaml
│   │   └── stage_3.yaml
│   ├── dpo.py
│   └── sft.py
├── tests
│   ├── fixtures/raw_records.jsonl
│   ├── test_code_validator.py
│   ├── test_data_quality.py
│   ├── test_metrics.py
│   └── test_preprocessing.py
├── utils
│   ├── code_validator.py
│   ├── data_quality.py
│   ├── medical_chat_templates.py
│   └── tokenization.py
├── .pre-commit-config.yaml
├── pyproject.toml
└── uv.lock
```

---

## Environmental Configuration

```bash
# .env.example
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxx
WANDB_PROJECT=thce-training
WANDB_ENTITY=your-org
OPENAI_API_KEY=sk-your-key           # optional, synthetic generation
ANTHROPIC_API_KEY=sk-ant-your-key    # optional alternative provider
```

---

## Phase 1 Deliverables

- ✅ Data scaffolding, preprocessing pipeline, and validator tooling
- ✅ Stage-specific configuration and documentation
- ✅ Automated tests and CI-ready workflow
- ✅ Training entry points configured for future GPU runs

---

## License & Credits

This repository is provided for research and demonstration.  
Inspired by the Tiny Reasoning Language Model work by Shekswess et al. and adapted for the medical coding domain.

# THCE Development Guide
**Commands, Workflows, and Setup Instructions**

---

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Development Environment](#development-environment)
3. [Common Commands](#common-commands)
4. [Development Workflows](#development-workflows)
5. [Configuration Management](#configuration-management)
6. [Debugging Tips](#debugging-tips)
7. [Git Workflow](#git-workflow)

---

## Initial Setup

### Prerequisites

```bash
# Required
- Python 3.12+
- Git
- 16GB+ RAM
- 50GB+ free disk space

# Optional (for training)
- CUDA-compatible GPU
- 24GB+ VRAM
```

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/tennr-trl.git
cd tennr-trl
```

### Step 2: Install UV Package Manager

```bash
# Install UV (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install UV (Windows - PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

### Step 3: Create Virtual Environment

```bash
# Create and sync environment from pyproject.toml (includes dev tools)
uv sync --group dev

# Activate virtual environment
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows PowerShell

# Verify Python version
python --version  # Should be 3.12+
```

### Step 4: Install Dependencies

```bash
# Optional: install CPU-only PyTorch (development without GPU)
uv sync --extra cpu

# Optional: Install ROCm support (for AMD GPUs)
uv sync --extra rocm
```

### Step 5: Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file
nano .env
```

```bash
# .env file contents
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx        # Hugging Face token
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx   # Weights & Biases API key
WANDB_PROJECT=thce-training              # W&B project name
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxx     # OpenAI API (for synthetic data)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx        # Anthropic API (alternative)
```

### Step 6: Verify Setup

```bash
# Run automated checks
uv run --group dev pytest
uvx pre-commit run --all-files
```

---

## Development Environment

### Directory Structure Overview

```bash
# View current structure
tree -L 2 -I '__pycache__|*.pyc|.git'

# Or use find
find . -maxdepth 2 -type d | grep -v ".git" | sort
```

### IDE Setup

#### VS Code (Recommended)

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "none",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false
}
```

#### PyCharm

```
1. Open project
2. Settings → Project → Python Interpreter
3. Select existing virtualenv: .venv/
4. Enable Ruff linter
5. Configure pytest as test runner
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files

# Update hooks to latest versions
pre-commit autoupdate
```

---

## Common Commands

### Data Processing

#### Validate Medical Codes

```bash
# Validate a single code
python utils/code_validator.py --code "E11.9" --type ICD10

# Validate codes from a file
python utils/code_validator.py --input data/raw/codes.txt --output data/validated_codes.json

# Check code database statistics
python utils/code_validator.py --stats
```

#### Run Data Quality Checks

```bash
# Check data quality
python utils/data_quality.py --input data/examples/sample_stage_1.jsonl

# Check with detailed report
python utils/data_quality.py \
  --input data/examples/sample_stage_1.jsonl \
  --output reports/quality_report.json \
  --verbose

# Check all example files
for file in data/examples/*.jsonl; do
  echo "Checking $file..."
  python utils/data_quality.py --input "$file"
done
```

#### Collect and Merge Datasets

```bash
# Stage 1 data collection (using example data)
python data/data_collection.py \
  --config-path data/config/thce_stage_1.yaml \
  --output-dir data/artefacts/stage_1

# Stage 2 data collection
python data/data_collection.py \
  --config-path data/config/thce_stage_2.yaml \
  --output-dir data/artefacts/stage_2

# With memory monitoring
python data/data_collection.py \
  --config-path data/config/thce_stage_1.yaml \
  --output-dir data/artefacts/stage_1 \
  --memory-monitor

# Upload to Hugging Face Hub (future)
python data/data_collection.py \
  --config-path data/config/thce_stage_1.yaml \
  --upload-to-hub
```

#### Generate Synthetic Data (Future)

```bash
# Generate synthetic examples for Stage 1
python data/synthetic_generation.py \
  --stage 1 \
  --count 100 \
  --model gpt-4 \
  --output data/synthetic/stage_1_batch_1.jsonl

# Generate with cost estimation
python data/synthetic_generation.py \
  --stage 1 \
  --count 100 \
  --model gpt-4 \
  --dry-run \
  --estimate-cost

# Generate Stage 2 with reasoning
python data/synthetic_generation.py \
  --stage 2 \
  --count 100 \
  --model gpt-4 \
  --output data/synthetic/stage_2_batch_1.jsonl \
  --reasoning-quality high
```

### Training

#### Dry Run Training (No Actual Training)

```bash
# Validate Stage 1 config without training
python post_training/sft.py \
  --config-path post_training/config/thce_stage_1.yaml \
  --dry-run

# Test data loading for Stage 2
python post_training/sft.py \
  --config-path post_training/config/thce_stage_2.yaml \
  --dry-run \
  --max-steps 10
```

#### Run Training (Future)

```bash
# Stage 1: Non-reasoning SFT
python post_training/sft.py \
  --config-path post_training/config/thce_stage_1.yaml

# Stage 2: Reasoning SFT (starts from Stage 1 checkpoint)
python post_training/sft.py \
  --config-path post_training/config/thce_stage_2.yaml

# Stage 3: DPO alignment (starts from Stage 2 checkpoint)
python post_training/dpo.py \
  --config-path post_training/config/thce_stage_3.yaml

# Resume from checkpoint
python post_training/sft.py \
  --config-path post_training/config/thce_stage_1.yaml \
  --resume-from-checkpoint outputs/stage_1/checkpoint-1500
```

#### Monitor Training

```bash
# View training logs
tail -f outputs/stage_1/training.log

# Check GPU usage
watch -n 1 nvidia-smi

# View W&B dashboard
# Visit: https://wandb.ai/your-username/thce-training
```

### Evaluation

#### Run Model Evaluation (Future)

```bash
# Evaluate final model
python evaluation/evaluate_model.py \
  --model-path outputs/stage_3_final \
  --test-data data/examples/sample_test.jsonl \
  --output results/evaluation_report.json

# Evaluate with baseline comparison
python evaluation/evaluate_model.py \
  --model-path outputs/stage_3_final \
  --test-data data/examples/sample_test.jsonl \
  --compare-baseline \
  --output results/evaluation_with_baseline.json

# Quick evaluation on small sample
python evaluation/evaluate_model.py \
  --model-path outputs/stage_3_final \
  --test-data data/examples/sample_test.jsonl \
  --max-samples 50 \
  --output results/quick_eval.json
```

#### Calculate Metrics

```bash
# Calculate metrics from predictions
python evaluation/metrics.py \
  --predictions results/predictions.jsonl \
  --ground-truth data/examples/sample_test.jsonl \
  --output results/metrics.json

# Calculate metrics by code type
python evaluation/metrics.py \
  --predictions results/predictions.jsonl \
  --ground-truth data/examples/sample_test.jsonl \
  --breakdown-by-type \
  --output results/metrics_by_type.json
```

#### Run Baseline Comparisons

```bash
# Run all baselines
python evaluation/baseline.py \
  --test-data data/examples/sample_test.jsonl \
  --output results/baseline_results.json

# Run specific baseline
python evaluation/baseline.py \
  --test-data data/examples/sample_test.jsonl \
  --baseline keyword_matching \
  --output results/keyword_baseline.json
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_code_validator.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -v -m "not slow"

# Run in parallel (faster)
pytest tests/ -n auto
```

### Code Quality

```bash
# Run linter
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Format code
ruff format .

# Type checking (if using type checker)
pyright .

# Check all quality at once
ruff check . && ruff format --check . && pytest tests/
```

---

## Development Workflows

### Workflow 1: Adding a New Medical Code Type

**Scenario:** Adding NDC (National Drug Code) support

```bash
# Step 1: Add NDC code database
mkdir -p data/code_databases
# (Download NDC database from FDA)
# Save as: data/code_databases/ndc_codes.json

# Step 2: Update code validator
# Edit: utils/code_validator.py
# Add NDC validation logic

# Step 3: Add NDC to chat templates
# Edit: utils/medical_chat_templates.py
# Add "NDC:" prefix option

# Step 4: Create test data
# Create: data/examples/sample_ndc_codes.jsonl

# Step 5: Add tests
# Edit: tests/test_code_validator.py
# Add test_validate_ndc_code()

# Step 6: Test changes
pytest tests/test_code_validator.py::test_validate_ndc_code -v

# Step 7: Update documentation
# Edit: docs/spec.md
# Document NDC support

# Step 8: Commit changes
git add .
git commit -m "feat: Add NDC (National Drug Code) support"
```

### Workflow 2: Creating Sample Training Data

```bash
# Step 1: Create a new example file
touch data/examples/new_examples.jsonl

# Step 2: Add examples in correct format
cat > data/examples/new_examples.jsonl << 'EOF'
{"messages": [{"role": "system", "content": "..."}, ...]}
{"messages": [{"role": "system", "content": "..."}, ...]}
EOF

# Step 3: Validate data quality
python utils/data_quality.py --input data/examples/new_examples.jsonl

# Step 4: Test with data collector
python data/data_collection.py \
  --config-path data/config/test_config.yaml \
  --output-dir data/artefacts/test_run

# Step 5: Verify output
ls -lh data/artefacts/test_run/
head data/artefacts/test_run/train/*.parquet

# Step 6: Commit examples
git add data/examples/new_examples.jsonl
git commit -m "data: Add new training examples for X"
```

### Workflow 3: Testing Training Pipeline End-to-End

```bash
# Step 1: Prepare minimal test dataset
python data/data_collection.py \
  --config-path data/config/thce_stage_1.yaml \
  --output-dir data/artefacts/stage_1_test

# Step 2: Run training dry-run
python post_training/sft.py \
  --config-path post_training/config/thce_stage_1.yaml \
  --dry-run

# Step 3: Run minimal training (few steps)
# Edit config temporarily: max_steps: 10
python post_training/sft.py \
  --config-path post_training/config/thce_stage_1.yaml

# Step 4: Check outputs
ls outputs/stage_1/
python -c "from transformers import AutoModelForCausalLM; \
           model = AutoModelForCausalLM.from_pretrained('outputs/stage_1/checkpoint-10'); \
           print('Model loaded successfully!')"

# Step 5: Clean up test outputs
rm -rf outputs/stage_1/
```

### Workflow 4: Updating Configuration Files

```bash
# Step 1: Create new config from template
cp data/config/thce_stage_1.yaml data/config/my_custom_config.yaml

# Step 2: Edit configuration
nano data/config/my_custom_config.yaml

# Step 3: Validate config syntax
python -c "import yaml; yaml.safe_load(open('data/config/my_custom_config.yaml'))"

# Step 4: Test config with dry-run
python data/data_collection.py \
  --config-path data/config/my_custom_config.yaml \
  --output-dir /tmp/test_output

# Step 5: Commit config
git add data/config/my_custom_config.yaml
git commit -m "config: Add custom data collection config"
```

### Workflow 5: Adding a New Evaluation Metric

```bash
# Step 1: Edit metrics module
# File: evaluation/metrics.py
# Add new metric function

# Step 2: Add unit test
# File: tests/test_metrics.py
cat >> tests/test_metrics.py << 'EOF'
def test_new_metric():
    predictions = [...]
    ground_truth = [...]
    result = calculate_new_metric(predictions, ground_truth)
    assert result > 0
EOF

# Step 3: Test new metric
pytest tests/test_metrics.py::test_new_metric -v

# Step 4: Integrate into evaluation pipeline
# Edit: evaluation/evaluate_model.py
# Call new metric in main evaluation loop

# Step 5: Test end-to-end
python evaluation/metrics.py \
  --predictions results/test_predictions.jsonl \
  --ground-truth data/examples/sample_test.jsonl

# Step 6: Document metric
# Edit: docs/EVALUATION.md

# Step 7: Commit changes
git add evaluation/metrics.py tests/test_metrics.py docs/EVALUATION.md
git commit -m "feat: Add new evaluation metric X"
```

---

## Configuration Management

### Data Configuration (data/config/*.yaml)

```yaml
# Structure of data config files
name: "thce-sft-stage-1"
description: "Medical coding training data without reasoning"

datasets:
  - name: "your-username/medical-narratives"
    subset: "icd10_codes"
    split: "train"
    entries: 50000
    drop_columns: ["metadata", "source_id"]
    rename_columns:
      old_code: "code"
      old_narrative: "narrative"

  - name: "mtsamples/transcriptions"
    subset: null
    split: "train"
    entries: 5000
```

**Common Parameters:**
- `name`: HuggingFace dataset name
- `subset`: Dataset configuration name
- `split`: Train/validation/test split
- `entries`: Number of examples to take (null = all)
- `drop_columns`: Columns to remove
- `rename_columns`: Column renaming map

### Training Configuration (post_training/config/*.yaml)

```yaml
# Structure of training config files
model: "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer: "HuggingFaceTB/SmolLM2-135M-Instruct"

dataset:
  name: "your-username/thce-stage-1-data"
  subset: null
  split: "train"
  num_proc: 8
  streaming: false

tokenizer_additional_special_tokens:
  - "<think>"
  - "</think>"

chat_template: |
  <|im_start|>system
  {system_message}<|im_end|>
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
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_seq_length: 4096
  bf16: true
  gradient_checkpointing: true
  logging_steps: 10
  save_steps: 500
  save_total_limit: 3
  report_to: "wandb"
  hub_model_id: "your-username/thce-stage-1"
  push_to_hub: false
```

### Environment Variables (.env)

```bash
# Authentication
HF_TOKEN=hf_xxxxx                    # Hugging Face API token
WANDB_API_KEY=xxxxx                  # Weights & Biases API key
OPENAI_API_KEY=sk-xxxxx              # OpenAI API key (for synthetic data)
ANTHROPIC_API_KEY=sk-ant-xxxxx       # Anthropic API key (alternative)

# Project Configuration
WANDB_PROJECT=thce-training          # W&B project name
WANDB_ENTITY=your-username           # W&B username/team
HF_USERNAME=your-username            # HuggingFace username

# Paths (optional, defaults exist)
DATA_DIR=./data                      # Data directory
OUTPUT_DIR=./outputs                 # Training outputs
CACHE_DIR=~/.cache/huggingface      # HF cache directory

# Training Configuration
CUDA_VISIBLE_DEVICES=0               # GPU device ID
TOKENIZERS_PARALLELISM=true          # Parallel tokenization
```

---

## Debugging Tips

### Common Issues and Solutions

#### Issue: "CUDA out of memory"

```bash
# Solution 1: Reduce batch size
# Edit config: per_device_train_batch_size: 16  (instead of 32)

# Solution 2: Increase gradient accumulation
# Edit config: gradient_accumulation_steps: 8  (instead of 4)

# Solution 3: Enable gradient checkpointing (already enabled)
# Config: gradient_checkpointing: true

# Solution 4: Use CPU offloading (slower but works)
# Edit config: device_map: "auto"
```

#### Issue: "Dataset not found"

```bash
# Check if dataset exists on HuggingFace
python -c "from datasets import load_dataset; load_dataset('your-username/dataset-name')"

# If local dataset, check path
ls -l data/artefacts/stage_1/

# Use absolute paths
python data/data_collection.py --output-dir $(pwd)/data/artefacts/stage_1
```

#### Issue: "Token limit exceeded"

```bash
# Check tokenizer
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
print(f'Max length: {tokenizer.model_max_length}')
"

# Reduce max_seq_length in config
# Edit config: max_seq_length: 2048  (instead of 4096)
```

#### Issue: "Invalid code in training data"

```bash
# Run data quality check
python utils/data_quality.py --input data/examples/sample_stage_1.jsonl

# Validate all codes
python utils/code_validator.py --input data/codes_to_check.txt --output invalid_codes.txt

# Filter dataset
python scripts/filter_invalid_codes.py \
  --input data/raw/dataset.jsonl \
  --output data/clean/dataset.jsonl
```

### Debugging Training

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python post_training/sft.py --config-path post_training/config/thce_stage_1.yaml

# Run with minimal data
# Edit config temporarily: max_steps: 10
python post_training/sft.py --config-path post_training/config/thce_stage_1.yaml

# Check data loading
python -c "
from datasets import load_dataset
dataset = load_dataset('path/to/dataset')
print(dataset[0])
"

# Profile memory usage
python -m memory_profiler post_training/sft.py --config-path post_training/config/thce_stage_1.yaml
```

### Debugging Data Pipeline

```bash
# Check YAML syntax
python -c "import yaml; print(yaml.safe_load(open('data/config/thce_stage_1.yaml')))"

# Test data collection with small sample
# Edit config: entries: 100  (for each dataset)
python data/data_collection.py --config-path data/config/thce_stage_1.yaml

# Check output format
head -1 data/artefacts/stage_1/train/*.parquet | python -m json.tool

# Validate schema
python utils/validate_dataset_schema.py --input data/artefacts/stage_1/
```

---

## Git Workflow

### Branching Strategy

```bash
# Main branches
main                    # Stable, production-ready code
develop                # Integration branch

# Feature branches (naming convention)
feature/add-ndc-support
fix/code-validator-bug
docs/update-readme
config/stage-1-tuning
```

### Common Git Commands

```bash
# Check status
git status

# Create feature branch
git checkout -b feature/my-new-feature

# Stage changes
git add utils/code_validator.py tests/test_code_validator.py

# Commit with descriptive message
git commit -m "feat: Add NDC code validation support

- Add NDC database loader
- Implement NDC validation logic
- Add unit tests for NDC codes
- Update documentation"

# Push to remote
git push -u origin feature/my-new-feature

# Pull latest changes
git pull origin main

# Merge main into feature branch
git checkout feature/my-new-feature
git merge main

# View commit history
git log --oneline --graph --decorate

# View changes
git diff
git diff --staged
```

### Commit Message Convention

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `config`: Configuration changes
- `test`: Adding tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

**Examples:**

```bash
# Feature
git commit -m "feat(validator): Add CPT code validation"

# Bug fix
git commit -m "fix(data): Handle missing code fields gracefully"

# Documentation
git commit -m "docs(readme): Update setup instructions"

# Configuration
git commit -m "config(stage-1): Increase batch size to 64"

# Multiple changes
git commit -m "feat(evaluation): Add baseline comparison

- Implement keyword matching baseline
- Add most-frequent-code baseline
- Compare against model predictions
- Generate comparison report"
```

### Working with Remotes

```bash
# View remotes
git remote -v

# Add remote
git remote add origin https://github.com/yourusername/tennr-trl.git

# Fetch from remote
git fetch origin

# Pull from specific branch
git pull origin main

# Push to specific branch
git push origin feature/my-feature

# Delete remote branch
git push origin --delete feature/old-feature
```

---

## Quick Reference

### One-Liners

```bash
# Validate setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check dataset size
python -c "from datasets import load_from_disk; print(len(load_from_disk('data/artefacts/stage_1')))"

# Count examples in JSONL
wc -l data/examples/sample_stage_1.jsonl

# View first example
head -1 data/examples/sample_stage_1.jsonl | python -m json.tool

# Check model size
du -sh outputs/stage_1/checkpoint-1000/

# Monitor GPU
watch -n 1 nvidia-smi

# Find Python scripts
find . -name "*.py" -not -path "./.venv/*" | sort

# Check disk space
df -h
```

### Useful Aliases

```bash
# Add to ~/.bashrc or ~/.zshrc

alias thce-activate='source .venv/bin/activate'
alias thce-test='pytest tests/ -v'
alias thce-lint='ruff check .'
alias thce-format='ruff format .'
alias thce-validate='python scripts/validate_setup.py'
alias thce-gpu='watch -n 1 nvidia-smi'
```

---

**Last Updated:** 2025-10-30
**Next:** See [testing.md](testing.md) for testing workflows

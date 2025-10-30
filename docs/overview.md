# THCE System Overview
**Tiny Reasoning Language Model for Health Code Extraction**

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Technology Stack](#technology-stack)
3. [Data Flow](#data-flow)
4. [Model Architecture](#model-architecture)
5. [Component Overview](#component-overview)
6. [Integration Points](#integration-points)

---

## System Architecture

> **Phase 1 note:** Sample datasets shipped with the repository contain 10 demonstration records per stage. The diagrams below describe the full-scale pipeline envisioned for Phase 2 (50k–80k examples per stage).

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     THCE System Architecture                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Data Sources    │
│  ├─ MIMIC-III    │
│  ├─ MTSamples    │
│  └─ Synthetic    │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data Pipeline Layer                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │   Code      │  │  Data        │  │   Synthetic        │    │
│  │ Validation  │→ │ Collection   │← │   Generation       │    │
│  │  System     │  │  & Cleaning  │  │   (Future)         │    │
│  └─────────────┘  └──────┬───────┘  └────────────────────┘    │
└────────────────────────────┼──────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Stage 1    │    │   Stage 2    │    │   Stage 3    │
│  Non-Reason  │    │ With Reason  │    │     DPO      │
│   Dataset    │    │   Dataset    │    │  Preference  │
│  (50-60k)    │    │  (70-80k)    │    │   (40-50k)   │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Training Pipeline Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Stage 1    │→ │   Stage 2    │→ │   Stage 3    │         │
│  │  SFT Train   │  │  SFT Train   │  │  DPO Train   │         │
│  │  (3 epochs)  │  │  (1 epoch)   │  │  (1 epoch)   │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
    Checkpoint 1       Checkpoint 2       Final Model
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Evaluation & Analysis Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Metrics    │  │  Baseline    │  │   Error      │         │
│  │ Calculation  │  │  Comparison  │  │  Analysis    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Component Layers

#### 1. **Data Pipeline Layer**
Handles all data ingestion, validation, and preprocessing.

**Key Components:**
- **Code Validation System**: Ensures all medical codes are valid
- **Data Collection**: Streams and merges datasets
- **Quality Checks**: Validates data format and content
- **Synthetic Generation**: Creates synthetic training data (future)

#### 2. **Storage Layer**
Manages code databases, datasets, and configurations.

**Key Components:**
- **Code Databases**: ICD-10, CPT, HCPCS reference data
- **Dataset Storage**: Processed training datasets
- **Configuration Files**: YAML configs for all stages
- **Model Checkpoints**: Saved model states

#### 3. **Training Layer**
Executes the three-stage training pipeline.

**Key Components:**
- **SFT Trainer**: Supervised fine-tuning (Stages 1 & 2)
- **DPO Trainer**: Direct preference optimization (Stage 3)
- **Checkpoint Manager**: Saves/loads model states
- **Metrics Logger**: Tracks training progress

#### 4. **Evaluation Layer**
Measures model performance and analyzes errors.

**Key Components:**
- **Metrics Calculator**: Accuracy, precision, recall, F1
- **Baseline Comparisons**: Compare against simple baselines
- **Error Analyzer**: Identify failure patterns
- **Report Generator**: Create evaluation reports

---

## Technology Stack

### Core Framework
```yaml
Language: Python 3.12+
ML Framework: PyTorch 2.0+
Training Library: Hugging Face Transformers + TRL
Dataset Management: Hugging Face Datasets
```

### Key Libraries

#### Training & Model
```python
torch>=2.7.0                # Deep learning framework
transformers==4.56.2        # Model architectures & training
trl==0.23.0                # Transformer RL (SFT, DPO)
peft==0.17.1               # Parameter-efficient fine-tuning
accelerate==1.10.1         # Distributed training utilities
```

#### Data Processing
```python
datasets==4.0.0            # Dataset loading & processing
pyyaml==6.0.2             # Configuration management
python-dotenv==1.1.1      # Environment variable management
```

#### Monitoring & Logging
```python
wandb==0.22.0             # Experiment tracking
loguru==0.7.3             # Logging
```

#### Utilities
```python
huggingface-hub==0.34.4   # Model & dataset hosting
immutabledict>=4.2.1      # Immutable data structures
langdetect>=1.0.9         # Language detection
```

#### Development & Testing
```python
pytest                     # Testing framework
ruff                      # Linting & formatting
pre-commit                # Git hooks
ipykernel                 # Jupyter notebook support
```

### Hardware Requirements

#### Minimum (Development)
- **CPU**: 8-core modern processor
- **RAM**: 16GB
- **Storage**: 50GB SSD
- **GPU**: None (for script development)

#### Recommended (Training)
- **GPU**: RTX 3090 (24GB VRAM) or equivalent
- **RAM**: 32GB
- **Storage**: 100GB NVMe SSD
- **Compute**: ~50 GPU-hours for full training

#### Cloud Alternatives
```yaml
Options:
  - AWS: p3.2xlarge (V100)
  - Lambda Labs: 1x RTX 3090
  - RunPod: A5000 / A6000
  - Vast.ai: RTX 3090 / 4090

Estimated Cost: $1.50/hour × 50 hours = $75
```

---

## Data Flow

### Training Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    Data Flow Diagram                          │
└──────────────────────────────────────────────────────────────┘

1. RAW DATA SOURCES
   ├─ MIMIC-III: clinical_notes.csv
   ├─ MTSamples: transcriptions.json
   └─ Synthetic: generated_examples.jsonl

2. CODE VALIDATION
   Raw Data → Code Validator → Validated Data
   ├─ Check: Is ICD-10 code in official database?
   ├─ Check: Is CPT code valid?
   ├─ Check: Is HCPCS code valid?
   └─ Output: data_with_valid_codes.jsonl

3. DATA PREPROCESSING
   Validated Data → Preprocessor → Formatted Data
   ├─ Normalize: Standardize text format
   ├─ Clean: Remove PHI, fix encoding
   ├─ Structure: Convert to chat format
   └─ Output: stage_1_formatted.jsonl

4. DATA COLLECTION (MERGE & SHUFFLE)
   Formatted Data → Data Collector → Final Dataset
   ├─ Merge: Combine all sources
   ├─ Shuffle: Randomize order (seed=42)
   ├─ Add Metadata: Track source info
   └─ Output: stage_1_final.parquet

5. TRAINING CONSUMPTION
   Final Dataset → DataLoader → Training Batches
   ├─ Tokenize: Convert text to tokens
   ├─ Batch: Group into batches (size=32)
   ├─ Collate: Pad sequences
   └─ Feed to Model
```

### Inference Data Flow (Future)

```
User Input → Tokenizer → Model → Decoder → Output Parser → Result

Example:
"Patient has hypertension"
  → [tokens]
  → Model Forward Pass
  → Generated Tokens
  → "<think>High blood pressure, essential hypertension</think>ICD10: I10"
  → {"code": "I10", "reasoning": "...", "confidence": 0.95}
```

---

## Model Architecture

### Base Model: SmolLM2-135M-Instruct

```
Model Specifications:
┌────────────────────────────────────────────┐
│ Parameters:       135 Million              │
│ Architecture:     Transformer Decoder      │
│ Context Length:   4096 tokens              │
│ Vocabulary Size:  ~49,152 tokens           │
│ Hidden Size:      576                      │
│ Layers:           30                       │
│ Attention Heads:  9                        │
│ Training Dtype:   bfloat16 / float16       │
│ Model Size:       ~500MB                   │
└────────────────────────────────────────────┘
```

### Tokenization Strategy

```python
# Base tokenizer + Special tokens
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

# Add medical coding special tokens
SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "<think>",      # Start reasoning
        "</think>",     # End reasoning
    ]
}

tokenizer.add_special_tokens(SPECIAL_TOKENS)
model.resize_token_embeddings(len(tokenizer))
```

### Chat Template Format

```python
MEDICAL_CODING_TEMPLATE = """<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>"""

# Stage 1 System Message
"You are a medical coding assistant. Convert clinical narratives to billing codes."

# Stage 2 System Message
"You are a medical coding expert. Use <think> tags to show your reasoning."
```

### Training Configuration

#### Stage 1: Non-Reasoning SFT
```yaml
epochs: 3
batch_size: 32
gradient_accumulation: 4
learning_rate: 3e-4
lr_scheduler: cosine
weight_decay: 0.01
warmup_ratio: 0.1
max_seq_length: 4096
neftune_noise_alpha: 0.01
gradient_checkpointing: true
bf16: true
```

#### Stage 2: Reasoning SFT
```yaml
epochs: 1
batch_size: 32
gradient_accumulation: 4
learning_rate: 3e-4
lr_scheduler: cosine
weight_decay: 0.02
warmup_ratio: 0.1
max_seq_length: 4096
neftune_noise_alpha: 0.02
gradient_checkpointing: true
bf16: true
```

#### Stage 3: DPO
```yaml
epochs: 1
batch_size: 32
gradient_accumulation: 4
learning_rate: 1e-5
lr_scheduler: cosine
beta: 0.1  # DPO temperature
loss_type: apo_zero
max_grad_norm: 0.2
gradient_checkpointing: true
bf16: true
```

---

## Component Overview

### Data Pipeline Components

#### 1. Code Validator (`utils/code_validator.py`)
```python
Purpose: Validate medical codes against official databases
Input:   Code string + code type
Output:  Boolean (valid/invalid) + metadata
Example: validate_code("E11.9", "ICD10") → True
```

#### 2. Data Collector (`data/data_collection.py`)
```python
Purpose: Stream and merge multiple datasets
Input:   YAML config specifying sources
Output:  Merged, shuffled dataset (parquet/arrow)
Features:
  - Streaming to handle large datasets
  - Memory-efficient chunking
  - Source tracking metadata
```

#### 3. Medical Preprocessor (`data/preprocess_medical.py`)
```python
Purpose: Clean and format medical narratives
Input:   Raw medical text + codes
Output:  Formatted chat messages
Features:
  - PHI removal patterns
  - Text normalization
  - Code format standardization
```

#### 4. Synthetic Generator (`data/synthetic_generation.py`)
```python
Purpose: Generate synthetic training data (future use)
Input:   Prompt template + LLM API
Output:  Synthetic examples in correct format
Features:
  - GPT-4 / Claude integration
  - Cost tracking
  - Quality validation
```

### Training Components

#### 1. SFT Trainer (`post_training/sft.py`)
```python
Purpose: Supervised fine-tuning (Stages 1 & 2)
Input:   Base model + dataset + config
Output:  Fine-tuned checkpoint
Features:
  - Chat template application
  - Special token handling
  - Checkpoint management
  - W&B logging
```

#### 2. DPO Trainer (`post_training/dpo.py`)
```python
Purpose: Preference alignment (Stage 3)
Input:   Stage 2 model + preference pairs
Output:  Aligned final model
Features:
  - Chosen/rejected pair handling
  - Reward margin tracking
  - Advanced DPO objectives
```

### Evaluation Components

#### 1. Metrics Calculator (`evaluation/metrics.py`)
```python
Purpose: Calculate medical coding metrics
Metrics:
  - Exact Match Accuracy
  - Top-K Accuracy
  - Precision / Recall / F1
  - Code Type Breakdown
  - Reasoning Quality Score
```

#### 2. Model Evaluator (`evaluation/evaluate_model.py`)
```python
Purpose: End-to-end model evaluation
Input:   Model + test dataset
Output:  Comprehensive evaluation report
Features:
  - Batch inference
  - Error categorization
  - Statistical significance tests
```

#### 3. Baseline Comparator (`evaluation/baseline.py`)
```python
Purpose: Compare against simple baselines
Baselines:
  - Random code selection
  - Keyword matching
  - Most frequent code
  - Rule-based system
```

### Utility Components

#### 1. Chat Templates (`utils/medical_chat_templates.py`)
```python
Purpose: Define chat format templates
Templates:
  - stage_1_template (no reasoning)
  - stage_2_template (with <think> tags)
  - stage_3_template (preference format)
```

#### 2. Data Quality (`utils/data_quality.py`)
```python
Purpose: Validate data quality
Checks:
  - Schema validation
  - Code validity
  - Text quality (length, language)
  - Reasoning presence (Stage 2+)
  - Preference pair consistency (Stage 3)
```

---

## Integration Points

### External Services Integration

#### Hugging Face Hub
```python
# Dataset hosting
dataset.push_to_hub("username/thce-stage-1-data")

# Model hosting
model.push_to_hub("username/thce-model-stage-1")

# Loading
dataset = load_dataset("username/thce-stage-1-data")
model = AutoModelForCausalLM.from_pretrained("username/thce-model-stage-1")
```

#### Weights & Biases
```python
# Training tracking
import wandb
wandb.init(project="thce-training", name="stage-1-run-1")

# Log metrics
wandb.log({"train/loss": loss, "train/accuracy": acc})
```

#### OpenAI / Anthropic (Future)
```python
# Synthetic data generation
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

### File System Integration

```
Data Flow on Disk:
├─ data/
│  ├─ raw/                     # Downloaded raw data
│  ├─ processed/               # Cleaned & formatted
│  │  ├─ stage_1/
│  │  ├─ stage_2/
│  │  └─ stage_3/
│  └─ artefacts/              # Final datasets
│     ├─ stage_1_final/
│     ├─ stage_2_final/
│     └─ stage_3_final/
│
├─ outputs/                    # Training outputs
│  ├─ stage_1/
│  │  ├─ checkpoint-1000/
│  │  ├─ checkpoint-2000/
│  │  └─ final/
│  ├─ stage_2/
│  └─ stage_3/
│
└─ results/                    # Evaluation results
   ├─ stage_1_eval.json
   ├─ stage_2_eval.json
   └─ stage_3_eval.json
```

---

## Performance Considerations

### Memory Optimization
```python
Techniques:
1. Gradient Checkpointing: Save memory during backprop
2. Mixed Precision (bf16): Reduce memory by 2x
3. Gradient Accumulation: Simulate larger batches
4. Streaming Datasets: Don't load all data into RAM
```

### Speed Optimization
```python
Techniques:
1. DataLoader Workers: Parallel data loading
2. Persistent Workers: Reuse worker processes
3. Pin Memory: Faster CPU→GPU transfer
4. Compiled Model: PyTorch 2.0 compile
```

### Cost Optimization
```python
Techniques:
1. Use smaller models for development
2. Early stopping to avoid overtraining
3. Cloud spot instances for training
4. Batch synthetic data generation
```

---

## Security & Privacy

### PHI Handling
```python
# All medical narratives must be:
1. De-identified (no patient names, DOB, MRN)
2. Compliant with HIPAA regulations
3. Scrubbed for sensitive information

# Automated checks in preprocessing:
- Remove patterns: SSN, phone, email
- Redact dates to year only
- Remove facility/doctor names
```

### Code Security
```python
# Environment variables for secrets
# .env file (not committed to git)
HF_TOKEN=hf_...
WANDB_API_KEY=...
OPENAI_API_KEY=sk-...

# Load in code
from dotenv import load_dotenv
load_dotenv()
```

---

## Monitoring & Observability

### Training Monitoring
```python
Metrics to Track:
- Loss (training & validation)
- Token Accuracy
- Gradient Norms
- Learning Rate
- GPU Utilization
- Training Speed (samples/sec)
```

### Data Monitoring
```python
Quality Checks:
- Code validity rate
- Average narrative length
- Language distribution
- Missing fields
- Duplicate rate
```

### Model Monitoring (Future)
```python
Production Metrics:
- Inference latency
- Prediction accuracy
- Code distribution
- Error rate by code type
- User feedback scores
```

---

## Scalability Considerations

### Current Scale (Phase 1)
- Datasets: 50-80k examples per stage
- Model: 135M parameters
- Training: Single GPU
- Storage: ~100GB

### Future Scale (Phase 2+)
- Datasets: 200k+ examples
- Model: 250M-1B parameters
- Training: Multi-GPU
- Storage: ~500GB

### Horizontal Scaling
```python
# Multi-GPU training (future)
accelerate launch --num_processes 4 post_training/sft.py

# Distributed data processing
from datasets import load_dataset
dataset = load_dataset(..., num_proc=16)
```

---

## Extension Points

### Easy to Add
1. New code types (NDC, DRG, etc.)
2. Additional data sources
3. New evaluation metrics
4. More baseline models

### Requires Modification
1. Multi-code outputs
2. Code hierarchy awareness
3. Real-time inference API
4. Multi-language support

### Architectural Changes
1. Ensemble models
2. Retrieval-augmented generation
3. Multi-modal inputs (images, PDFs)
4. Active learning loop

---

## References

### Official Documentation
- SmolLM2: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- TRL Library: https://huggingface.co/docs/trl/
- ICD-10-CM: https://www.cms.gov/medicare/coordination-benefits-recovery/overview/icd-code-information
- CPT Codes: https://www.ama-assn.org/practice-management/cpt

### Research Papers
- DPO: https://arxiv.org/abs/2305.18290
- Constitutional AI: https://arxiv.org/abs/2212.08073
- trlm Technical Report: (see project README)

---

**Last Updated:** 2025-10-30
**Maintainer:** THCE Project Team
**Status:** ✅ Architecture Finalized

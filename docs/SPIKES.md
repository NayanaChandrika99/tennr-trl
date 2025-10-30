# THCE Feasibility Spikes
**Time-boxed validation of core assumptions before full implementation**

---

## Overview

**Total Time:** 7 hours
**Goal:** Validate riskiest assumptions before building full system
**Status:** ⬜ Not Started

**Decision Gate:** All 4 spikes must pass to proceed with current architecture.

---

## Spike 1: Model Runtime & Generation

**Time Box:** 2 hours
**File:** `scripts/spike_model.py`

### Assumptions to Test
1. SmolLM2-135M loads in target environment
2. Can add special tokens (`<think>`, `</think>`)
3. Can apply medical chat template
4. Generates coherent medical code outputs
5. Latency is acceptable (<200ms on GPU)

### Implementation

```python
"""Spike: Validate model loading and generation."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

def spike_model():
    print("=== Spike 1: Model Runtime & Generation ===\n")

    # Test 1: Load model
    print("Test 1: Loading SmolLM2-135M...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    load_time = time.time() - start
    print(f"✓ Model loaded in {load_time:.2f}s")
    print(f"  Device: {model.device}")
    print(f"  Dtype: {model.dtype}")
    print(f"  Params: {model.num_parameters():,}\n")

    # Test 2: Add special tokens
    print("Test 2: Adding special tokens...")
    original_vocab = len(tokenizer)
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<think>", "</think>"]
    })
    model.resize_token_embeddings(len(tokenizer))
    print(f"✓ Vocab size: {original_vocab} → {len(tokenizer)}")
    print(f"  <think> token ID: {tokenizer.convert_tokens_to_ids('<think>')}")
    print(f"  </think> token ID: {tokenizer.convert_tokens_to_ids('</think>')}\n")

    # Test 3: Apply chat template
    print("Test 3: Chat template application...")
    messages = [
        {"role": "system", "content": "You are a medical coding assistant."},
        {"role": "user", "content": "Patient has Type 2 diabetes without complications."},
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"✓ Template applied:")
    print(f"{formatted[:200]}...\n")

    # Test 4: Generate response
    print("Test 4: Generation test...")
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    start = time.time()
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )
    gen_time = time.time() - start

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"✓ Generated in {gen_time:.3f}s")
    print(f"  Tokens: {outputs.shape[1]}")
    print(f"  Tokens/sec: {outputs.shape[1]/gen_time:.1f}")
    print(f"\nResponse:\n{response}\n")

    # Test 5: Latency benchmark
    print("Test 5: Latency benchmark (10 runs)...")
    latencies = []
    for i in range(10):
        start = time.time()
        outputs = model.generate(inputs, max_new_tokens=50, do_sample=False)
        latencies.append(time.time() - start)

    print(f"✓ p50 latency: {sorted(latencies)[5]:.3f}s")
    print(f"  p90 latency: {sorted(latencies)[9]:.3f}s")
    print(f"  Mean: {sum(latencies)/len(latencies):.3f}s\n")

    # Decision
    p50_latency = sorted(latencies)[5]
    decision = "PASS" if p50_latency < 0.2 else "FAIL"

    print(f"=== Result: {decision} ===")
    if decision == "FAIL":
        print(f"  Reason: p50 latency {p50_latency:.3f}s > 0.2s threshold")

    return {
        "load_time": load_time,
        "p50_latency": p50_latency,
        "tokens_per_sec": outputs.shape[1]/gen_time,
        "decision": decision
    }

if __name__ == "__main__":
    results = spike_model()
    print(f"\nResults: {results}")
```

### Success Criteria
- ✓ Model loads in <30s
- ✓ Special tokens add successfully
- ✓ Chat template formats correctly
- ✓ Generation produces text
- ✓ p50 latency <200ms (GPU) or <2s (CPU)

### Go/No-Go
- **PASS:** Proceed with SmolLM2-135M
- **FAIL:** Consider alternatives (Qwen2.5-0.5B, Phi-2)

---

## Spike 2: Code Validation Logic

**Time Box:** 1 hour
**File:** `scripts/spike_validator.py`

### Assumptions to Test
1. Can validate ICD-10 code format (alphanumeric with dot)
2. Can validate CPT code format (5 digits)
3. Can validate HCPCS code format (letter + 4 digits)
4. Code lookup works with sample database
5. Performance is acceptable (<1ms per code)

### Implementation

```python
"""Spike: Validate medical code validation logic."""

import re
import json
import time
from pathlib import Path

# Sample code database (minimal)
SAMPLE_CODES = {
    "ICD10": {
        "E11.9": "Type 2 diabetes mellitus without complications",
        "I10": "Essential (primary) hypertension",
        "J20.9": "Acute bronchitis, unspecified",
        "E11.65": "Type 2 diabetes mellitus with hyperglycemia",
    },
    "CPT": {
        "99213": "Office visit, established patient, 15-20 min",
        "99214": "Office visit, established patient, 25-40 min",
        "80053": "Comprehensive metabolic panel",
    },
    "HCPCS": {
        "E0130": "Walker, rigid, adjustable or fixed height",
        "E0143": "Walker, folding, wheeled, adjustable",
        "A4253": "Blood glucose test strips",
    }
}

def validate_code(code: str, code_type: str) -> dict:
    """Validate a medical code."""

    # Format validation
    patterns = {
        "ICD10": r"^[A-Z]\d{2}(\.\d{1,4})?$",
        "CPT": r"^\d{5}$",
        "HCPCS": r"^[A-Z]\d{4}$"
    }

    if code_type not in patterns:
        return {"valid": False, "error": f"Unknown code type: {code_type}"}

    if not re.match(patterns[code_type], code):
        return {"valid": False, "error": f"Invalid {code_type} format: {code}"}

    # Database lookup
    if code in SAMPLE_CODES.get(code_type, {}):
        return {
            "valid": True,
            "code": code,
            "code_type": code_type,
            "description": SAMPLE_CODES[code_type][code]
        }
    else:
        return {"valid": False, "error": f"Code not in database: {code}"}

def spike_validator():
    print("=== Spike 2: Code Validation Logic ===\n")

    # Test cases
    test_cases = [
        ("E11.9", "ICD10", True),
        ("I10", "ICD10", True),
        ("E11.65", "ICD10", True),
        ("INVALID", "ICD10", False),
        ("E11", "ICD10", False),  # Missing decimal
        ("99213", "CPT", True),
        ("9921", "CPT", False),  # Too short
        ("E0130", "HCPCS", True),
        ("E013", "HCPCS", False),  # Too short
    ]

    results = []
    print("Running validation tests...\n")

    for code, code_type, expected_valid in test_cases:
        result = validate_code(code, code_type)
        actual_valid = result.get("valid", False)
        status = "✓" if actual_valid == expected_valid else "✗"

        print(f"{status} {code_type}:{code} → {result}")
        results.append(actual_valid == expected_valid)

    # Performance test
    print("\nPerformance test (1000 validations)...")
    start = time.time()
    for _ in range(1000):
        validate_code("E11.9", "ICD10")
    elapsed = time.time() - start
    per_code = elapsed / 1000

    print(f"✓ 1000 validations in {elapsed:.3f}s")
    print(f"  Per code: {per_code*1000:.3f}ms")

    # Decision
    all_pass = all(results)
    perf_pass = per_code < 0.001  # <1ms per code

    decision = "PASS" if all_pass and perf_pass else "FAIL"
    print(f"\n=== Result: {decision} ===")
    print(f"  Tests passed: {sum(results)}/{len(results)}")
    print(f"  Performance: {per_code*1000:.3f}ms < 1ms → {'✓' if perf_pass else '✗'}")

    return {
        "tests_passed": sum(results),
        "tests_total": len(results),
        "latency_ms": per_code * 1000,
        "decision": decision
    }

if __name__ == "__main__":
    results = spike_validator()
    print(f"\nResults: {results}")
```

### Success Criteria
- ✓ All test cases pass (9/9)
- ✓ Format validation works for all code types
- ✓ Database lookup returns descriptions
- ✓ Performance <1ms per code

### Go/No-Go
- **PASS:** Current validation design works
- **FAIL:** Need more sophisticated parsing (e.g., ICD-10 hierarchy)

---

## Spike 3: Data Pipeline & Tokenization

**Time Box:** 2 hours
**File:** `scripts/spike_data.py`

### Assumptions to Test
1. Medical narrative → chat format conversion works
2. Tokenization produces correct format
3. Average sequence length fits in context (4096 tokens)
4. TRL SFTTrainer accepts our data format
5. Can load and process 100 examples in <10s

### Implementation

```python
"""Spike: Validate data pipeline and tokenization."""

from transformers import AutoTokenizer
from datasets import Dataset
import json
import time

def create_example_data():
    """Create sample medical coding examples."""
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a medical coding assistant."},
                {"role": "user", "content": "Patient presents with Type 2 diabetes mellitus without complications."},
                {"role": "assistant", "content": "ICD10: E11.9 (Type 2 diabetes mellitus without complications)"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a medical coding assistant."},
                {"role": "user", "content": "Patient diagnosed with essential hypertension during routine checkup. Blood pressure 145/95. Started on medication."},
                {"role": "assistant", "content": "ICD10: I10 (Essential (primary) hypertension)"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a medical coding expert. Use <think> tags to show your reasoning."},
                {"role": "user", "content": "Patient requires walker for mobility after hip replacement surgery."},
                {"role": "assistant", "content": "<think>\n- Patient needs durable medical equipment (DME)\n- Walker is post-surgical mobility aid\n- HCPCS E-codes cover DME\n- E0130 is standard walker\n</think>\nHCPCS: E0130 (Walker, rigid, adjustable or fixed height)"}
            ]
        },
    ]

def spike_data():
    print("=== Spike 3: Data Pipeline & Tokenization ===\n")

    # Test 1: Load tokenizer
    print("Test 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<think>", "</think>"]
    })
    print(f"✓ Tokenizer loaded, vocab size: {len(tokenizer)}\n")

    # Test 2: Create dataset
    print("Test 2: Creating sample dataset...")
    examples = create_example_data()
    dataset = Dataset.from_list(examples)
    print(f"✓ Dataset created: {len(dataset)} examples\n")

    # Test 3: Apply chat template
    print("Test 3: Applying chat template...")
    formatted = tokenizer.apply_chat_template(
        examples[0]["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    print(f"✓ Chat template applied:")
    print(f"{formatted}\n")

    # Test 4: Tokenize examples
    print("Test 4: Tokenizing examples...")
    token_lengths = []

    for i, example in enumerate(examples):
        tokens = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            add_generation_prompt=False
        )
        token_lengths.append(len(tokens))
        print(f"  Example {i+1}: {len(tokens)} tokens")

    print(f"\n✓ Statistics:")
    print(f"  Mean: {sum(token_lengths)/len(token_lengths):.1f} tokens")
    print(f"  Max: {max(token_lengths)} tokens")
    print(f"  Min: {min(token_lengths)} tokens\n")

    # Test 5: Batch processing
    print("Test 5: Batch processing (100 examples)...")
    large_dataset = Dataset.from_list(examples * 34)  # ~100 examples

    start = time.time()
    tokenized = []
    for example in large_dataset:
        tokens = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            add_generation_prompt=False
        )
        tokenized.append(tokens)
    elapsed = time.time() - start

    print(f"✓ Processed {len(large_dataset)} examples in {elapsed:.2f}s")
    print(f"  Throughput: {len(large_dataset)/elapsed:.1f} examples/sec\n")

    # Test 6: Check reasoning tags
    print("Test 6: Validating reasoning tags...")
    has_think = any("<think>" in tokenizer.decode(t) for t in tokenized)
    print(f"✓ Reasoning tags present: {has_think}\n")

    # Decision
    max_length_ok = max(token_lengths) < 4096
    speed_ok = len(large_dataset)/elapsed > 10  # >10 examples/sec

    decision = "PASS" if max_length_ok and speed_ok else "FAIL"
    print(f"=== Result: {decision} ===")
    print(f"  Max length: {max(token_lengths)} < 4096 → {'✓' if max_length_ok else '✗'}")
    print(f"  Processing speed: {len(large_dataset)/elapsed:.1f} > 10 ex/s → {'✓' if speed_ok else '✗'}")

    return {
        "mean_tokens": sum(token_lengths)/len(token_lengths),
        "max_tokens": max(token_lengths),
        "throughput": len(large_dataset)/elapsed,
        "decision": decision
    }

if __name__ == "__main__":
    results = spike_data()
    print(f"\nResults: {results}")
```

### Success Criteria
- ✓ Chat template formats correctly
- ✓ Tokenization produces valid sequences
- ✓ Mean sequence length <1000 tokens
- ✓ Max sequence length <4096 tokens
- ✓ Processing speed >10 examples/sec
- ✓ Reasoning tags (`<think>`) tokenize correctly

### Go/No-Go
- **PASS:** Current data format works with TRL
- **FAIL:** Adjust chat template or truncation strategy

---

## Spike 4: Training Setup & Minimal Run

**Time Box:** 2 hours
**File:** `scripts/spike_train.py`

### Assumptions to Test
1. Can load model + tokenizer + dataset together
2. TRL SFTTrainer initializes without errors
3. Can run 10 training steps successfully
4. Loss decreases over 10 steps
5. GPU memory usage is acceptable
6. Can save and load checkpoint

### Implementation

```python
"""Spike: Validate training setup with minimal run."""

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import torch

def spike_train():
    print("=== Spike 4: Training Setup & Minimal Run ===\n")

    # Test 1: Load components
    print("Test 1: Loading model, tokenizer, dataset...")

    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<think>", "</think>"]
    })
    model.resize_token_embeddings(len(tokenizer))

    # Create tiny dataset
    examples = [
        {
            "messages": [
                {"role": "system", "content": "You are a medical coding assistant."},
                {"role": "user", "content": "Patient has Type 2 diabetes."},
                {"role": "assistant", "content": "ICD10: E11.9"}
            ]
        }
    ] * 10  # 10 examples

    dataset = Dataset.from_list(examples)

    print(f"✓ Components loaded:")
    print(f"  Model: {model.config.name_or_path}")
    print(f"  Dataset: {len(dataset)} examples")
    print(f"  Device: {model.device}")
    print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB\n" if torch.cuda.is_available() else "  CPU mode\n")

    # Test 2: Configure trainer
    print("Test 2: Configuring SFTTrainer...")

    training_args = SFTConfig(
        output_dir="./spike_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        logging_steps=1,
        max_steps=10,  # Only 10 steps
        save_steps=10,
        max_seq_length=512,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=False,  # Disable for speed
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print(f"✓ Trainer configured\n")

    # Test 3: Run 10 training steps
    print("Test 3: Running 10 training steps...")

    try:
        train_result = trainer.train()

        print(f"✓ Training completed successfully")
        print(f"  Steps: {train_result.metrics['train_steps']}")
        print(f"  Final loss: {train_result.metrics['train_loss']:.4f}")

        # Check if loss decreased
        # Get first and last loss from logs
        logs = trainer.state.log_history
        if len(logs) >= 2:
            first_loss = logs[0].get('loss', 0)
            last_loss = logs[-2].get('loss', 0)  # -1 is usually final summary
            loss_decreased = last_loss < first_loss
            print(f"  First loss: {first_loss:.4f}")
            print(f"  Last loss: {last_loss:.4f}")
            print(f"  Loss decreased: {loss_decreased}\n")
        else:
            loss_decreased = True
            print(f"  (Insufficient logs to compare)\n")

    except Exception as e:
        print(f"✗ Training failed: {e}\n")
        return {"decision": "FAIL", "error": str(e)}

    # Test 4: Save and load checkpoint
    print("Test 4: Checkpoint save/load...")

    try:
        trainer.save_model("./spike_output/checkpoint")

        # Try loading
        loaded_model = AutoModelForCausalLM.from_pretrained(
            "./spike_output/checkpoint"
        )

        print(f"✓ Checkpoint saved and loaded successfully\n")
        checkpoint_ok = True

    except Exception as e:
        print(f"✗ Checkpoint failed: {e}\n")
        checkpoint_ok = False

    # Test 5: GPU memory check
    if torch.cuda.is_available():
        print("Test 5: GPU memory usage...")
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"✓ Peak GPU memory: {max_memory:.2f}GB")
        memory_ok = max_memory < 20  # <20GB for RTX 3090 (24GB)
        print(f"  Under 20GB limit: {memory_ok}\n")
    else:
        memory_ok = True
        print("Test 5: CPU mode - memory check skipped\n")

    # Decision
    decision = "PASS" if loss_decreased and checkpoint_ok and memory_ok else "FAIL"

    print(f"=== Result: {decision} ===")
    print(f"  Training completed: ✓")
    print(f"  Loss decreased: {'✓' if loss_decreased else '✗'}")
    print(f"  Checkpoint works: {'✓' if checkpoint_ok else '✗'}")
    print(f"  Memory acceptable: {'✓' if memory_ok else '✗'}")

    # Cleanup
    import shutil
    shutil.rmtree("./spike_output", ignore_errors=True)

    return {
        "training_completed": True,
        "loss_decreased": loss_decreased,
        "checkpoint_ok": checkpoint_ok,
        "memory_ok": memory_ok,
        "decision": decision
    }

if __name__ == "__main__":
    results = spike_train()
    print(f"\nResults: {results}")
```

### Success Criteria
- ✓ All components load together
- ✓ Trainer initializes without errors
- ✓ 10 training steps complete
- ✓ Loss decreases
- ✓ GPU memory <20GB (or runs on CPU)
- ✓ Checkpoint save/load works

### Go/No-Go
- **PASS:** Training infrastructure works
- **FAIL:** Adjust batch size, gradient accumulation, or max_seq_length

---

## Feasibility Gate Summary

### Decision Matrix

| Spike | Criteria | Threshold | Pass/Fail |
|-------|----------|-----------|-----------|
| **1: Model** | p50 latency | <200ms (GPU) | ⬜ |
| | Loads successfully | True | ⬜ |
| **2: Validator** | Test pass rate | 100% (9/9) | ⬜ |
| | Latency per code | <1ms | ⬜ |
| **3: Data** | Max sequence length | <4096 tokens | ⬜ |
| | Processing speed | >10 ex/s | ⬜ |
| **4: Training** | Training completes | True | ⬜ |
| | Loss decreases | True | ⬜ |
| | GPU memory | <20GB | ⬜ |

### Overall Decision

```
IF all 4 spikes PASS:
  → Proceed with implementation plan (docs/tasks.md)
  → Current architecture is validated

IF any spike FAILS:
  → Document failure in ADR (Architecture Decision Record)
  → Adjust architecture/specs
  → Re-run failed spike
  → Update tasks.md with new approach
```

---

## Running the Spikes

### Quick Start

```bash
# Install dependencies first
uv sync

# Run all spikes
python scripts/spike_model.py > results/spike_1.log
python scripts/spike_validator.py > results/spike_2.log
python scripts/spike_data.py > results/spike_3.log
python scripts/spike_train.py > results/spike_4.log

# Review results
cat results/spike_*.log | grep "Result:"
```

### Individual Spikes

```bash
# Spike 1: Model (requires GPU/CPU)
python scripts/spike_model.py

# Spike 2: Validator (runs anywhere)
python scripts/spike_validator.py

# Spike 3: Data (runs anywhere)
python scripts/spike_data.py

# Spike 4: Training (requires GPU for realistic test)
python scripts/spike_train.py
```

### Expected Runtime

```
Spike 1: ~5-10 minutes (model download + tests)
Spike 2: ~10 seconds
Spike 3: ~30 seconds
Spike 4: ~5-10 minutes (10 training steps)

Total: ~15-25 minutes
```

---

## Outputs & Artifacts

### Files Generated

```
results/
├── spike_1.log              # Model test results
├── spike_2.log              # Validator test results
├── spike_3.log              # Data pipeline results
├── spike_4.log              # Training test results
└── spike_summary.json       # Combined results
```

### Summary Report Format

```json
{
  "timestamp": "2025-10-30T10:30:00Z",
  "spikes": {
    "model": {
      "decision": "PASS",
      "load_time": 12.3,
      "p50_latency": 0.127,
      "tokens_per_sec": 245.3
    },
    "validator": {
      "decision": "PASS",
      "tests_passed": 9,
      "latency_ms": 0.032
    },
    "data": {
      "decision": "PASS",
      "mean_tokens": 156.4,
      "max_tokens": 342,
      "throughput": 45.2
    },
    "training": {
      "decision": "PASS",
      "training_completed": true,
      "loss_decreased": true,
      "memory_ok": true
    }
  },
  "overall_decision": "PASS",
  "notes": "All spikes passed. Ready to proceed with implementation."
}
```

---

## Architecture Decision Records (ADRs)

If any spike fails, document the decision:

### ADR Template

```markdown
# ADR-001: [Title of Decision]

**Date:** 2025-10-30
**Status:** Accepted | Rejected | Superseded
**Spike:** [Which spike revealed this]

## Context
[What assumption was tested and failed]

## Decision
[What we're changing]

## Consequences
- **Positive:** [Benefits]
- **Negative:** [Tradeoffs]
- **Risks:** [New risks introduced]

## Alternatives Considered
1. [Alternative 1]
2. [Alternative 2]

## Implementation Impact
- [ ] Update specs
- [ ] Update configs
- [ ] Update tasks.md
- [ ] Re-run affected spikes
```

### Example ADR

```markdown
# ADR-001: Reduce Batch Size from 32 to 16

**Date:** 2025-10-30
**Status:** Accepted
**Spike:** Spike 4 (Training Setup)

## Context
Spike 4 revealed OOM errors at batch_size=32 on RTX 3090 (24GB).
Peak memory usage: 22.4GB > 20GB safe threshold.

## Decision
Change batch_size from 32 to 16 in all training configs.
Keep effective batch size by changing gradient_accumulation_steps from 4 to 8.

## Consequences
- **Positive:** Fits in GPU memory with headroom
- **Negative:** Training 5-10% slower due to more gradient accumulation
- **Risks:** None, effective batch size unchanged

## Implementation Impact
- [x] Update post_training/config/thce_stage_*.yaml
- [x] Update docs/spec.md training section
- [x] Re-run Spike 4 with new settings
```

---

## Next Steps After Spikes

### If All Pass (Expected)

```
Day 1-2: Run spikes → All pass
Day 3: Review specs, make minor adjustments
Day 4-7: Start Week 1 tasks (docs/tasks.md)
  └─ Build code validator, data quality checker, etc.
```

### If Any Fail (Contingency)

```
Day 1-2: Run spikes → 1+ fails
Day 3: Document failure in ADR
     → Redesign affected components
     → Update specs/configs
Day 4: Re-run failed spikes
Day 5-7: If pass, start Week 1 tasks
        If fail again, consider alternative approach
```

---

**Last Updated:** 2025-10-30
**Status:** Ready to Execute
**Estimated Time:** 7 hours (spikes) + 1 hour (documentation) = 8 hours total

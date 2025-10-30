# THCE Task Tracker

This document captures the engineering checklist for Phase 1 (repository readiness). Tasks are grouped by capability area with current status indicators.

Legend: ✅ Completed | 🟡 In Progress | ⬜ Not Started | ⏸️ Deferred

---

## 1. Core Utilities

| Task | Status | Notes |
| ---- | ------ | ----- |
| Implement `utils/code_validator.py` | ✅ | Supports ICD-10, CPT, HCPCS with CLI helper |
| Implement `utils/medical_chat_templates.py` | ✅ | Provides stage-aware prompts |
| Implement `utils/data_quality.py` | ✅ | Generates JSON quality reports |
| Implement `utils/tokenization.py` | ✅ | Adds `<think>`/code prefix tokens |
| Add unit tests for utilities | ✅ | `tests/test_code_validator.py`, `tests/test_data_quality.py` |

---

## 2. Data Pipeline

| Task | Status | Notes |
| ---- | ------ | ----- |
| Curate sample code databases | ✅ | JSON subsets in `data/code_databases/` |
| Build preprocessing script | ✅ | `data/preprocess_medical.py` with validation hooks |
| Create synthetic generator stub | ✅ | `data/synthetic_generation.py` for demos |
| Provide stage-aligned example datasets | ✅ | 10 records per stage under `data/examples/` |
| Adapt data configs for medical focus | 🟡 | `data/config/stage_*.yaml` still reference legacy sources |

---

## 3. Training & Alignment

| Task | Status | Notes |
| ---- | ------ | ----- |
| Review `post_training/sft.py` for THCE defaults | ✅ | Configs now point to THCE datasets and support dry-run/small-run |
| Review `post_training/dpo.py` workflow | ✅ | Updated for local datasets and tested on Stage 3 sample |
| Document training procedures | ✅ | `docs/TRAINING.md` |
| Prepare tokenizer workflow | ✅ | `utils/tokenization.py` usage documented |

---

## 4. Evaluation

| Task | Status | Notes |
| ---- | ------ | ----- |
| Implement metric helpers (`evaluation/metrics.py`) | ✅ | Accuracy, validity, reasoning heuristics |
| Implement evaluation runner | ✅ | `evaluation/evaluate_model.py` with baseline integration |
| Implement baseline strategies | ✅ | Echo, random, majority baselines |
| Draft evaluation documentation | ✅ | `docs/EVALUATION.md` outlines approach |
| Add report analysis helper | ✅ | `evaluation/analysis.py` for quick inspection |

---

## 5. Documentation

| Task | Status | Notes |
| ---- | ------ | ----- |
| Update README for THCE | ✅ | Complete rewrite |
| Create setup guide | ✅ | `docs/SETUP.md` |
| Refresh development guide | ✅ | `docs/development.md` |
| Document data preparation | ✅ | `docs/DATA_PREPARATION.md` |
| Document testing strategy | ✅ | `docs/testing.md` |
| Update system overview | ✅ | Added Phase 1 note |
| Maintain roadmap (`docs/REVISED_PLAN.md`) | 🟡 | Continues to track pending work |

---

## 6. Testing & CI

| Task | Status | Notes |
| ---- | ------ | ----- |
| Add pytest suite | ✅ | 9 tests covering utilities/pipeline |
| Configure CI workflow | 🟡 | `uv-ci.yaml` exists; may require medical-specific checks |
| Configure pre-commit hooks | ✅ | `.pre-commit-config.yaml` ready |
| Track coverage metrics | ⏸️ | Defer until evaluation stack lands |

---

## 7. Next Actions

1. Update data and training configs to reference THCE assets. ✅ configs now reference `data/processed/*.jsonl`.
2. Use `scripts/generate_predictions.py` + `evaluation/evaluate_model.py` to produce Stage 2/3 reports once models are trained.
3. Extend synthetic generation with richer prompts and connect an API client inside `scripts/generate_synthetic.py` when budget is available.
4. Integrate dataset quality checks into CI once large datasets land.

This tracker should be updated as new commits land or priorities shift.

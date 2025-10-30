# THCE Testing Guide

This guide explains how automated tests are organised and how to extend coverage as the THCE project grows.

---

## 1. Philosophy

- **Fast feedback** – keep the suite light enough to run before every commit.
- **Confidence over coverage** – prioritise tests that protect critical behaviour (data validation, preprocessing, configs).
- **Reusable fixtures** – centralise sample data so it mirrors production formats.
- **Documented workflows** – tests double as executable examples of how to use scripts and utilities.

---

## 2. Current Test Layout

```
tests/
├── __init__.py
├── fixtures/
│   └── raw_records.jsonl           # Sample raw records used in preprocessing tests
├── test_code_validator.py          # Unit tests for utils.code_validator
├── test_data_quality.py            # Unit tests for utils.data_quality
├── test_metrics.py                 # Unit tests for evaluation metrics
└── test_preprocessing.py           # Integration tests for data.preprocess_medical
```

Future additions:
- `test_tokenization.py` for tokenizer utilities.
- `test_training.py` for config validation and dry-run checks.
- `test_evaluation.py` covering evaluation metrics.

---

## 3. Running The Suite

```bash
uv run --group dev pytest
```

Useful options:
- `pytest -k validator` – run tests matching a keyword.
- `pytest -vv` – verbose output.
- `pytest --maxfail=1` – stop after first failure (fast iteration).

CI mirrors the same command via `.github/workflows/uv-ci.yaml`.

---

## 4. Fixtures & Sample Data

- `tests/fixtures/raw_records.jsonl` – two raw records that exercise ICD-10 and HCPCS flows.
- `data/examples/sample_stage_*.jsonl` – authoritative examples for verifying dataset format.
- `data/code_databases/*.json` – shared validation source for tests and production code.

When adding new datasets or schemas, update fixtures in lockstep to keep tests meaningful.

---

## 5. Patterns & Recommendations

### Utility Modules
- Keep functions pure where possible to simplify testing.
- Use the shared `CodeValidator` cache to avoid I/O in tests.
- Mock network-dependent functions when real calls are required (e.g., future synthetic generation via APIs).

### Scripts
- Structure CLI entry points around functions that accept parameters (easy to call from tests).
- Use `tmp_path` fixtures and in-memory data rather than writing into the repo.

### Data Quality
- Extend `validate_dataset` with new checks and add regression tests whenever rules change.
- Capture the first few error messages in tests to confirm user-facing behaviour.

---

## 6. Coverage Targets (Phase 1)

| Area                  | Target |
| --------------------- | ------ |
| Utilities (`utils/`)  | 85%    |
| Data pipeline         | 80%    |
| Evaluation            | 70%    |
| Training (dry runs)   | 60%    |

The focus is to keep the tooling reliable while the model training phase is deferred.

---

## 7. Roadmap

- Add smoke tests for Stage 1/2/3 configs (`post_training/config`).
- Build evaluation regression tests once metrics are finalised.
- Integrate dataset quality checks into CI to prevent malformed JSONL files.
- Measure coverage with `pytest --cov` once the codebase stabilises.

Testing is the backbone of the “showcase-ready” repository—keep it current as new capabilities land.

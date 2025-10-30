"""
Utility to generate THCE model predictions for evaluation.

Loads a trained checkpoint, feeds a dataset (local jsonl or Hugging Face ID),
and writes predictions to JSONL with the schema expected by
`evaluation/evaluate_model.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_data(dataset_cfg: Dict[str, str]) -> Dataset:
    dataset_type = dataset_cfg.get("type", "huggingface").lower()
    if dataset_type == "local":
        data_format = dataset_cfg.get("format", "jsonl").lower()
        path = dataset_cfg["path"]
        if data_format not in {"jsonl", "json"}:
            raise ValueError(f"Unsupported local dataset format: {data_format}")
        return load_dataset(
            "json",
            data_files=path,
            split=dataset_cfg.get("split", "validation"),
        )
    return load_dataset(
        dataset_cfg["name"],
        dataset_cfg.get("subset"),
        split=dataset_cfg.get("split", "validation"),
    )


def generate(
    model: AutoModelForCausalLM,
    tokenizer,
    dataset: Dataset,
    max_new_tokens: int,
    temperature: float,
) -> Iterator[Dict[str, str]]:
    for sample in dataset:
        if "messages" in sample:
            prompt = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        elif "prompt" in sample:
            prompt = tokenizer.apply_chat_template(sample["prompt"], tokenize=False, add_generation_prompt=True)
        else:
            prompt = sample.get("narrative", "")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        yield {
            "prompt": prompt,
            "prediction": generated.strip(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate THCE predictions for evaluation.")
    parser.add_argument("--model-path", required=True, help="Path or HF ID of the trained model.")
    parser.add_argument("--tokenizer-path", help="Optional tokenizer path (defaults to model).")
    parser.add_argument(
        "--dataset",
        required=True,
        help="JSON string pointing to dataset config, e.g. '{\"type\": \"local\", \"path\": \"data/processed/stage_2_val.jsonl\"}'",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_cfg = json.loads(args.dataset)
    dataset = load_data(dataset_cfg)

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in generate(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

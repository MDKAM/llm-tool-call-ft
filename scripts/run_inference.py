#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.dataset_io import read_jsonl, truncate_rows, record_to_messages


def build_generation_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Use model chat template when available.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model_and_tokenizer(
    model_name_or_path: str,
    adapter_path: str | None = None,
    load_in_4bit: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    device_map = "auto"

    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    prompt_text = build_generation_prompt(tokenizer, messages)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0.0

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.95 if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/test.jsonl")
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter_path", type=str, default=None)
    ap.add_argument("--out_path", type=str, default="artifacts/model_predictions.jsonl")
    ap.add_argument("--max_examples", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.data_path)
    rows = truncate_rows(rows, args.max_examples)

    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path,
        adapter_path=args.adapter_path,
        load_in_4bit=not args.no_4bit,
    )

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_path, "w", encoding="utf-8") as f:
        for rec in tqdm(rows, desc="Running inference", ncols=100):
            messages = record_to_messages(rec, include_assistant=False)
            pred = generate_one(
                model,
                tokenizer,
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            f.write(json.dumps({"prediction": pred}, ensure_ascii=False) + "\n")

    print(f"Wrote predictions to {args.out_path}")


if __name__ == "__main__":
    main()
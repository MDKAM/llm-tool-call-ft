#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.demo_utils import (
    load_tools,
    build_demo_messages,
    pretty_print_tools,
    validate_prediction_text,
    mock_execute_calls,
)


def load_model_and_tokenizer(model_name_or_path: str, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quant_config,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_prediction(model, tokenizer, messages, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    tokenized = {k: v.to(model.device) for k, v in tokenized.items()}

    input_len = tokenized["input_ids"].shape[1]
    do_sample = temperature > 0.0

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    outputs = model.generate(
        **tokenized,
        **gen_kwargs,
    )

    gen_ids = outputs[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter_path", type=str, default=None)
    ap.add_argument("--tools_path", type=str, default="demo/sample_tools.json")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    tools = load_tools(args.tools_path)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path,
        adapter_path=args.adapter_path,
    )

    print("\n=== TOOL CATALOG ===")
    print(pretty_print_tools(tools))
    print("\nType a prompt and press Enter. Type 'exit' to quit.\n")

    while True:
        user_prompt = input("Prompt> ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in {"exit", "quit"}:
            break

        messages = build_demo_messages(user_prompt, tools)
        prediction = generate_prediction(
            model,
            tokenizer,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        print("\n=== MODEL OUTPUT ===")
        print(prediction)

        result, parsed = validate_prediction_text(prediction, tools, lenient_types=True)

        print("\n=== VALIDATION RESULT ===")
        print(json.dumps(result.__dict__, indent=2))

        if parsed is not None:
            print("\n=== PARSED CALLS ===")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))

            print("\n=== MOCK EXECUTION ===")
            for line in mock_execute_calls(parsed):
                print(line)

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
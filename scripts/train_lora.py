from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from src.dataset_io import read_jsonl, truncate_rows


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_example_text(tokenizer, record: Dict[str, Any]) -> str:
    messages = record["messages"]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def make_hf_dataset(records: List[Dict[str, Any]], tokenizer) -> Dataset:
    texts = [format_example_text(tokenizer, r) for r in records]
    return Dataset.from_dict({"text": texts})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/training_config.json")
    ap.add_argument("--train_path", type=str, default="data/train.jsonl")
    ap.add_argument("--val_path", type=str, default="data/val.jsonl")
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--clear_output_dir", action="store_true")
    ap.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg["output_dir"]

    if args.clear_output_dir and Path(output_dir).exists():
        shutil.rmtree(output_dir)

    train_rows = read_jsonl(args.train_path)
    val_rows = read_jsonl(args.val_path)

    train_rows = truncate_rows(train_rows, cfg.get("train_max_examples", 0))
    val_rows = truncate_rows(val_rows, cfg.get("val_max_examples", 0))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quant_config,
        dtype =torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["target_modules"],
    )

    model = get_peft_model(model, peft_config)

    train_ds = make_hf_dataset(train_rows, tokenizer)
    val_ds = make_hf_dataset(val_rows, tokenizer)

    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        report_to="none",
        save_total_limit=2,
        do_train=True,
        do_eval=True,
        dataset_text_field="text",
        max_length=cfg["max_seq_length"],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        packing=False,
        gradient_checkpointing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(Path(output_dir) / "used_training_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Saved adapter/tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
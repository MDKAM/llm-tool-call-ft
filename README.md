# Reliable LLM Tool Calling via LoRA Fine-Tuning 
**Reliable function calling with valid JSON, correct tool selection, and schema-faithful arguments**

## Overview
This project fine-tunes a small instruction LLM for **tool calling**.  
The goal is to improve three core behaviors:

1. **Choose the correct tool** from a provided tool catalog  
2. **Generate valid JSON** in a strict function-call format  
3. **Produce correct arguments** that match the selected tool schema  

The project includes:
- dataset validation and filtering
- preprocessing into chat-style JSONL
- strict + lenient output validation
- baseline and fine-tuned evaluation
- error analysis
- a simple interactive CLI demo for tool-calling inference

---

## Problem Statement
Modern LLM applications often rely on tool/function calling, but base models can fail in several ways:
- invalid JSON
- wrong tool selection
- missing or hallucinated arguments
- argument values that do not match schema requirements

This project frames tool calling as a supervised fine-tuning task and evaluates it with reproducible metrics.

---

## Dataset
**Dataset:** `Salesforce/xlam-function-calling-60k`

Each example contains:
- `query`: the user request
- `tools`: a list of available tools, each with:
  - `name`
  - `description`
  - `parameters`
- `answers`: one or more ground-truth tool calls with:
  - `name`
  - `arguments`

### Dataset gate checks
Before training, the dataset was verified for:
- number of unique tools
- parseable tools / answers
- multi-call coverage
- tool diversity

This ensured the dataset was suitable for real tool-calling fine-tuning.

---

## Output Format
The model is trained to generate **only** a JSON array:

```json
[
  {"name": "tool_name", "arguments": {...}}
]
```
## Project Pipeline

### 1. Dataset Gate

Validated the dataset for:

- Tool diversity
- Parseable tool specifications
- Parseable ground-truth calls

### 2. Output Validator

Implemented a dataset-aware validator that checks:

- Valid JSON
- Top-level array
- Exact call shape: `{"name", "arguments"}`
- Whether the tool exists in the provided catalog
- Whether arguments satisfy dataset-native parameter rules

### 3. Data Preprocessing

Converted the dataset into chat-format JSONL:

- **system**: strict tool-calling instructions + tool catalog
- **user**: query
- **assistant**: ground-truth JSON array

Generated:

- `data/train.jsonl`
- `data/val.jsonl`
- `data/test.jsonl`

### 4. Baseline Evaluation

Built an evaluation harness to compute:

- JSON parse rate
- Valid array rate
- Call-shape validity
- Strict valid rate
- Tool selection exact match
- Argument exact match
- Joint exact match
- Tool-name precision / recall / F1
- Argument-key precision / recall / F1

### 5. Fine-Tuning

Fine-tuned a small open model using LoRA on a free GPU setup.

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Method:** 4-bit loading + LoRA + SFT

### 6. Error Analysis

Grouped failures into:

- Wrong tool selection
- Wrong argument keys
- Wrong argument values
- Schema-invalid arguments

This made the final results interpretable instead of just reporting one accuracy number.

### 7. Demo

Built a simple CLI demo that:

- Loads a tool catalog
- Accepts a prompt
- Generates tool-call JSON
- Validates it
- Prints mock execution output

Also added an interactive CLI version that loads the model once and supports multiple prompts in one session.

---

## Key Scripts

### Data + Preprocessing

- `scripts/build_dataset.py` — build train/val/test JSONL
- `scripts/validate_output.py` — validate a predicted JSON string against a tool catalog
- `scripts/slice_jsonl.py` — create small subsets for quick evaluation

### Evaluation

- `scripts/evaluate_predictions.py` — compute main metrics
- `scripts/analyze_predictions.py` — per-tool metrics + error summaries
- `scripts/make_dummy_predictions.py` — oracle / empty baseline sanity checks

### Training + Inference

- `scripts/train_lora.py` — LoRA fine-tuning
- `scripts/run_inference.py` — run batch inference over a dataset split

### Demo

- `scripts/run_demo.py` — one-shot CLI demo
- `scripts/run_demo_chat.py` — interactive CLI demo

---

## How to Reproduce

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Build dataset
```bash
python -m scripts.build_dataset \
  --dataset Salesforce/xlam-function-calling-60k \
  --split train
```

### 3. Train
```bash
python -m scripts.train_lora \
  --config configs/training_config.json \
  --clear_output_dir
```

### 4. Run inference
```bash
python -m scripts.run_inference \
  --data_path data/test.jsonl \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/qwen2_5_1_5b_toolcall_lora \
  --out_path artifacts/finetuned_predictions.jsonl \
  --max_examples 100
```

### 5. Evaluate
```bash
python -m scripts.evaluate_predictions \
  --gold_path data/test.jsonl \
  --pred_path artifacts/finetuned_predictions.jsonl \
  --metrics_out artifacts/finetuned_metrics.json \
  --rows_out artifacts/finetuned_rows.jsonl \
  --lenient_types \
  --max_examples 100
```

### 6. Analyze errors
```bash
python -m scripts.analyze_predictions \
  --gold_path data/test.jsonl \
  --pred_path artifacts/finetuned_predictions.jsonl \
  --metrics_out artifacts/final_metrics.json \
  --per_tool_out artifacts/per_tool_metrics.json \
  --error_summary_out artifacts/error_analysis.json \
  --error_examples_out artifacts/error_examples.json \
  --lenient_types \
  --max_examples 100
```

### 7. Run demo
```bash
python -m scripts.run_demo_chat \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/qwen2_5_1_5b_toolcall_lora \
  --tools_path demo/sample_tools.json
```

## Example Result Snapshot

On a test set, the fine-tuned model achieved:

- **JSON parse rate:** `0.99`
- **Array rate:** `0.99`
- **Call shape rate:** `0.99`
- **Strict valid rate:** `0.99`
- **Tool name exact rate:** `0.98`
- **Argument exact rate:** `0.77`
- **Joint exact rate:** `0.77`
- **Tool-name F1:** `0.99`
- **Argument-key F1:** `0.96`

### Error Analysis

Error analysis on the test set with 1198 examples, showed:

- **Correct:** `920`
- **Wrong argument keys:** `103`
- **Wrong argument values:** `144`
- **Bad arguments:** `4`
- **Wrong tool selection:** `16`

This indicates the fine-tuned model learned JSON formatting and tool routing very well, while most remaining errors are concentrated in argument construction.

---

```md id="oxlfug"
## Results

| Model | JSON Parse | Valid Array | Call Shape | Strict Valid | Tool Exact | Arg Exact | Joint Exact | Tool F1 | Arg-Key F1 |
|------|------------:|------------:|-----------:|-------------:|-----------:|----------:|------------:|--------:|-----------:|
| Base model | 0.38 | 0.37 | 0.17 | 0.17 | 0.16 | 0.12 | 0.12 | 0.24 | 0.24 |
| Fine-tuned model | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 | 0.77 | 0.77 | 0.99 | 0.96 |

---

## Key Takeaways

- Fine-tuning substantially improved structured output reliability
- The model achieved near-perfect JSON validity and tool selection
- Remaining failures are mostly argument-level schema and value issues, not formatting failures
- The project provides a complete and reproducible pipeline from dataset gate to demo

<!-- ## Future Improvements

- Add constrained decoding for stricter schema adherence
- Compare against larger or instruction-tuned base models
- Add semantic argument normalization metrics
- Add a lightweight web demo -->
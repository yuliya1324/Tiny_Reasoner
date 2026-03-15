"""
Data utilities — loading and formatting GSM8K for different training methods.

GSM8K format:
  - question: "Natalia sold clips to 48 of her friends..."
  - answer: "Natalia sold 48/2 = <<48/2=24>>24 clips...\n#### 72"

We convert to our prompt format with <think>/<answer> tags.
"""

import os
import re
import json
from pathlib import Path
from datasets import load_dataset, Dataset

def _ensure_writable_hf_cache():
    """Use a HuggingFace cache under the project so we have write access (avoids PermissionError when /Data is read-only)."""
    cache_root = Path(__file__).resolve().parents[2] / ".cache" / "huggingface"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")

# ---------------------------------------------------------------------------
# Answer extraction from GSM8K format
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numerical answer from GSM8K's '#### N' format."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()


def extract_gsm8k_reasoning(answer_text: str) -> str:
    """Extract the reasoning steps (everything before ####)."""
    parts = answer_text.split("####")
    if len(parts) > 1:
        return parts[0].strip()
    return answer_text.strip()

import re

def compress_gsm8k_reasoning(reasoning: str) -> str:
    """
    Convert GSM8K reasoning into a compact equation-only form.

    Examples:
      "Natalia sold 48/2 = 24 clips in May." -> "48/2=24"
      "Betty has only 100 / 2 = $50." -> "100/2=50"
      "Working 50 minutes, she earned 0.2 x 50 = $10." -> "0.2x50=10"
    """
    lines = [line.strip() for line in reasoning.split("\n") if line.strip()]
    equations = []

    for line in lines:
        # Skip tags
        if line.startswith("<think>") or line.startswith("</think>") or line.startswith("<answer>"):
            continue

        # 1. Prefer GSM8K calculator annotations
        ann_matches = re.findall(r"<<(.*?)>>", line)
        if ann_matches:
            for eq in ann_matches:
                eq = eq.replace("$", "").replace(",", "")
                eq = eq.replace("×", "x").replace("X", "x")
                eq = re.sub(r"\s+", "", eq)
                equations.append(eq)
            continue

        clean = line.replace("$", "").replace(",", "").strip().rstrip(".")
        clean = clean.replace("×", "x").replace("X", "x")

        if "=" not in clean:
            continue

        left, right = clean.split("=", 1)

        # 2. Extract the last equation-like expression on the left
        # allow digits, decimals, parentheses, + - * / x
        # and capture something that actually looks like an expression
        candidates = re.findall(
            r'(?<![A-Za-z])(?:\d+(?:\.\d+)?|\(\s*[-+]?\d+(?:\.\d+)?\s*\))'
            r'(?:\s*[-+*/x]\s*(?:\d+(?:\.\d+)?|\(\s*[-+]?\d+(?:\.\d+)?\s*\)))*',
            left
        )
        if not candidates:
            continue

        lhs = candidates[-1]
        lhs = re.sub(r"\s+", "", lhs)

        # 3. Extract only the first numeric value on the right
        right_match = re.search(r"-?\d+(?:\.\d+)?", right)
        if not right_match:
            continue
        rhs = right_match.group(0)

        equations.append(f"{lhs}={rhs}")

    if not equations:
        short = re.sub(r"<<.*?>>", "", reasoning)
        short = re.sub(r"<.*?>", "", short)
        short = re.sub(r"\s+", " ", short).strip()
        return short

    return ", ".join(equations)

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve problems step by step. "
    "Put your reasoning inside <think>...</think> tags and your "
    "final numerical answer inside <answer>...</answer> tags."
)


def format_prompt(question: str) -> str:
    """Format a question into our standard prompt."""
    return (
        f"Solve this math problem step by step.\n"
        f"Put your reasoning inside <think>...</think> tags "
        f"and your final numerical answer inside <answer>...</answer> tags.\n\n"
        f"Problem: {question}"
    )


def format_chat_prompt(question: str, tokenizer) -> str:
    """Format using the model's chat template if available."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {question}"},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return format_prompt(question)


def format_sft_target(reasoning: str, answer: str) -> str:
    """Format the target completion for SFT training."""
    # Clean up GSM8K's calculator annotations like <<48/2=24>>
    clean_reasoning = re.sub(r"<<.*?>>", "", reasoning).strip()
    return f"<think>\n{clean_reasoning}\n</think>\n<answer>{answer}</answer>"

# ---------------------------------------------------------------------------
# DPO utilities
# ---------------------------------------------------------------------------
def make_wrong_answer(answer: str) -> str:
    
    import random
    from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

    """
    Create a minimally corrupted wrong numerical answer for DPO rejected samples.

    Rule:
      - Integer: ±max(round(10%), 1)
      - Decimal: ±max(10%, 0.1)
      - Randomly choose + or -
      - Fallback: append '_wrong'
    """
    answer = answer.strip().replace(",", "")

    try:
        value = Decimal(answer)

        # Integer-like answer
        if value == value.to_integral_value():
            offset = max(
                int((abs(value) * Decimal("0.1")).to_integral_value(rounding=ROUND_HALF_UP)),
                1,
            )
            sign = random.choice([-1, 1])
            wrong_value = int(value) + sign * offset
            return str(wrong_value)

        # Decimal answer
        offset = max(abs(value) * Decimal("0.1"), Decimal("0.1"))
        sign = random.choice([-1, 1])
        wrong_value = value + (Decimal(sign) * offset)

        return str(wrong_value.quantize(Decimal("0.01")).normalize())

    except (InvalidOperation, ValueError):
        if answer.endswith("_wrong"):
            return answer + "1"
        return answer + "_wrong"


def build_dpo_example(example, tokenizer):
    """
    Convert one GSM8K sample into a DPO example with:
      - prompt
      - chosen
      - rejected
    """
    question = example["question"]
    answer_text = example["answer"]

    reasoning = extract_gsm8k_reasoning(answer_text)
    answer = extract_gsm8k_answer(answer_text)

    prompt = format_chat_prompt(question, tokenizer)
    chosen = format_sft_target(reasoning, answer)
    rejected = format_sft_target(reasoning, make_wrong_answer(answer))

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_gsm8k(split: str = "train", max_samples: int = None) -> Dataset:
    """Load GSM8K dataset from HuggingFace."""
    _ensure_writable_hf_cache()
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def prepare_sft_dataset(tokenizer, split: str = "train", max_samples: int = None,
                         max_length: int = 512) -> Dataset:
    """
    Prepare GSM8K for SFT: each example becomes (prompt, completion) pair.

    Returns a Dataset with 'text' column containing the full sequence.
    """
    ds = load_gsm8k(split, max_samples)

    def format_example(example):
        question = example["question"]
        answer_text = example["answer"]

        reasoning = extract_gsm8k_reasoning(answer_text)
        answer = extract_gsm8k_answer(answer_text)

        prompt = format_chat_prompt(question, tokenizer)
        completion = format_sft_target(reasoning, answer)

        return {"text": prompt + completion, "prompt": prompt, "completion": completion}

    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds

def prepare_dpo_dataset(
    tokenizer,
    split: str = "train",
    max_samples: int = None,
) -> Dataset:
    """
    Prepare GSM8K for DPO.

    Returns a Dataset with:
      - prompt
      - chosen
      - rejected
      - answer
    """
    ds = load_gsm8k(split, max_samples)

    ds = ds.map(
        lambda example: build_dpo_example(example, tokenizer),
        remove_columns=ds.column_names,
    )
    return ds


import json
from datasets import Dataset

def load_jsonl_dataset(path: str) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)

def prepare_dpo_dataset_from_sft_outputs(path: str) -> Dataset:
    ds = load_jsonl_dataset(path)

    def keep_fields(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
            "question": example.get("question", ""),
            "gt_answer": example.get("gt_answer", ""),
        }

    ds = ds.map(keep_fields, remove_columns=ds.column_names)

    ds = ds.filter(
        lambda x: x["rejected"].strip() != "" and x["rejected"].strip() != x["chosen"].strip()
    )

    return ds

def build_dpo_example_short_chosen(example, tokenizer):
    question = example["question"]
    answer_text = example["answer"]

    full_reasoning = extract_gsm8k_reasoning(answer_text)
    answer = extract_gsm8k_answer(answer_text)

    short_reasoning = compress_gsm8k_reasoning(full_reasoning)

    prompt = format_chat_prompt(question, tokenizer)

    chosen = f"<think>\n{short_reasoning}\n</think>\n<answer>{answer}</answer>"
    rejected = format_sft_target(full_reasoning, make_wrong_answer(answer))

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "answer": answer,
    }

def prepare_dpo_dataset_short_chosen(
    tokenizer,
    split: str = "train",
    max_samples: int = None,
) -> Dataset:
    """
    Prepare GSM8K for DPO with:
      - chosen: compressed GT reasoning + correct answer
      - rejected: compressed GT reasoning + synthetic wrong answer
    """
    ds = load_gsm8k(split, max_samples)

    ds = ds.map(
        lambda example: build_dpo_example_short_chosen(example, tokenizer),
        remove_columns=ds.column_names,
    )
    return ds

def prepare_dpo_dataset_from_sft_outputs_short_chosen(path: str) -> Dataset:
    """
    DPO dataset:
      - chosen: compressed ground-truth reasoning + correct answer
      - rejected: SFT model output
    """

    ds = load_jsonl_dataset(path)

    def convert_example(example):
        chosen = example["chosen"]

        # parse chosen into reasoning and answer
        think_match = re.search(r"<think>\s*(.*?)\s*</think>", chosen, re.DOTALL)
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", chosen, re.DOTALL)

        reasoning = think_match.group(1).strip() if think_match else ""
        answer = answer_match.group(1).strip() if answer_match else ""

        short_reasoning = compress_gsm8k_reasoning(reasoning)
        short_chosen = f"<think>\n{short_reasoning}\n</think>\n<answer>{answer}</answer>"

        return {
            "prompt": example["prompt"],
            "chosen": short_chosen,
            "rejected": example["rejected"],
        }

    ds = ds.map(convert_example, remove_columns=ds.column_names)

    ds = ds.filter(
        lambda x: x["rejected"].strip() != "" and x["rejected"].strip() != x["chosen"].strip()
    )

    return ds

def prepare_grpo_dataset(split: str = "train", max_samples: int = None) -> Dataset:
    """
    Prepare GSM8K for GRPO: each example has a chat-message prompt + ground-truth answer.

    GRPOTrainer expects:
      - "prompt": list of message dicts (conversational format)
      - "answer": ground-truth number string (passed to reward functions via kwargs)

    Returns a HuggingFace Dataset.
    """
    ds = load_gsm8k(split, max_samples)

    def format_example(example):
        question = example["question"]
        answer = extract_gsm8k_answer(example["answer"])
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {question}"},
        ]
        return {"prompt": prompt, "answer": answer}

    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds

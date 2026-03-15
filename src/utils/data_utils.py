"""
Data utilities — loading and formatting GSM8K for different training methods.

GSM8K format:
  - question: "Natalia sold clips to 48 of her friends..."
  - answer: "Natalia sold 48/2 = <<48/2=24>>24 clips...\n#### 72"

We convert to our prompt format with <think>/<answer> tags.
"""

import os
import re
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

"""
Data utilities — loading and formatting GSM8K for different training methods.

GSM8K format:
  - question: "Natalia sold clips to 48 of her friends..."
  - answer: "Natalia sold 48/2 = <<48/2=24>>24 clips...\n#### 72"

We convert to our prompt format with <think>/<answer> tags.
"""

import re
from datasets import load_dataset, Dataset

import random
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


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
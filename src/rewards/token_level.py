"""
Token-level reward functions for PPO.

The key idea: parse the ground truth for intermediate calculations
(e.g. <<48/2=24>>), then scan the generated tokens left-to-right.
When a new correct calculation appears in the decoded prefix, reward
that token.

Usage with PPO:
    rewards = token_level_reward(generated_ids, tokenizer, ground_truth)
    # rewards is a list[torch.Tensor], one tensor per example
    ppo_trainer.step(query_tensors, response_tensors, rewards)
"""

import re
from typing import Optional

import torch

from src.utils.math_verify import (
    extract_answer_from_response,
    has_correct_format,
    check_answer,
    normalize_number,
)


# ---------------------------------------------------------------------------
# Ground truth parsing
# ---------------------------------------------------------------------------

def parse_intermediate_calculations(gt_text: str) -> list[str]:
    """
    Extract intermediate calculations from GSM8K ground truth.

    Input:  "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n..."
    Output: ["48/2=24", "48+24=72"]
    """
    return re.findall(r"<<(.+?)>>", gt_text)


def normalize_calculation(calc: str) -> Optional[tuple[str, float]]:
    """
    Parse a calculation string like '48/2=24' into (expression, result).
    Returns the result as a normalized number for flexible matching.
    """
    if "=" not in calc:
        return None
    parts = calc.rsplit("=", 1)
    expression = parts[0].strip()
    result = normalize_number(parts[1].strip())
    if result is None:
        return None
    return (expression, result)


def check_calculation_in_text(text: str, calc: str) -> bool:
    """
    Check if a calculation result appears in the text.

    We look for the result number near the expression or on its own.
    Flexible matching: '48/2 = 24', '48/2=24', '24', etc.
    """
    parsed = normalize_calculation(calc)
    if parsed is None:
        return False
    expression, result = parsed

    # Check if the result number appears in the text
    # Match as a standalone number (not part of a larger number)
    result_str = str(int(result)) if result == int(result) else str(result)
    pattern = r'(?<!\d)' + re.escape(result_str) + r'(?!\d)'

    if not re.search(pattern, text):
        return False

    # Also check that the expression or its components appear
    # (to avoid false positives from unrelated numbers)
    # Extract operands from the expression
    operands = re.findall(r'\d+\.?\d*', expression)
    if len(operands) >= 2:
        # At least 2 operands from the expression should appear in text
        operand_matches = sum(
            1 for op in operands
            if re.search(r'(?<!\d)' + re.escape(op) + r'(?!\d)', text)
        )
        return operand_matches >= min(2, len(operands))

    return True


# ---------------------------------------------------------------------------
# Token-level reward computation
# ---------------------------------------------------------------------------

def _compute_token_rewards(
    generated_ids: torch.Tensor,
    tokenizer,
    gt_text: str,
    calc_reward: float = 0.5,
    correct_answer_reward: float = 1.0,
    wrong_answer_reward: float = 0.0,
    format_reward: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute per-token rewards for a single generated sequence.

    Args:
        generated_ids: Token IDs of the generated response (1D tensor).
        tokenizer: The tokenizer for decoding.
        gt_text: The full GSM8K ground truth string (with <<>> annotations).
        calc_reward: Reward given when a new intermediate calculation is found.
        correct_answer_reward: Reward for the token closing </answer> if correct.
        wrong_answer_reward: Reward for the token closing </answer> if wrong.
        format_reward: If not None, give this bonus at the token where
                       <think> and </think> tags are completed.

    Returns:
        Tensor of shape (seq_len,) with per-token rewards.
    """
    seq_len = generated_ids.shape[0]
    rewards = torch.zeros(seq_len)

    # Parse ground truth
    calculations = parse_intermediate_calculations(gt_text)
    gt_answer = re.search(r"####\s*(.+)", gt_text)
    gt_answer = gt_answer.group(1).strip().replace(",", "") if gt_answer else None

    # Track which calculations have already been rewarded
    calc_found = [False] * len(calculations)

    # Decode all tokens at once, then build prefixes by accumulation
    # This avoids O(n²) decoding — we decode each token once
    token_strings = [
        tokenizer.decode([tid], skip_special_tokens=True)
        for tid in generated_ids
    ]

    prefix_text = ""
    for t in range(seq_len):
        prefix_text += token_strings[t]

        # --- Check intermediate calculations ---
        for i, calc in enumerate(calculations):
            if not calc_found[i] and check_calculation_in_text(prefix_text, calc):
                calc_found[i] = True
                rewards[t] += calc_reward

        # --- Check answer correctness at </answer> token ---
        if prefix_text.rstrip().endswith("</answer>"):
            predicted = extract_answer_from_response(prefix_text)
            if predicted is not None and gt_answer is not None:
                if check_answer(predicted, gt_answer):
                    rewards[t] += correct_answer_reward
                else:
                    rewards[t] += wrong_answer_reward

        # --- Format shaping (optional) ---
        if format_reward is not None:
            # Reward when <think> tag is opened
            if prefix_text.rstrip().endswith("<think>"):
                rewards[t] += format_reward * 0.25

            # Reward when </think> tag is closed
            if prefix_text.rstrip().endswith("</think>"):
                rewards[t] += format_reward * 0.25

            # Reward when <answer> tag is opened
            if prefix_text.rstrip().endswith("<answer>"):
                rewards[t] += format_reward * 0.25

            # Reward when </answer> tag is closed (format complete)
            if prefix_text.rstrip().endswith("</answer>"):
                rewards[t] += format_reward * 0.25

    return rewards


# ---------------------------------------------------------------------------
# Public API: two variants
# ---------------------------------------------------------------------------

def token_level_reward(
    generated_ids_batch: list[torch.Tensor],
    tokenizers,
    gt_texts: list[str],
    calc_reward: float = 0.5,
) -> list[torch.Tensor]:
    """
    Token-level reward: intermediate calculation checks + answer correctness.

    Args:
        generated_ids_batch: List of 1D tensors, one per example.
        tokenizer: The tokenizer.
        gt_texts: List of GSM8K ground truth strings (with <<>> annotations).

    Returns:
        List of reward tensors, one per example, each of shape (seq_len,).
    """
    return [
        _compute_token_rewards(
            gen_ids, tokenizer, gt,
            calc_reward=calc_reward,
            format_reward=None,
        )
        for gen_ids, tokenizer, gt in zip(generated_ids_batch, tokenizers, gt_texts)
    ]


def token_level_reward_with_format(
    generated_ids_batch: list[torch.Tensor],
    tokenizer,
    gt_texts: list[str],
    calc_reward: float = 0.5,
    format_reward: float = 0.2,
) -> list[torch.Tensor]:
    """
    Token-level reward: intermediate calculations + answer + format shaping.

    Same as token_level_reward but adds bonuses for using <think>/<answer> tags.

    Args:
        generated_ids_batch: List of 1D tensors, one per example.
        tokenizer: The tokenizer.
        gt_texts: List of GSM8K ground truth strings (with <<>> annotations).
        format_reward: Total format bonus (split across the 4 tag events).

    Returns:
        List of reward tensors, one per example, each of shape (seq_len,).
    """
    return [
        _compute_token_rewards(
            gen_ids, tokenizer, gt,
            calc_reward=calc_reward,
            format_reward=format_reward,
        )
        for gen_ids, gt in zip(generated_ids_batch, gt_texts)
    ]
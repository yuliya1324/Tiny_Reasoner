"""
Reward functions for GRPO/PPO training.

All reward functions take (response: str, ground_truth: str) -> float.
"""

from src.utils.math_verify import (
    extract_answer_from_response,
    extract_thinking_from_response,
    has_correct_format,
    check_answer,
)


def sparse_reward(response: str, ground_truth: str) -> float:
    """
    Sparse binary reward.
    reward = 1.0 if final answer is correct, 0.0 otherwise.
    """
    predicted = extract_answer_from_response(response)
    if predicted is None:
        return 0.0
    return 1.0 if check_answer(predicted, ground_truth) else 0.0


def format_shaped_reward(response: str, ground_truth: str,
                          format_bonus: float = 0.2, **kwargs) -> float:
    """
    Binary correctness + format bonus.
    reward = correctness (0 or 1) + format_bonus if <think>/<answer> used.
    """
    base = sparse_reward(response, ground_truth, **kwargs)
    fmt = format_bonus if has_correct_format(response) else 0.0
    return base + fmt


def composite_reward(response: str, ground_truth: str,
                      format_bonus: float = 0.2,
                      length_penalty_threshold: int = 300,
                      length_penalty_weight: float = 0.001,
                      **kwargs) -> float:
    """
    Format-shaped + length penalty.
    Penalizes excessively long responses to prevent reward hacking via verbosity.

    reward = correctness + format_bonus - length_penalty
    """
    base = format_shaped_reward(response, ground_truth, format_bonus, **kwargs)

    # Length penalty: kick in after threshold words
    thinking = extract_thinking_from_response(response)
    if thinking:
        word_count = len(thinking.split())
        excess = max(0, word_count - length_penalty_threshold)
        penalty = excess * length_penalty_weight
    else:
        penalty = 0.0

    return max(0.0, base - penalty)

"""
Math answer verification — extract and compare numerical answers.
"""

import re
from typing import Optional


def extract_answer_from_response(response: str) -> Optional[str]:
    """Extract numerical answer from <answer>...</answer> tags."""
    match = re.search(r"<answer>\s*(.+?)\s*</answer>", response, re.DOTALL)
    if match:
        answer = match.group(1).strip().replace(",", "")
        # Try to normalize to a number
        try:
            return str(float(answer))
        except ValueError:
            return answer
    return None


def extract_thinking_from_response(response: str) -> Optional[str]:
    """Extract reasoning from <think>...</think> tags."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def has_correct_format(response: str) -> bool:
    """Check if response uses <think>...</think><answer>...</answer> format."""
    has_think = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL))
    return has_think and has_answer


def normalize_number(s: str) -> Optional[float]:
    """Try to parse a string as a number."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


def check_answer(predicted: str, ground_truth: str, tolerance: float = 1e-5) -> bool:
    """
    Check if predicted answer matches ground truth.

    Handles: integers, floats, and string comparison as fallback.
    """
    pred_num = normalize_number(predicted)
    gt_num = normalize_number(ground_truth)

    if pred_num is not None and gt_num is not None:
        return abs(pred_num - gt_num) < tolerance

    # Fallback to string comparison
    return predicted.strip().lower() == ground_truth.strip().lower()


def verify_response(response: str, ground_truth: str) -> dict:
    """
    Full verification of a model response.

    Returns dict with:
        correct: bool — answer matches ground truth
        has_format: bool — uses <think>/<answer> tags
        has_reasoning: bool — non-trivial content in <think> tags
        predicted_answer: str or None
        reasoning_length: int — number of tokens/words in reasoning
    """
    predicted = extract_answer_from_response(response)
    thinking = extract_thinking_from_response(response)

    correct = False
    if predicted is not None:
        correct = check_answer(predicted, ground_truth)

    has_reasoning = thinking is not None and len(thinking.split()) > 5
    reasoning_length = len(thinking.split()) if thinking else 0

    return {
        "correct": correct,
        "has_format": has_correct_format(response),
        "has_reasoning": has_reasoning,
        "predicted_answer": predicted,
        "reasoning_length": reasoning_length,
    }

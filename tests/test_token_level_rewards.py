"""
Tests for token-level reward functions.

Run with: python -m pytest tests/test_token_level_rewards.py -v
"""

import torch
import pytest
from unittest.mock import MagicMock

from src.rewards.token_level import (
    parse_intermediate_calculations,
    normalize_calculation,
    check_calculation_in_text,
    _compute_token_rewards,
    token_level_reward,
    token_level_reward_with_format,
)


# ---------------------------------------------------------------------------
# Test ground truth parsing
# ---------------------------------------------------------------------------

class TestParseIntermediateCalculations:
    def test_basic(self):
        gt = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n#### 24"
        assert parse_intermediate_calculations(gt) == ["48/2=24"]

    def test_multiple(self):
        gt = (
            "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
            "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
            "#### 72"
        )
        assert parse_intermediate_calculations(gt) == ["48/2=24", "48+24=72"]

    def test_no_calculations(self):
        gt = "The answer is 42.\n#### 42"
        assert parse_intermediate_calculations(gt) == []

    def test_multiplication(self):
        gt = "She has 3*4 = <<3*4=12>>12 apples.\n#### 12"
        assert parse_intermediate_calculations(gt) == ["3*4=12"]

    def test_chained_calculations(self):
        gt = (
            "Step 1: 10+5 = <<10+5=15>>15.\n"
            "Step 2: 15*2 = <<15*2=30>>30.\n"
            "Step 3: 30-8 = <<30-8=22>>22.\n"
            "#### 22"
        )
        result = parse_intermediate_calculations(gt)
        assert result == ["10+5=15", "15*2=30", "30-8=22"]


# ---------------------------------------------------------------------------
# Test calculation normalization
# ---------------------------------------------------------------------------

class TestNormalizeCalculation:
    def test_basic(self):
        expr, result = normalize_calculation("48/2=24")
        assert expr == "48/2"
        assert result == 24.0

    def test_float_result(self):
        expr, result = normalize_calculation("10/3=3.33")
        assert expr == "10/3"
        assert abs(result - 3.33) < 0.01

    def test_no_equals(self):
        assert normalize_calculation("48+24") is None

    def test_non_numeric_result(self):
        assert normalize_calculation("x=hello") is None

    def test_multiple_equals(self):
        # "3+2=5=5" should split on last "="
        expr, result = normalize_calculation("3+2=5=5")
        assert expr == "3+2=5"
        assert result == 5.0


# ---------------------------------------------------------------------------
# Test calculation checking in text
# ---------------------------------------------------------------------------

class TestCheckCalculationInText:
    def test_exact_match(self):
        assert check_calculation_in_text("48/2 = 24 clips", "48/2=24")

    def test_no_spaces(self):
        assert check_calculation_in_text("48/2=24", "48/2=24")

    def test_result_missing(self):
        assert not check_calculation_in_text("48/2 equals something", "48/2=24")

    def test_operands_missing(self):
        # Result 24 is there but operands 48 and 2 aren't both present
        # (2 appears as part of 24, but not standalone)
        assert not check_calculation_in_text("The number is 24.", "48/2=24")

    def test_result_embedded_in_larger_number(self):
        # "240" contains "24" but shouldn't match
        assert not check_calculation_in_text("48 divided by 2 gives 240", "48/2=24")

    def test_with_surrounding_text(self):
        text = "First, 48 divided by 2 is 24. Then we add."
        assert check_calculation_in_text(text, "48/2=24")

    def test_addition(self):
        text = "48 + 24 = 72 total"
        assert check_calculation_in_text(text, "48+24=72")


# ---------------------------------------------------------------------------
# Mock tokenizer for testing
# ---------------------------------------------------------------------------

class MockTokenizer:
    """
    Simple tokenizer mock that treats each word/symbol as one token.
    """
    def __init__(self, token_map: list[str]):
        """token_map: list of strings, one per token ID."""
        self.token_map = token_map

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, list) and len(ids) == 1:
            idx = ids[0]
            if 0 <= idx < len(self.token_map):
                return self.token_map[idx]
            return ""
        return ""


def make_mock_sequence(tokens: list[str]):
    """
    Create a mock tokenizer and token ID tensor from a list of token strings.
    Returns (token_ids_tensor, tokenizer).
    """
    tokenizer = MockTokenizer(tokens)
    ids = torch.arange(len(tokens))
    return ids, tokenizer


# ---------------------------------------------------------------------------
# Test _compute_token_rewards
# ---------------------------------------------------------------------------

class TestComputeTokenRewards:
    def test_correct_answer_rewarded(self):
        tokens = ["<think>", "\n", "Simple", ".", "\n", "</think>", "\n", "<answer>", "72", "</answer>"]
        ids, tokenizer = make_mock_sequence(tokens)
        gt = "Some reasoning <<48+24=72>>72.\n#### 72"

        rewards = _compute_token_rewards(ids, tokenizer, gt)

        # </answer> token (index 9) should get correct_answer_reward=1.0
        assert rewards[9].item() == pytest.approx(1.0)

    def test_wrong_answer_no_reward(self):
        tokens = ["<think>", "\n", "Guess", ".", "\n", "</think>", "\n", "<answer>", "99", "</answer>"]
        ids, tokenizer = make_mock_sequence(tokens)
        gt = "48+24 = <<48+24=72>>72.\n#### 72"

        rewards = _compute_token_rewards(ids, tokenizer, gt)

        # </answer> should get wrong_answer_reward=0.0
        assert rewards[9].item() == pytest.approx(0.0)

    def test_intermediate_calc_rewarded(self):
        tokens = ["First", ",", " 48", "/", "2", " = ", "24", ".", " Then", " 48", "+", "24", "=", "72", "."]
        ids, tokenizer = make_mock_sequence(tokens)
        gt = "48/2 = <<48/2=24>>24.\n48+24 = <<48+24=72>>72.\n#### 72"

        rewards = _compute_token_rewards(ids, tokenizer, gt, calc_reward=0.5)

        # There should be exactly 2 calc rewards totalling 1.0
        total_calc_reward = rewards.sum().item()
        # Could also include answer reward if </answer> is present, but here it isn't
        assert total_calc_reward == pytest.approx(1.0)

    def test_calc_rewarded_only_once(self):
        # After "24" appears, subsequent tokens shouldn't re-trigger the same calc
        tokens = ["48", "/", "2", "=", "24", ".", " So", " 24", " is", " correct"]
        ids, tokenizer = make_mock_sequence(tokens)
        gt = "<<48/2=24>>24.\n#### 24"

        rewards = _compute_token_rewards(ids, tokenizer, gt, calc_reward=0.5)

        # Only one 0.5 reward for the first time "24" appears with operands
        calc_rewards = [r.item() for r in rewards if r.item() > 0]
        assert len(calc_rewards) == 1
        assert calc_rewards[0] == pytest.approx(0.5)

    def test_no_calculations_in_gt(self):
        tokens = ["<answer>", "42", "</answer>"]
        ids, tokenizer = make_mock_sequence(tokens)
        gt = "The answer is 42.\n#### 42"

        rewards = _compute_token_rewards(ids, tokenizer, gt)

        # Only answer reward, no calc rewards
        assert rewards[2].item() == pytest.approx(1.0)  # correct answer
        assert rewards[0].item() == pytest.approx(0.0)
        assert rewards[1].item() == pytest.approx(0.0)

    def test_format_reward_on_tags(self):
        tokens = ["<think>", "\n", "Work", ".", "\n", "</think>", "\n", "<answer>", "5", "</answer>"]
        ids, tokenizer = make_mock_sequence(tokens)
        gt = "#### 5"

        rewards = _compute_token_rewards(ids, tokenizer, gt, format_reward=1.0)

        # format_reward=1.0 split across 4 tags, 0.25 each
        assert rewards[0].item() == pytest.approx(0.25)   # <think>
        assert rewards[5].item() == pytest.approx(0.25)   # </think>
        assert rewards[7].item() == pytest.approx(0.25)   # <answer>
        # </answer> gets 0.25 (format) + 1.0 (correct answer)
        assert rewards[9].item() == pytest.approx(1.25)

    def test_no_format_reward_by_default(self):
        tokens = ["<think>", "work", "</think>", "<answer>", "5", "</answer>"]
        ids, tokenizer = make_mock_sequence(tokens)
        gt = "#### 5"

        rewards = _compute_token_rewards(ids, tokenizer, gt, format_reward=None)

        # <think> token should have 0 reward (no format shaping)
        assert rewards[0].item() == pytest.approx(0.0)

    def test_all_zeros_for_garbage(self):
        tokens = ["Hello", " world", " this", " is", " nonsense"]
        ids, tokenizer = make_mock_sequence(tokens)
        gt = "48/2 = <<48/2=24>>24.\n#### 24"

        rewards = _compute_token_rewards(ids, tokenizer, gt)

        assert rewards.sum().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test batch-level API
# ---------------------------------------------------------------------------

class TestBatchAPI:
    def _make_example(self, tokens, gt):
        ids, tok = make_mock_sequence(tokens)
        return ids, tok, gt

    def test_token_level_reward_batch(self):
        tokens1 = ["<answer>", "72", "</answer>"]
        tokens2 = ["<answer>", "99", "</answer>"]

        ids1, tok1 = make_mock_sequence(tokens1)
        ids2, tok2 = make_mock_sequence(tokens2)

        gt1 = "#### 72"
        gt2 = "#### 72"

        # Both use same tokenizer mock, so we need separate ones
        rewards = token_level_reward([ids1, ids2], [tok1, tok2], [gt1, gt2])

        assert len(rewards) == 2
        assert rewards[0][2].item() == pytest.approx(1.0)  # correct
        assert rewards[1][2].item() == pytest.approx(0.0)  # wrong

    def test_token_level_reward_with_format_batch(self):
        tokens = ["<think>", "work", "</think>", "<answer>", "5", "</answer>"]
        ids, tok = make_mock_sequence(tokens)
        gt = "#### 5"

        rewards = token_level_reward_with_format([ids], tok, [gt], format_reward=0.4)

        assert len(rewards) == 1
        r = rewards[0]
        # <think> gets 0.1, </think> gets 0.1, <answer> gets 0.1
        assert r[0].item() == pytest.approx(0.1)
        assert r[2].item() == pytest.approx(0.1)
        assert r[3].item() == pytest.approx(0.1)
        # </answer> gets 0.1 (format) + 1.0 (correct answer)
        assert r[5].item() == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# Test edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_sequence(self):
        ids = torch.tensor([], dtype=torch.long)
        tok = MockTokenizer([])
        rewards = _compute_token_rewards(ids, tok, "#### 42")
        assert len(rewards) == 0

    def test_single_token(self):
        ids, tok = make_mock_sequence(["42"])
        rewards = _compute_token_rewards(ids, tok, "#### 42")
        # No <answer> tags, so no answer reward even though 42 is correct
        assert rewards[0].item() == pytest.approx(0.0)

    def test_repeated_numbers_in_different_calcs(self):
        # Both calcs produce "12" but from different expressions
        tokens = ["3", "*", "4", "=", "12", ".", " ", "6", "+", "6", "=", "12", "."]
        ids, tok = make_mock_sequence(tokens)
        gt = "3*4=<<3*4=12>>12. 6+6=<<6+6=12>>12.\n#### 12"

        rewards = _compute_token_rewards(ids, tok, gt, calc_reward=0.5)

        # Both calculations should be found and rewarded
        nonzero = [i for i in range(len(rewards)) if rewards[i].item() > 0]
        assert len(nonzero) == 2
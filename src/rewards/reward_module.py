"""
TRL-compatible wrapper for rule-based reward functions.

TRL's get_reward() expects a neural reward model interface:
  - model.base_model_prefix  → name of backbone sub-module
  - model.<backbone>(input_ids, ...) → obj with .hidden_states
  - model.score(hidden_states) → (B, T, 1) scalar tensor

This module bridges that interface to arbitrary (response, ground_truth) -> float functions.
Ground truths are injected per-batch by GroundTruthPPOTrainer (see custom_ppo_trainer.py).
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import Callable

import src.rewards.sparse as r
from src.rewards.token_level import token_level_reward_with_format
from src.utils.math_verify import extract_answer_from_response
from src.utils.data_utils import extract_gsm8k_answer


class _TokenPassthroughBackbone(nn.Module):
    """
    Fake LM backbone that stores input_ids and echoes them as 'hidden states'.
    RuleBasedRewardWrapper.score() decodes them for rule-based reward computation.
    """

    def __init__(self):
        super().__init__()
        self._last_input_ids: torch.Tensor | None = None
        # Dummy param so the module registers properly with PyTorch
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self._last_input_ids = input_ids
        # Return input_ids as float 'hidden states' — shape (B, T, 1)
        fake_hidden = input_ids.unsqueeze(-1).float()
        return SimpleNamespace(hidden_states=(fake_hidden,))


class RuleBasedRewardWrapper(nn.Module):
    """
    Wraps a (response: str, ground_truth: str) -> float reward function
    in the interface expected by TRL's experimental PPOTrainer.

    Usage:
        wrapper = RuleBasedRewardWrapper(reward_fn=composite_reward, tokenizer=tokenizer)
        # Pass as reward_model to GroundTruthPPOTrainer — ground truths are injected
        # automatically per batch; you never call set_ground_truths() manually.

    To add new reward functions, just pass a different callable to reward_fn.
    The callable must have signature: (response: str, ground_truth: str) -> float.
    """

    # Required by TRL's get_reward(): getattr(model, model.base_model_prefix)
    base_model_prefix = "backbone"

    def __init__(
        self,
        reward_fn: str,
        reward_fn_kwargs: dict,
        tokenizer,
        gt_lookup,
    ):
        super().__init__()

        funcs = {
            "sparse": r.sparse_reward,
            "format_shaped": r.format_shaped_reward,
            "composite": r.composite_reward,
            "intermediate_calc": token_level_reward_with_format
        }

        self.reward_fn_str = reward_fn
        self.reward_fn = funcs[reward_fn]
        self.reward_fn_kwargs = reward_fn_kwargs
        self.tokenizer = tokenizer
        self.backbone = _TokenPassthroughBackbone()
        self._ground_truths: list[str] | None = None
        self.gt_lookup = gt_lookup

    # ------------------------------------------------------------------
    # TRL interface
    # ------------------------------------------------------------------
    def calculate_reward(
        self, 
        response_encoded=None, 
        response_decoded=None,
        gt=None,
    ):
        if self.reward_fn_str == "intermediate_calc":
            return self.reward_fn([response_encoded], self.tokenizer, gt_texts=[gt], **self.reward_fn_kwargs)
        else:
            return self.reward_fn(response_decoded, extract_gsm8k_answer(gt), **self.reward_fn_kwargs)


    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sequence scalar rewards.

        Args:
            hidden_states: (B, T, 1) — ignored; we decode from stored input_ids.

        Returns:
            (B, T, 1) reward tensor. TRL picks the value at the last non-pad position,
            so we broadcast the same scalar across all T positions.
        """
        input_ids = hidden_states.int().squeeze(-1)
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        prompts = [re.search(r"system.*?assistant\n", t, re.DOTALL) for t in texts]
        responses = [texts[i][prompts[i].end():] for i in range(len(texts))]
        prompts = [p.group(0) for p in prompts]
        
        gts_pred = [self.gt_lookup[p] for p in prompts]
        rewards = [
            self.calculate_reward(input_ids[j], responses[j], gts_pred[j])
            for j in range(len(responses))
        ]

        import random
        rand_idx = random.randint(0, len(texts)-1)

        print("  Prompt  ".center(30, "#"))
        print(prompts[rand_idx])
        print("  Response  ".center(30, "#"))
        print(responses[rand_idx])
        print("  Predicted  ".center(30, "#"))
        print(extract_answer_from_response(responses[rand_idx]))
        print("  GT  ".center(30, "#"))
        print(extract_gsm8k_answer(gts_pred[rand_idx]))
        print("  Reward  ".center(30, "#"))
        print(rewards[rand_idx])
        print("#"*60)

        reward_tensor = torch.tensor(
            rewards, dtype=torch.float32, device=hidden_states.device
        )  # (B,)

        B, T, _ = hidden_states.shape
        return reward_tensor.view(B, 1, 1).expand(B, T, 1).contiguous()

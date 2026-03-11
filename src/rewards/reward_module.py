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

import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import Callable


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
        reward_fn: Callable[[str, str], float],
        tokenizer,
    ):
        super().__init__()
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.backbone = _TokenPassthroughBackbone()
        self._ground_truths: list[str] | None = None

    # ------------------------------------------------------------------
    # Called by GroundTruthPPOTrainer before each get_reward() invocation
    # ------------------------------------------------------------------

    def set_ground_truths(self, ground_truths: list[str]) -> None:
        self._ground_truths = ground_truths

    # ------------------------------------------------------------------
    # TRL interface
    # ------------------------------------------------------------------

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sequence scalar rewards.

        Args:
            hidden_states: (B, T, 1) — ignored; we decode from stored input_ids.

        Returns:
            (B, T, 1) reward tensor. TRL picks the value at the last non-pad position,
            so we broadcast the same scalar across all T positions.
        """
        if self._ground_truths is None:
            raise RuntimeError(
                "Ground truths not set. Make sure you are using GroundTruthPPOTrainer, "
                "which automatically calls set_ground_truths() before each reward step."
            )

        input_ids = self.backbone._last_input_ids  # (B, T)
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        rewards = [
            self.reward_fn(text, gt)
            for text, gt in zip(texts, self._ground_truths)
        ]

        reward_tensor = torch.tensor(
            rewards, dtype=torch.float32, device=hidden_states.device
        )  # (B,)

        B, T, _ = hidden_states.shape
        return reward_tensor.view(B, 1, 1).expand(B, T, 1).contiguous()

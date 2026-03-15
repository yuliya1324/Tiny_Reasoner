"""
Custom PPOTrainer with ground-truth injection for rule-based reward functions.

TRL's PPOTrainer calls get_reward(reward_model, query_responses, pad_token_id, context_length).
We patch that function at the module level to intercept calls to our RuleBasedRewardWrapper,
decode the prompt portion of query_responses, look up ground truths, and inject them
before the actual reward computation runs.

This means:
  - No subclassing of TRL internals beyond PPOTrainer itself
  - No changes to your reward functions
  - Clean separation: gt injection lives here, reward logic stays in reward modules
"""

from __future__ import annotations

import trl.experimental.ppo.ppo_trainer as _trl_ppo_module
from trl.experimental.ppo import PPOTrainer

from src.rewards.reward_module import RuleBasedRewardWrapper


class GroundTruthPPOTrainer(PPOTrainer):
    """
    PPOTrainer that supports rule-based reward functions requiring ground truths.

    At each reward computation step, it:
      1. Extracts the prompt tokens (first context_length positions) from query_responses
      2. Decodes them to text
      3. Looks up the corresponding ground truth in ground_truth_lookup
      4. Injects ground truths into the RuleBasedRewardWrapper before TRL calls score()

    Args:
        ground_truth_lookup: dict mapping question_text (str) -> ground_truth_answer (str).
                             Built by prepare_ppo_dataset_with_gt().
        reward_wrapper:      RuleBasedRewardWrapper instance. Do NOT pass reward_model
                             separately — this is forwarded automatically.
        All other args/kwargs are passed through to PPOTrainer.
    """

    def __init__(
        self,
        *args,
        ground_truth_lookup: dict[str, str],
        reward_wrapper: RuleBasedRewardWrapper,
        **kwargs,
    ):
        # Inject reward_wrapper as reward_model for TRL
        super().__init__(*args, reward_model=reward_wrapper, **kwargs)
        self.gt_lookup = ground_truth_lookup
        self.reward_wrapper = reward_wrapper
        self._original_get_reward = _trl_ppo_module.get_reward
        self._install_patch()

    def _install_patch(self) -> None:
        """Patch trl's get_reward to inject ground truths when called for our wrapper."""
        reward_wrapper = self.reward_wrapper
        gt_lookup = self.gt_lookup
        tokenizer = self.processing_class
        original = self._original_get_reward

        def patched_get_reward(model, query_responses, pad_token_id, context_length):
            if model is reward_wrapper:
                # Prompt tokens are the first context_length positions per sample
                prompt_ids = query_responses[:, :context_length]
                prompts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                ground_truths = [gt_lookup.get(p, "") for p in prompts]
                reward_wrapper.set_ground_truths(ground_truths)

            return original(model, query_responses, pad_token_id, context_length)

        _trl_ppo_module.get_reward = patched_get_reward

    def _restore_patch(self) -> None:
        _trl_ppo_module.get_reward = self._original_get_reward

    def __del__(self):
        try:
            self._restore_patch()
        except Exception:
            pass
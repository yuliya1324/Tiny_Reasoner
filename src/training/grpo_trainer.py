"""
GRPO Trainer — Group Relative Policy Optimization on GSM8K.

Uses TRL's GRPOTrainer with configurable reward functions.
"""

from transformers import TrainerCallback
from trl import GRPOTrainer, GRPOConfig

from src.models.loader import (
    load_model_and_tokenizer,
    load_model_from_checkpoint,
    get_lora_config,
)
from src.utils.data_utils import prepare_grpo_dataset
from src.rewards.sparse import sparse_reward
from src.utils.math_verify import (
    has_correct_format,
    extract_thinking_from_response,
)


# ---------------------------------------------------------------------------
# Reward function wrappers
#
# The raw reward functions in src/rewards/sparse.py have signature
#   (response: str, ground_truth: str) -> float
#
# GRPOTrainer calls reward functions with keyword arguments:
#   (completions, answer, **kwargs) -> list[float]
#
# With conversational format, each element of `completions` is a list of
# message dicts, e.g. [{"role": "assistant", "content": "..."}].
# We extract the text content before passing to the underlying reward.
#
# Each wrapper must have a unique __name__ for GRPOTrainer logging.
# ---------------------------------------------------------------------------

def _get_completion_text(completion) -> str:
    """Extract plain text from a completion (handles both str and message-dict format)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "")
    return str(completion)


def _accuracy_reward(completions, answer, **kwargs):
    """Binary correctness reward."""
    return [sparse_reward(_get_completion_text(c), gt) for c, gt in zip(completions, answer)]


def _format_reward(completions, **kwargs):
    """Format adherence reward: 1.0 if <think>/<answer> tags present."""
    return [1.0 if has_correct_format(_get_completion_text(c)) else 0.0 for c in completions]


def _length_penalty_reward(completions, **kwargs):
    """Penalises excessively long reasoning to prevent reward hacking."""
    rewards = []
    for c in completions:
        text = _get_completion_text(c)
        thinking = extract_thinking_from_response(text)
        if thinking:
            excess = max(0, len(thinking.split()) - 300)
            rewards.append(max(0.0, -(excess * 0.001)))
        else:
            rewards.append(0.0)
    return rewards


def _build_reward_funcs(reward_type: str):
    """Return (reward_funcs, reward_weights) for the given reward type."""
    if reward_type == "sparse":
        return [_accuracy_reward], None
    elif reward_type == "format":
        return [_accuracy_reward, _format_reward], [1.0, 0.5]
    elif reward_type == "composite":
        return (
            [_accuracy_reward, _format_reward, _length_penalty_reward],
            [1.0, 0.5, 0.3],
        )
    else:
        raise ValueError(f"Unknown reward_type: {reward_type!r}")


# ---------------------------------------------------------------------------
# Metrics logging callback — prints to stdout so nohup log tracks progress
# ---------------------------------------------------------------------------

class _MetricsLoggingCallback(TrainerCallback):
    """Print a compact metrics line at each logging step for log-file tracking."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        step = state.global_step
        parts = [f"Step {step}"]
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        for k, v in sorted(logs.items()):
            if k == "loss" or not isinstance(v, (int, float)):
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(" | ".join(parts), flush=True)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_grpo(cfg: dict):
    """
    Run GRPO training.

    Args:
        cfg: Merged config dict (base.yaml + grpo.yaml).
    """
    grpo_cfg = cfg["grpo"]
    output_dir = cfg.get("output_dir")
    sft_checkpoint = cfg.get("sft_checkpoint")

    # ------------------------------------------------------------------
    # Load model + tokenizer
    # ------------------------------------------------------------------
    if sft_checkpoint:
        model, tokenizer = load_model_from_checkpoint(sft_checkpoint, cfg)
        model.train()
        peft_config = None  # already a PeftModel
    else:
        model, tokenizer = load_model_and_tokenizer(cfg, for_training=False)
        peft_config = get_lora_config(cfg)

    # ------------------------------------------------------------------
    # Prepare dataset
    # ------------------------------------------------------------------
    train_dataset = prepare_grpo_dataset(
        split="train",
        max_samples=cfg["data"].get("max_train_samples"),
    )

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------
    reward_funcs, reward_weights = _build_reward_funcs(cfg.get("reward_type", "sparse"))

    # ------------------------------------------------------------------
    # GRPOConfig
    # ------------------------------------------------------------------
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_generations=grpo_cfg.get("num_generations", 8),
        per_device_train_batch_size=grpo_cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=grpo_cfg.get("gradient_accumulation_steps", 2),
        learning_rate=grpo_cfg.get("learning_rate", 5e-6),
        num_train_epochs=grpo_cfg.get("num_train_epochs", 1),
        max_completion_length=grpo_cfg.get("max_completion_length", 1024),
        max_prompt_length=grpo_cfg.get("max_prompt_length", 256),
        temperature=grpo_cfg.get("temperature", 1.0),
        beta=grpo_cfg.get("beta", 0.0),
        loss_type=grpo_cfg.get("loss_type", "dr_grpo"),
        bf16=grpo_cfg.get("bf16", True),
        fp16=grpo_cfg.get("fp16", False),
        use_liger_loss=grpo_cfg.get("use_liger_loss", True),
        gradient_checkpointing=grpo_cfg.get("gradient_checkpointing", True),
        logging_steps=grpo_cfg.get("logging_steps", 10),
        save_steps=grpo_cfg.get("save_steps", 200),
        reward_weights=reward_weights,
        report_to="none" if __import__("os").environ.get("WANDB_DISABLED") else "wandb",
        run_name=cfg["experiment_name"],
        seed=cfg.get("seed", 42),
    )

    # ------------------------------------------------------------------
    # Create trainer
    # ------------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[_MetricsLoggingCallback()],
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    resume = cfg.get("resume_from_checkpoint")
    if resume is True:
        # Find latest checkpoint with valid trainer_state.json (skip empty/corrupt from failed saves)
        import os
        TRAINER_STATE_NAME = "trainer_state.json"
        checkpoints = []
        for name in os.listdir(output_dir):
            if name.startswith("checkpoint-"):
                path = os.path.join(output_dir, name)
                state_file = os.path.join(path, TRAINER_STATE_NAME)
                if os.path.isfile(state_file) and os.path.getsize(state_file) > 0:
                    checkpoints.append(path)
        if checkpoints:
            checkpoints.sort(key=lambda p: int(p.split("checkpoint-")[-1]))
            resume = checkpoints[-1]
            print(f"Resuming from {resume} (latest valid checkpoint)")
        else:
            resume = None
            print(f"No valid checkpoint in {output_dir}, starting from scratch")
    if resume:
        print(f"Resuming GRPO training — output: {output_dir}")
    else:
        print(f"Starting GRPO training — output: {output_dir}")

    # Resume: if optimizer state doesn't match (e.g. param group size), skip loading it and retry
    import os
    import shutil
    OPTIMIZER_NAME = "optimizer.pt"
    SCHEDULER_NAME = "scheduler.pt"
    try:
        trainer.train(resume_from_checkpoint=resume)
    except ValueError as e:
        if resume and ("parameter group" in str(e).lower() or "optimizer" in str(e).lower()):
            opt_path = os.path.join(resume, OPTIMIZER_NAME)
            sched_path = os.path.join(resume, SCHEDULER_NAME)
            if os.path.isfile(opt_path) or os.path.isfile(sched_path):
                for p, name in [(opt_path, OPTIMIZER_NAME), (sched_path, SCHEDULER_NAME)]:
                    if os.path.isfile(p):
                        bak = p + ".skip_bak"
                        shutil.move(p, bak)
                        print(f"Skipping incompatible optimizer/scheduler: moved {name} -> {name}.skip_bak")
                trainer.train(resume_from_checkpoint=resume)
            else:
                raise
        else:
            raise

    # Save final model
    final_path = f"{output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")

    return trainer

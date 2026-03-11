"""
PPO Trainer — reinforcement learning from reward signal on GSM8K.

Uses TRL's PPOTrainer with a reference model and reward model.
"""
import torch

from trl.experimental.ppo import PPOTrainer, PPOConfig

from src.models.loader import load_model_and_tokenizer, load_value_model, get_lora_config
from src.utils.data_utils import prepare_ppo_dataset
from src.rewards.reward_module import RewardModule


def train_ppo(cfg: dict):
    """
    Run PPO fine-tuning.
    Args:
        cfg: Merged config dict (base.yaml + ppo.yaml).
    """
    train_cfg = cfg["training"]
    output_dir = cfg.get("output_dir")

    # Load policy model + tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg, for_training=False)

    # Load value model
    value_model = load_value_model(cfg, for_training=False)

    # Load frozen reference model
    ref_model, _ = load_model_and_tokenizer(cfg, for_training=False)

    # Load reward model
    reward_model = RewardModule("sparse")

    # Prepare dataset
    train_dataset = prepare_ppo_dataset(
        tokenizer,
        split="train",
        max_samples=cfg["data"].get("max_train_samples"),
        max_length=cfg["data"].get("max_length", 512),
    )

    # PPO config
    ppo_config = PPOConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_epochs", 1),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 1e-5),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        fp16=train_cfg.get("fp16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 200),
        report_to="wandb",
        run_name=cfg["ppo"],
        seed=cfg.get("seed", 42),
        # PPO-specific
        num_ppo_epochs=train_cfg.get("num_ppo_epochs", 4),
        num_mini_batches=train_cfg.get("num_mini_batches", 1),
        kl_coef=train_cfg.get("kl_coef", 0.05),
        cliprange=train_cfg.get("cliprange", 0.2),
        vf_coef=train_cfg.get("vf_coef", 0.1),
        gamma=train_cfg.get("gamma", 1.0),
        lam=train_cfg.get("lam", 0.95),
        # Generation
        response_length=train_cfg.get("response_length", 256),
        missing_eos_penalty=train_cfg.get("missing_eos_penalty", 1.0),
    )

    # Create trainer
    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=train_dataset,
        value_model=value_model,
        data_collator=None,
        eval_dataset=None,
        # optimizers=None,
        # peft_config=get_lora_config(cfg)
    )

    # Train
    print(f"Starting PPO training — output: {output_dir}")
    trainer.train()

    # Save final model
    final_path = f"{output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")

    return trainer
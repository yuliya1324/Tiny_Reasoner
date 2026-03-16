"""
PPO Trainer — reinforcement learning from reward signal on GSM8K.
Uses TRL's PPOTrainer with a reference model and rule-based reward function.
"""
import torch
from trl.experimental.ppo import PPOConfig, PPOTrainer

from src.models.loader import load_model_from_checkpoint, load_value_model
from src.utils.data_utils import prepare_ppo_dataset_with_gt
from src.rewards.reward_module import RuleBasedRewardWrapper


def train_ppo(cfg: dict):
    """
    Run PPO fine-tuning.

    Args:
        cfg: Merged config dict (base.yaml + ppo.yaml).
    """
    train_cfg = cfg["training"]
    output_dir = cfg.get("output_dir")

    # Load policy model + tokenizer
    model, tokenizer = load_model_from_checkpoint(cfg["model"]["checkpoint"], cfg)
    tokenizer.padding_side = "left"
    model.train()

    # Load value and ref models
    value_model = load_value_model(cfg)
    ref_model, _ = load_model_from_checkpoint(cfg["model"]["checkpoint"], cfg)

    # Prepare dataset with ground truth
    train_dataset, gt_lookup = prepare_ppo_dataset_with_gt(
        tokenizer,
        split="train",
        max_samples=cfg["data"].get("max_train_samples"),
        max_length=cfg["data"].get("max_length", 512),
    )

    # Load reward model
    reward_wrapper = RuleBasedRewardWrapper(
        reward_fn=train_cfg.get("reward"),
        reward_fn_kwargs=train_cfg.get("reward_kwargs", {}),
        tokenizer=tokenizer,
        gt_lookup=gt_lookup
    )

    # PPO config
    ppo_config = PPOConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_epochs", 1),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        
        # Optmizer
        learning_rate=train_cfg.get("learning_rate", 1e-5),
        adam_epsilon=train_cfg.get("adam_epsilon", 1e-8),
        weight_decay=train_cfg.get("weight_decay", 0.01),

        # scheduler
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        lr_scheduler_kwargs=train_cfg.get("lr_scheduler_kwargs", {}),
        warmup_steps=train_cfg.get("warmup_steps", 100),

        # Precision
        fp16=train_cfg.get("fp16", True),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=False,
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        
        # Logging
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 200),
        report_to="wandb",
        run_name=cfg["experiment_name"],
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
        response_length=train_cfg.get("response_length", 512),
        missing_eos_penalty=train_cfg.get("missing_eos_penalty", 1.0),
    )

    # Create trainer
    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_model=reward_wrapper,
        train_dataset=train_dataset,
        value_model=value_model,
        data_collator=None,
        eval_dataset=None,
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

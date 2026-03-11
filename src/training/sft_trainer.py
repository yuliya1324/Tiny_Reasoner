"""
SFT Trainer — supervised fine-tuning on GSM8K correct solutions.

Uses TRL's SFTTrainer which handles packing, LoRA, and all the plumbing.
"""

from trl import SFTTrainer, SFTConfig

from src.models.loader import load_model_and_tokenizer
from src.utils.data_utils import prepare_sft_dataset


def train_sft(cfg: dict):
    """
    Run supervised fine-tuning.

    Args:
        cfg: Merged config dict (base.yaml + sft.yaml).
    """
    train_cfg = cfg["training"]
    output_dir = cfg.get("output_dir", "results/sft_baseline")

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg, for_training=True)

    # Prepare dataset
    train_dataset = prepare_sft_dataset(
        tokenizer,
        split="train",
        max_samples=cfg["data"].get("max_train_samples"),
        max_length=cfg["data"].get("max_length", 512),
    )
    eval_dataset = prepare_sft_dataset(
        tokenizer,
        split="test",
        max_samples=200,  # small eval set for speed
        max_length=cfg["data"].get("max_length", 512),
    )

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 200),
        eval_steps=train_cfg.get("eval_steps", 200),
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        run_name=cfg["experiment_name"],
        max_length=cfg["data"].get("max_length", 512),
        dataset_text_field="text",
        seed=cfg.get("seed", 42),
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    print(f"Starting SFT training — output: {output_dir}")
    trainer.train()

    # Save final model
    final_path = f"{output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")

    return trainer
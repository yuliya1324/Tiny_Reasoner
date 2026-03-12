"""
DPO Trainer — preference optimization on GSM8K chosen/rejected pairs.
"""

import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

from src.utils.data_utils import (
    prepare_dpo_dataset,
    prepare_dpo_dataset_from_sft_outputs,
)


def get_bnb_config():
    """4-bit NF4 quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer_for_dpo(cfg: dict):
    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def load_policy_model_from_sft_checkpoint(cfg: dict):
    checkpoint_path = cfg["model_init"]["checkpoint_path"]
    model_name = cfg["model"]["name"]
    use_quant = cfg["model"].get("quantization") == "4bit"

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }
    if use_quant:
        model_kwargs["quantization_config"] = get_bnb_config()

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=True)
    model.train()
    model.print_trainable_parameters()
    return model


def load_ref_model_from_sft_checkpoint(cfg: dict):
    checkpoint_path = cfg["model_init"]["checkpoint_path"]
    model_name = cfg["model"]["name"]
    use_quant = cfg["model"].get("quantization") == "4bit"

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }
    if use_quant:
        model_kwargs["quantization_config"] = get_bnb_config()

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    ref_model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=False)
    ref_model.eval()

    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model


def load_dpo_train_dataset(cfg: dict, tokenizer):
    """
    Load training dataset depending on config:
      - synthetic
      - sft_outputs
    """
    dpo_source = cfg["data"].get("dpo_source", "synthetic")

    if dpo_source == "synthetic":
        return prepare_dpo_dataset(
            tokenizer,
            split="train",
            max_samples=cfg["data"].get("max_train_samples"),
        )

    if dpo_source == "sft_outputs":
        ds = prepare_dpo_dataset_from_sft_outputs(
            cfg["data"]["train_pair_path"]
        )
        max_samples = cfg["data"].get("max_train_samples")
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        return ds

    raise ValueError(f"Unknown dpo_source: {dpo_source}")


def load_dpo_eval_dataset(cfg: dict, tokenizer):
    """
    Keep eval simple and consistent:
    always use the synthetic test set for now.
    """
    return prepare_dpo_dataset(
        tokenizer,
        split="test",
        max_samples=200,
    )


def train_dpo(cfg: dict):
    train_cfg = cfg["training"]
    dpo_cfg = cfg["dpo"]
    output_dir = cfg["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = load_tokenizer_for_dpo(cfg)

    train_dataset = load_dpo_train_dataset(cfg, tokenizer)
    eval_dataset = load_dpo_eval_dataset(cfg, tokenizer)

    policy_model = load_policy_model_from_sft_checkpoint(cfg)
    ref_model = load_ref_model_from_sft_checkpoint(cfg)

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_epochs", 1),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 1e-5),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 200),
        eval_steps=train_cfg.get("eval_steps", 200),
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none" if os.environ.get("WANDB_DISABLED") == "true" else "wandb",
        run_name=cfg["experiment_name"],
        fp16=train_cfg.get("fp16", True),
        beta=dpo_cfg.get("beta", 0.1),
        max_length=dpo_cfg.get("max_length", 512),
        max_prompt_length=dpo_cfg.get("max_prompt_length", 256),
        seed=cfg.get("seed", 42),
    )

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print(f"Starting DPO training — output: {output_dir}")
    print(f"DPO source: {cfg['data'].get('dpo_source', 'synthetic')}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    trainer.train()

    final_path = f"{output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")

    return trainer
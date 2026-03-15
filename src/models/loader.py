"""
Model loader — Qwen2.5-0.5B-Instruct with 4-bit quantization and LoRA.

Usage:
    from src.models.loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


def get_bnb_config():
    """4-bit NF4 quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(cfg: dict) -> LoraConfig:
    """Build LoRA config from the config dict."""
    lora_cfg = cfg["model"]["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def load_tokenizer(model_name: str):
    """Load tokenizer with proper padding setup."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_and_tokenizer(cfg: dict, for_training: bool = True):
    """
    Load the model and tokenizer.

    Args:
        cfg: Full config dict (parsed from YAML).
        for_training: If True, apply LoRA and prepare for training.
                      If False, load base model only (for inference/eval).

    Returns:
        (model, tokenizer) tuple.
    """
    model_name = cfg["model"]["name"]
    use_quant = cfg["model"].get("quantization") == "4bit"

    # Load tokenizer
    tokenizer = load_tokenizer(model_name)

    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    if use_quant:
        model_kwargs["quantization_config"] = get_bnb_config()
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if for_training:
        if use_quant:
            model = prepare_model_for_kbit_training(model)
        lora_config = get_lora_config(cfg)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def load_model_from_checkpoint(checkpoint_path: str, cfg: dict):
    """Load a fine-tuned LoRA model from a checkpoint (local path only)."""
    import os
    from peft import PeftModel

    # Resolve to absolute path so PEFT loads from disk, not Hub
    path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"SFT checkpoint not found at '{path}'. "
            "Copy the SFT checkpoint to this machine, e.g. "
            "scp -r other_machine:/path/to/results/sft_baseline/final /Data/yash.bhardwaj/Tiny_Reasoner/results/sft_baseline/"
        )
    adapter_config = os.path.join(path, "adapter_config.json")
    if not os.path.isfile(adapter_config):
        raise FileNotFoundError(
            f"Not a valid PEFT checkpoint: missing adapter_config.json in '{path}'."
        )

    model_name = cfg["model"]["name"]
    tokenizer = load_tokenizer(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    # Must be called before loading PEFT adapters so gradients flow through
    # the 4-bit base layers correctly during any subsequent training.
    model = prepare_model_for_kbit_training(model)
    # is_trainable=True keeps adapters in training mode and casts them to float16
    # so the GradScaler fp16 hooks fire correctly during GRPO training.
    model = PeftModel.from_pretrained(model, path, is_trainable=True)
    model.eval()

    return model, tokenizer

"""
Model loader — Qwen2.5-0.5B-Instruct with 4-bit quantization and LoRA.

Usage:
    from src.models.loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel


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
    base_model_name = cfg["model"].get("base_name", model_name)

    tokenizer = load_tokenizer(base_model_name)

    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if use_quant:
        model_kwargs["quantization_config"] = get_bnb_config()
    model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

    # If model_name != base_name, it's a LoRA checkpoint — load adapter
    if model_name != base_model_name:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_name)

    if for_training:
        if use_quant:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        lora_config = get_lora_config(cfg)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def load_model_from_checkpoint(checkpoint_path: str, cfg: dict, peft: bool = True):
    """Load a fine-tuned LoRA model from a checkpoint."""
    model_name = cfg["model"]["name"]
    base_model_name = cfg["model"].get("base_name", model_name)

    tokenizer = load_tokenizer(base_model_name)

    if peft:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # quantization_config=get_bnb_config(),
            device_map="auto",
            torch_dtype=torch.bfloat16,      
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    return model, tokenizer

def add_new_adapter(model, cfg, adapter_name):
    """ Add a new adapter on top of the model. Keep
    default adapter and freeze it.
    """
    lora_config = get_lora_config(cfg)
    model.add_adapter(adapter_name, lora_config)
    model.base_model.set_adapter(["default", adapter_name])
    for name, param in model.named_parameters():
        if "lora_" in name and ".default." in name:
            param.requires_grad = False    
    model.print_trainable_parameters()
    return model

def load_adapters(model, checkpoint, adapters):
    """ Load non-default adapters
    """
    for adapter in adapters:
        model.load_adapter(
            checkpoint,
            adapter_name=adapter
        )
    model.base_model.set_adapter(["default"] + adapters)
    return model

def load_value_model(cfg: dict) -> torch.nn.Module:
    model_name = cfg["model"]["name"]
    base_model_name = cfg["model"].get("base_name", model_name)  # use base
    use_quant = cfg["model"].get("quantization") == "4bit"

    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16, "num_labels": 1}
    if use_quant:
        model_kwargs["quantization_config"] = get_bnb_config()
    model_kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, **model_kwargs)
    model.config.pad_token_id = model.config.eos_token_id

    if use_quant:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["model"]["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.SEQ_CLS,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model

"""
GRPO experiment — entry point.

Usage:
    python -m experiments.grpo.train --config configs/grpo.yaml
    python -m experiments.grpo.train  # uses defaults
"""

import argparse
import yaml
from pathlib import Path
from src.training.grpo_trainer import train_grpo


def load_config(config_path: str = None) -> dict:
    """Load and merge base + experiment configs."""
    base_path = Path("configs/base.yaml")
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    if config_path:
        with open(config_path) as f:
            exp_cfg = yaml.safe_load(f)
        # Shallow merge: experiment config overrides base
        for key, value in exp_cfg.items():
            if isinstance(value, dict) and key in cfg and isinstance(cfg[key], dict):
                cfg[key].update(value)
            else:
                cfg[key] = value

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train GRPO")
    parser.add_argument("--config", type=str, default="configs/grpo.yaml",
                        help="Path to experiment config YAML")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from this checkpoint path, or 'true' to use latest in output_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resume_from_checkpoint is not None:
        cfg["resume_from_checkpoint"] = args.resume_from_checkpoint if args.resume_from_checkpoint.lower() != "true" else True

    if args.no_wandb:
        import os
        os.environ["WANDB_DISABLED"] = "true"

    trainer = train_grpo(cfg)
    print("GRPO training complete!")


if __name__ == "__main__":
    main()

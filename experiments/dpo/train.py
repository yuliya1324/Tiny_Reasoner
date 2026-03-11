"""
DPO Baseline experiment — entry point.

Usage:
    python -m experiments.dpo.train --config configs/dpo.yaml
    python -m experiments.dpo.train  # uses defaults
"""

import argparse
import os
import yaml
from pathlib import Path
from src.training.dpo_trainer import train_dpo


def load_config(config_path: str = None) -> dict:
    """Load and merge base + experiment configs."""
    base_path = Path("configs/base.yaml")
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    if config_path:
        with open(config_path) as f:
            exp_cfg = yaml.safe_load(f)
        for key, value in exp_cfg.items():
            if isinstance(value, dict) and key in cfg and isinstance(cfg[key], dict):
                cfg[key].update(value)
            else:
                cfg[key] = value

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train DPO baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dpo.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    train_dpo(cfg)
    print("DPO training complete!")


if __name__ == "__main__":
    main()
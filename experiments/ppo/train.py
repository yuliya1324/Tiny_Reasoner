"""
PPO experiment — entry point.

Usage:
    python -m experiments.sft_baseline.train --config configs/sft.yaml
    python -m experiments.sft_baseline.train  # uses defaults
"""

import argparse
import yaml
from pathlib import Path
from src.training.ppo_trainer import train_ppo


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
    parser = argparse.ArgumentParser(description="Train PPO experiment")
    parser.add_argument("--config", type=str, default="configs/sft.yaml",
                        help="Path to experiment config YAML")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.no_wandb:
        import os
        os.environ["WANDB_DISABLED"] = "true"

    train_ppo(cfg)
    print("PPO training complete!")


if __name__ == "__main__":
    main()

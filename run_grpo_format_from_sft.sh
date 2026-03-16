#!/usr/bin/env bash
# Run GRPO format-reward from SFT checkpoint with nohup (survives SSH disconnect).
# Usage: nohup ./run_grpo_format_from_sft.sh &
#
# Track logs: tail -f grpo_format_from_sft_*.log

set -e
cd "$(dirname "$0")"

# Use home-dir cache to avoid /Data lock-file permission issues
export HF_HOME="/users/eleves-a/2025/yash.bhardwaj/.cache/huggingface"
export HF_DATASETS_CACHE="/users/eleves-a/2025/yash.bhardwaj/.cache/huggingface/datasets"

LOG="grpo_format_from_sft_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG" >&2
echo "Started at $(date)" >> "$LOG"
exec python -m experiments.grpo.train --config configs/grpo_format_from_sft.yaml >> "$LOG" 2>&1

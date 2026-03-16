#!/usr/bin/env bash
# Run GRPO from SFT with nohup (survives SSH disconnect).
# Usage: nohup ./run_grpo_from_sft.sh &

set -e
cd "$(dirname "$0")"
LOG="grpo_from_sft_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG" >&2
echo "Started at $(date)" > "$LOG"
exec python -m experiments.grpo.train --config configs/grpo_from_sft.yaml >> "$LOG" 2>&1

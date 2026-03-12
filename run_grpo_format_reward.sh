#!/usr/bin/env bash
# Run GRPO format reward experiment with nohup (survives SSH disconnect).
# Usage: nohup ./run_grpo_format_reward.sh &
#
# Track logs: tail -f grpo_format_reward_*.log

set -e
cd "$(dirname "$0")"
LOG="grpo_format_reward_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG" >&2
echo "Started at $(date)" >> "$LOG"
exec python -m experiments.grpo.train --config configs/grpo_format_reward.yaml >> "$LOG" 2>&1

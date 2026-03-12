#!/usr/bin/env bash
# Run GRPO composite-from-SFT with nohup (survives SSH disconnect).
# Usage: nohup ./run_grpo_composite_from_sft.sh &
# Resume: nohup ./run_grpo_composite_from_sft.sh resume &

set -e
cd "$(dirname "$0")"
RESUME=""
[[ "${1:-}" == "resume" ]] && RESUME="--resume_from_checkpoint true"
LOG="grpo_composite_from_sft_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG" >&2
echo "Started at $(date)" > "$LOG"
exec python -m experiments.grpo.train --config configs/grpo_composite_from_sft.yaml $RESUME >> "$LOG" 2>&1

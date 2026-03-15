#!/bin/bash
# Evaluate GRPO composite final checkpoint and print metrics
set -e
cd "$(dirname "$0")"
CHECKPOINT="/Data/yash.bhardwaj/Tiny_Reasoner/results/grpo_composite/final"
OUT="results/eval_grpo_composite_test.json"
echo "Evaluating $CHECKPOINT on GSM8K test..."
python -m src.evaluation.evaluate \
  --checkpoint "$CHECKPOINT" \
  --config configs/base.yaml \
  --split test \
  --output "$OUT"
echo ""
echo "=== GRPO Composite (test) ==="
python -c "
import json
with open('$OUT') as f:
    m = json.load(f)
for k, v in m.items():
    if isinstance(v, float):
        print(f'  {k}: {v:.4f}')
    else:
        print(f'  {k}: {v}')
"
echo "Results saved to $OUT"

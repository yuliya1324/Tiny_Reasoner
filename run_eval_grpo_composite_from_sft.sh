#!/usr/bin/env bash
# Evaluate GRPO composite-from-SFT final checkpoint on GSM8K test.
set -e
cd "$(dirname "$0")"
CHECKPOINT="/Data/yash.bhardwaj/Tiny_Reasoner/results/grpo_composite_from_sft/final"
OUT="results/eval_grpo_composite_from_sft_test.json"
echo "Evaluating $CHECKPOINT on GSM8K test..."
python -m src.evaluation.evaluate \
  --checkpoint "$CHECKPOINT" \
  --config configs/base.yaml \
  --split test \
  --output "$OUT"
echo ""
echo "=== GRPO Composite from SFT (test) ==="
python -c "import json; f=open('$OUT'); m=json.load(f); [print(f'  {k}: {v:.4f}' if isinstance(v,float) else f'  {k}: {v}') for k,v in m.items()]"
echo "Results saved to $OUT"

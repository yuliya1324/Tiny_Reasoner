# Tiny Reasoner — GRPO vs. PPO for Teaching a 0.5B Model to Think

> RL Course Project: Investigating whether GRPO can teach a small language model
> chain-of-thought reasoning on math problems, compared against PPO, DPO, and SFT.

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Run SFT baseline (change `output_dir` in `configs/sft.yaml`)
python -m experiments.sft_baseline.train --config configs/dpo.yaml

# 3. Evaluate
python -m src.evaluation.evaluate --checkpoint <path_to_results_dir>/results/dpo_baseline/final --split test --output <path_to_results_dir>/results/dpo_baseline/eval_test.json --batch_size 16
```

## DPO design
### DPO baseline
Make wrong answers automatically.

Rule: 
- Integer: ±max(round(10%), 1)
- Decimal: ±max(10%, 0.1)
- Randomly choose + or -
- Fallback: append '_wrong'


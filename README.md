# Tiny Reasoner — GRPO vs. PPO for Teaching a 0.5B Model to Think

> RL Course Project: Investigating whether GRPO can teach a small language model
> chain-of-thought reasoning on math problems, compared against PPO, DPO, and SFT.

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Run SFT baseline (change `output_dir` in `configs/sft.yaml`)
python -m experiments.dpo.train --config configs/dpo.yaml

# 3. Evaluate
python -m src.evaluation.evaluate --checkpoint <path_to_results_dir>/results/dpo_baseline/final --split test --output <path_to_results_dir>/results/dpo_baseline/eval_test.json --batch_size 16
```

## DPO design
### DPO baseline
Changed only the numerical final answers for DPO rejected samples, while keeping the reasoning (`<think>...</think>`) unchanged. This design was motivated by the SFT baseline results: reasoning quality was already high (`has_reasoning > 0.8`), while answer accuracy remained low (`accuracy < 0.2`).

Wrong Answer Rule: 
- Integer: `±max(round(10%), 1)`
- Decimal: `±max(10%, 0.1)`
- Randomly choose `+` or `-`
- Fallback: append '_wrong'

Example: \
CHOSEN:
 <think>
Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
</think>
<answer>72</answer>

REJECTED:
 <think>
Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
</think>
<answer>79</answer>

#### Results
  n_examples: 1319
  accuracy: 0.0015
  format_adherence: 0.2540
  has_reasoning: 0.6861
  answer_extraction_rate: 0.2790
  avg_reasoning_length: 35.7597


Output Example\
<think>
The increase in value was 150% so that's 150/100 * 80,000 = $120,000
He spent 80,000 + 50,000 = $130,000 on repairs
So his profit was 130,000 - 120,000 = $10,000
</think กรกฎาคม> honeymoon_trip = 10000
cost_of_house = 80000
repair_value_increase = 150/100 * 80000 = $120,000
Total cost of trip = 10000 + 80000 = $90,000
Profit = 90000 - 10000 = $80,000
</scratch>80000
80000
</answer>80000

-> answer correctness を学ばせるつもりが、人工的すぎる preference pair のせいで出力分布そのものを壊した

### SFT baseline based


CHOSEN:
 <think>
In the beginning, Betty has only 100 / 2 = $50.
Betty's grandparents gave her 15 * 2 = $30.
This means, Betty needs 100 - 50 - 30 - 15 = $5 more.
</think>
<answer>5</answer>

REJECTED:
 <think>
Betty's parents gave her 15 + 2 * 15 = $60.
So in total, Betty has 60 + 15 = $75.
Thus, she still needs 100 - 75 = $25.
</think>
<answer>25</answer>


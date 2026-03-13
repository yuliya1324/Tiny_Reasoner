# Tiny Reasoner — DPO vs. PPO vs. GRPO for Teaching a 0.5B Model to Think

> RL Course Project: Investigating whether GRPO can teach a small language model
> chain-of-thought reasoning on math problems, compared against PPO, DPO, and SFT.

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Run SFT baseline (change `output_dir` in `configs/sft.yaml`)
python -m experiments.sft_baseline.train --config configs/sft.yaml

# 3. Evaluate
python -m src.evaluation.evaluate --checkpoint <path_to_results_dir>/results/sft_baseline/final --split test --output <path_to_results_dir>/results/sft_baseline/eval_test.json --batch_size 16
```

## Project Structure

```
Tiny_Reasoner/
├── configs/                    # YAML configs for all experiments
│   ├── base.yaml               # Shared defaults (model, data, tokenizer)
│   └── sft.yaml                # SFT baseline config
│
├── src/                        # Shared library code
│   ├── models/loader.py        # Load Qwen2.5-0.5B with LoRA + 4-bit quant
│   ├── training/               # Training loops (sft, grpo, ppo, dpo)
│   ├── rewards/                # Reward functions (sparse, format, composite, pbrs)
│   ├── evaluation/evaluate.py  # Accuracy, format adherence, reasoning quality
│   └── utils/                  # Data loading, math verification
│
├── experiments/                # One folder per experiment
│   └── sft_baseline/train.py
│
├── notebooks/analysis.ipynb    # Analysis and figures
├── requirements.txt
└── .gitignore
```

## What we use

**Model**: `Qwen/Qwen2.5-0.5B-Instruct`

It already has decent instruction following and some math ability. It's on HuggingFace and works out of the box with TRL + LoRA + 4-bit quantization.

**Dataset**: **GSM8K** (`openai/gsm8k` on HuggingFace)

7.5k train / 1.3k test grade-school math problems with step-by-step solutions. It's the standard benchmark for math reasoning.

The format we want the model to use:

```text
<think>
Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether.
</think>
<answer>72</answer>
```

**Metrics**:

- *Accuracy*: `check_answer()` parses the content inside `<answer>` tags and compares numerically against the ground truth.
- *Format adherence*: Did it use the `<think>/<answer>` structure at all? (`has_correct_format()`).
- *Has reasoning*: verifies that the content between `<think>` tags is at least the length of 5.
- *Avg reasoning length*: How many words are inside the `<think>` tags? This tells you whether the model is actually "thinking" or just guessing (or the level of reward hacking with verbosity).

## Results

| Method | Reward | Accuracy | Format Adherence | Has Reasoning | Avg Reasoning Len |
|--------|--------|:--------:|:----------------:|:-------------:|:-----------------:|
| SFT Baseline | — | 0.2995 | 0.9977 | 0.9977 | 46.5830 |
| DPO (WIP) | — | 0.2540 | 0.9947 | 0.9879 | 38.9530 |

## Team Workflow

1. Everyone runs `sft_baseline` to verify setup works on their GPU (see Quick Start and change `output_dir` in `configs/sft.yaml`)
2. Each member develops their experiments on a feature branch
    - Use `configs` directory to store the configs for your experiments
    -  Use `experiments/your_experiment` for your experiment code
    - Add the function to prepare a dataset for your experiment `prepare_<your_exp>_dataset` to  `src/utils/data_utils.py` if needed 
    - Add your reward function to `src/rewards`
3. Merge into `main` once experiments complete
4. Run analysis notebook to generate figures for the presentation (change `RESULTS_DIR`)

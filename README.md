# Tiny Reasoner — DPO vs. PPO vs. GRPO for Teaching a 0.5B Model to Think

> Investigating whether RL methods can teach a *tiny* language model chain-of-thought reasoning on math problems.

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
├── configs/                        # YAML configs for all experiments
│   └── base.yaml                   # Shared defaults (model, data, tokenizer)
│
├── src/                            # Shared library code
│   ├── models/loader.py            # Load Qwen2.5-0.5B with LoRA + 4-bit quant
│   ├── training/                   # Training loops (sft, grpo, ppo, dpo)
│   ├── rewards/                    # Reward functions (sparse, format, composite, pbrs)
│   ├── evaluation/evaluate.py      # Accuracy, format adherence, reasoning quality
│   └── utils/                      # Data loading, math verification
│
├── experiments/                    # One folder per experiment
│   └── sft_baseline/train.py       # Example of an SFT experiment
│
├── notebooks/
│   └── Tiny_Resoner_notebook.ipynb # Analysis and figures
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
| DPO | — | 0.2987 | 0.9977 | 0.9947 | 40.9856 |
| PPO  | sparse | 0.1175 | 0.9992 | 0.9909 | 24.93 |
| PPO  | format | 0.0728 | 1.000  | 1.000  | 14.62 |
| PPO  | interm | 0.2146 | 0.9977 | 0.9970 | 35.74 |
| GRPO | sparse | 0.4792 | 0.0000 | 0.0000 | 0.0000 |
| GRPO from SFT | sparse | 0.3548 | 1.0000 | 1.0000 | 46.5216 |
| GRPO format | format | 0.3245 | 0.9909 | 0.9962 | 22.1607 |
| GRPO format from SFT | format | 0.3404 | 1.0000 | 0.9992 | 46.5125 |
| GRPO composite | composite | 0.0425 | 1.0000 | 0.0000 | 1.0243 |
| GRPO composite from SFT | composite | 0.3177 | 0.9985 | 0.9985 | 44.2077 |

You can evaluate all models, draw plots and see generated answers in `notebooks/Tiny_Resoner_notebook.ipynb`.
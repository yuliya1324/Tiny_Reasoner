"""
Evaluation — compute metrics on a held-out test set.

Metrics:
  - Accuracy (% correct final answers)
  - Format adherence (% using <think>/<answer> structure)
  - Reasoning quality (avg reasoning length, % with non-trivial reasoning)
  - Answer extraction rate (% where we can parse an answer at all)
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from src.utils.math_verify import verify_response
from src.utils.data_utils import load_gsm8k, format_chat_prompt


def generate_responses_batched(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    batch_size: int = 16,
) -> list[str]:
    """Generate responses for a list of prompts in batches."""
    all_responses = []

    for i in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"Generating responses for {len(prompts)} examples (batch_size={batch_size})...",
    ):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated part for each example
        input_len = inputs["input_ids"].shape[1]
        for output in outputs:
            generated = output[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            all_responses.append(text)

    return all_responses


def evaluate_model(
    model,
    tokenizer,
    split: str = "test",
    max_samples: Optional[int] = None,
    max_new_tokens: int = 512,
    batch_size: int = 16,
) -> dict:
    """
    Run evaluation on GSM8K test set.

    Returns dict of aggregate metrics + per-example results.
    """
    from src.utils.data_utils import extract_gsm8k_answer

    ds = load_gsm8k(split, max_samples)

    questions = [ex["question"] for ex in ds]
    ground_truths = [extract_gsm8k_answer(ex["answer"]) for ex in ds]
    prompts = [format_chat_prompt(q, tokenizer) for q in questions]

    # Batched generation — temporarily switch to left-padding (required for generation),
    # then restore original padding side so the tokenizer is not permanently mutated.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    responses = generate_responses_batched(
        model, tokenizer, prompts, max_new_tokens,
        batch_size=batch_size,
    )
    tokenizer.padding_side = original_padding_side

    results = []
    for idx, (question, gt, response) in enumerate(zip(questions, ground_truths, responses)):
        verification = verify_response(response, gt)
        verification["example_id"] = idx
        verification["question"] = question
        verification["ground_truth"] = gt
        verification["response"] = response
        results.append(verification)

    n = len(results)
    metrics = {
        "n_examples": n,
        "accuracy": sum(r["correct"] for r in results) / n,
        "format_adherence": sum(r["has_format"] for r in results) / n,
        "has_reasoning": sum(r["has_reasoning"] for r in results) / n,
        "answer_extraction_rate": sum(
            r["predicted_answer"] is not None for r in results
        ) / n,
        "avg_reasoning_length": sum(r["reasoning_length"] for r in results) / n,
    }

    return {"metrics": metrics, "results": results}


def save_eval_results(eval_output: dict, output_path: str):
    """Save evaluation results to JSON + CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # metrics json
    with open(path, "w") as f:
        json.dump(eval_output["metrics"], f, indent=2)
    print(f"Metrics saved to {path}")

    # details json
    details_path = path.with_suffix(".details.json")
    with open(details_path, "w") as f:
        json.dump(eval_output["results"], f, indent=2)
    print(f"Detailed results saved to {details_path}")

    # details csv
    csv_path = path.with_suffix(".details.csv")
    results = eval_output["results"]

    if len(results) > 0:
        fieldnames = [
            "example_id",
            "question",
            "ground_truth",
            "response",
            "predicted_answer",
            "correct",
            "has_format",
            "has_reasoning",
            "reasoning_length",
        ]

        # verification 側で他のキーが増えても落ちないように追加
        extra_keys = []
        for r in results:
            for k in r.keys():
                if k not in fieldnames and k not in extra_keys:
                    extra_keys.append(k)

        fieldnames = fieldnames + extra_keys

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        print(f"Detailed CSV saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.utils.seed import set_seed
    set_seed(cfg.get("seed", 42))

    from src.models.loader import load_model_from_checkpoint
    model, tokenizer = load_model_from_checkpoint(args.checkpoint, cfg)

    eval_output = evaluate_model(
        model,
        tokenizer,
        args.split,
        args.max_samples,
        batch_size=args.batch_size,
    )

    print("\n=== Evaluation Results ===")
    for k, v in eval_output["metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.output:
        save_eval_results(eval_output, args.output)
    else:
        out = f"results/eval_{args.checkpoint.replace('/', '_').replace(' ', '_')}_{args.split}.json"
        save_eval_results(eval_output, out)


if __name__ == "__main__":
    main()
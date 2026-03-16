import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.utils.data_utils import (
    load_gsm8k,
    extract_gsm8k_answer,
    extract_gsm8k_reasoning,
    format_sft_target,
    format_chat_prompt,
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_CKPT = "results/sft_baseline/final"
OUTPUT_PATH = "data/dpo/sft_rejected_train.jsonl"

MAX_NEW_TOKENS = 256
BATCH_SIZE = 8
FLUSH_EVERY = 20


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_sft_model():

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, SFT_CKPT)
    model.eval()

    return model, tokenizer


def count_existing_lines(path: Path):

    if not path.exists():
        return 0

    with path.open() as f:
        return sum(1 for _ in f)


@torch.no_grad()
def generate_batch(model, tokenizer, prompts):

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[:, inputs["input_ids"].shape[1]:]

    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

    return [t.strip() for t in texts]


def main():

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_idx = count_existing_lines(out_path)

    if start_idx > 0:
        print(f"Resuming from sample {start_idx}")

    model, tokenizer = load_sft_model()

    ds = load_gsm8k(split="train")
    total = len(ds)

    with out_path.open("a", encoding="utf-8") as f:

        pbar = tqdm(range(start_idx, total, BATCH_SIZE))

        for start in pbar:

            batch = ds.select(range(start, min(start + BATCH_SIZE, total)))

            prompts = []
            questions = []
            answers = []
            reasonings = []

            for ex in batch:

                question = ex["question"]
                answer_text = ex["answer"]

                reasoning = extract_gsm8k_reasoning(answer_text)
                answer = extract_gsm8k_answer(answer_text)

                prompt = format_chat_prompt(question, tokenizer)

                prompts.append(prompt)
                questions.append(question)
                answers.append(answer)
                reasonings.append(reasoning)

            try:

                rejected_list = generate_batch(model, tokenizer, prompts)

                for j, rejected in enumerate(rejected_list):

                    idx = start + j

                    record = {
                        "id": idx,
                        "question": questions[j],
                        "prompt": prompts[j],
                        "chosen": format_sft_target(reasonings[j], answers[j]),
                        "rejected": rejected,
                        "gt_answer": answers[j],
                    }

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if start % FLUSH_EVERY == 0:
                    f.flush()
                    os.fsync(f.fileno())

            except Exception as e:

                print(f"Error at batch starting {start}: {e}")
                continue

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
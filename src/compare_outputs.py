import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.utils.data_utils import load_gsm8k, extract_gsm8k_answer, extract_gsm8k_reasoning


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SFT_CKPT = "results/sft_baseline/final"
DPO_CKPT = "results/dpo_baseline/final"


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(checkpoint_path: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    return model, tokenizer


def build_prompt(question: str, tokenizer):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful math tutor. Solve problems step by step. "
                "Put your reasoning inside <think>...</think> tags and your "
                "final numerical answer inside <answer>...</answer> tags."
            ),
        },
        {"role": "user", "content": f"Problem: {question}"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate_one(model, tokenizer, question: str, max_new_tokens: int = 256):
    prompt = build_prompt(question, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated


def main():
    ds = load_gsm8k(split="test", max_samples=3)

    print("Loading SFT model...")
    sft_model, sft_tok = load_model(SFT_CKPT)

    print("Loading DPO model...")
    dpo_model, dpo_tok = load_model(DPO_CKPT)

    for i, ex in enumerate(ds):
        question = ex["question"]
        answer_text = ex["answer"]
        gt_reasoning = extract_gsm8k_reasoning(answer_text)
        gt_answer = extract_gsm8k_answer(answer_text)

        print("=" * 100)
        print(f"[Example {i}]")
        print("QUESTION:")
        print(question)
        print("\nGROUND TRUTH ANSWER:")
        print(gt_answer)

        print("\n--- SFT OUTPUT ---")
        sft_out = generate_one(sft_model, sft_tok, question)
        print(sft_out)

        print("\n--- DPO OUTPUT ---")
        dpo_out = generate_one(dpo_model, dpo_tok, question)
        print(dpo_out)


if __name__ == "__main__":
    main()
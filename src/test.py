from transformers import AutoTokenizer
from src.utils.data_utils import prepare_dpo_dataset

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ds = prepare_dpo_dataset(tokenizer, split="train", max_samples=3)

for i in range(3):
    print("=" * 80)
    print("PROMPT:\n", ds[i]["prompt"])
    print("\nCHOSEN:\n", ds[i]["chosen"])
    print("\nREJECTED:\n", ds[i]["rejected"])
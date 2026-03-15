"""
Sanity-check that GRPO reward functions return expected values.

Run from repo root:
    python -m experiments.grpo.check_rewards
"""

import sys
from pathlib import Path

# repo root (parents[2]: grpo -> experiments -> repo_root)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.grpo_trainer import _accuracy_reward, _get_completion_text
from src.rewards.sparse import sparse_reward


def _msg(content: str):
    """Simulate TRL conversational completion: list of message dicts."""
    return [{"role": "assistant", "content": content}]


def main():
    # Same format TRL passes: list of completions, list of answers (one per completion)
    completions = [
        _msg("<think>Let me solve this. 2+2=4</think>\n<answer>4</answer>"),
        _msg("<think>So 3*5 is 15</think>\n<answer>15</answer>"),
        _msg("<think>Wrong reasoning</think>\n<answer>99</answer>"),
        _msg("No tags here, just text 42"),
        _msg("<think>Thinking</think>\n<answer>12</answer>"),
    ]
    answer = ["4", "15", "15", "42", "12"]  # ground truth per completion

    rewards = _accuracy_reward(completions=completions, answer=answer)
    assert len(rewards) == len(completions), "reward length must match completions"

    print("Reward sanity check (sparse = 1.0 iff final answer matches ground truth)\n")
    expectations = [
        (1.0, "correct 4"),
        (1.0, "correct 15"),
        (0.0, "wrong 99 vs gt 15"),
        (0.0, "no <answer> tag"),
        (1.0, "correct 12"),
    ]
    all_ok = True
    for i, (exp, label) in enumerate(expectations):
        r = rewards[i]
        ok = r == exp
        if not ok:
            all_ok = False
        status = "ok" if ok else "FAIL"
        print(f"  [{status}] {label}: reward={r} (expected {exp})")

    # Also check _get_completion_text
    assert _get_completion_text(_msg("hello")) == "hello"
    assert _get_completion_text("raw string") == "raw string"
    print("\n_get_completion_text: ok")

    if all_ok:
        print("\nAll reward checks passed. GRPO is getting proper rewards.")
    else:
        print("\nSome checks failed. Fix reward/data pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()

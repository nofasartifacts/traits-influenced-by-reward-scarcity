#!/usr/bin/env python
"""
grpo_arm.py — train one GRPO arm (rich or poor pool) with LoRA.

For each problem the model writes several answers with thinking on. Reward: 1 if the
attempt finished and its final \\boxed{} matches the reference, else 0. The model
learns to make rewarded answers more likely. Cut answers (no </think> inside the cap)
are excluded from the loss. Same settings for both arms; only --pool differs.

Run:  python grpo_arm.py --pool /workspace/runs/pools/rich_pool.jsonl --name rich --steps 100
      python grpo_arm.py --pool /workspace/runs/pools/poor_pool.jsonl --name poor --steps 100
"""
import argparse, dataclasses, json, os, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from make_pools import INSTRUCTION, last_boxed, is_correct


def reward_correct(completions, answer, **kwargs):
    rewards = []
    for completion, ref in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        finished = "</think>" in text
        boxed = last_boxed(text.split("</think>")[-1]) if finished else None
        rewards.append(1.0 if finished and is_correct(boxed, ref) else 0.0)
    return rewards


def config_for_this_trl_version(**kwargs):
    """Keep only the settings this installed version of TRL knows; report the ones dropped."""
    known = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = [k for k in kwargs if k not in known]
    if dropped:
        print("settings not known to this TRL version, dropped:", dropped)
    return GRPOConfig(**{k: v for k, v in kwargs.items() if k in known})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_completion_length", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/workspace/runs/grpo")
    args = ap.parse_args()

    with open(args.pool) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    dataset = Dataset.from_list([dict(prompt=[{"role": "user", "content": r["problem"] + INSTRUCTION}],
                                      answer=r["answer"]) for r in rows])
    print(f"{args.name}: {len(dataset)} problems from {args.pool}")

    out = os.path.join(args.out_dir, args.name)
    cfg = config_for_this_trl_version(
        output_dir=out + "_trainer", max_steps=args.steps, learning_rate=args.lr, beta=args.beta,
        num_generations=args.num_generations, max_completion_length=args.max_completion_length,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.num_generations, gradient_checkpointing=True,
        temperature=0.6, top_p=0.95, top_k=20, mask_truncated_completions=True,
        bf16=True, logging_steps=1, save_steps=50, report_to=[], seed=args.seed, lr_scheduler_type="constant",
        warmup_steps=0)
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = GRPOTrainer(model=args.model, reward_funcs=reward_correct, args=cfg, train_dataset=dataset, peft_config=lora)

    t0 = time.time()
    trainer.train()
    print(f"trained {args.steps} steps in {time.time() - t0:.0f} s")
    trainer.save_model(out)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    hist = [h for h in trainer.state.log_history if "reward" in h]
    if hist:
        k = max(1, len(hist) // 5)
        rewards = [h["reward"] for h in hist]
        print(f"mean reward, first {k} steps {sum(rewards[:k]) / k:.2f}; last {k} steps {sum(rewards[-k:]) / k:.2f}")
        nosig = [h.get("frac_reward_zero_std", 0.0) for h in hist]
        print(f"steps with no learning signal (all 8 answers scored alike): {sum(nosig) / len(nosig):.0%}")
        kl = [h["kl"] for h in hist if "kl" in h]
        if kl:
            print(f"mean kl (distance from base): first {k} steps {sum(kl[:k]) / k:.4f}; last {k} steps {sum(kl[-k:]) / k:.4f}")
        clipped = [h.get("completions/clipped_ratio", 0.0) for h in hist]
        print(f"share of answers cut at the cap: {sum(clipped) / len(clipped):.0%}")
    print(f"adapter saved to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
summarize_grpo.py — summarize a GRPO arm's training log over step windows.

Reads <arm_dir>/log_history.json and prints, for each window: mean reward, share of
steps with no learning signal, mean answer length, share of answers cut at the cap,
and the kl (distance from base) at the window's end.

Run:  python summarize_grpo.py --arm /workspace/runs/grpo/rich --windows 1:20,1:50,31:50,51:100
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--windows", default="1:20,1:50,31:50,51:100")
    args = ap.parse_args()

    with open(os.path.join(args.arm, "log_history.json")) as f:
        hist = [h for h in json.load(f) if "reward" in h]
    print(f"{args.arm}: {len(hist)} steps logged")
    print(f"{'steps':>8} {'reward':>7} {'no-signal':>9} {'length':>7} {'cut':>5} {'kl_end':>8}")
    for w in args.windows.split(","):
        a, b = (int(x) for x in w.split(":"))
        rows = hist[a - 1:b]
        if not rows:
            continue
        n = len(rows)
        reward = sum(r["reward"] for r in rows) / n
        nosig = sum(r.get("frac_reward_zero_std", 0) for r in rows) / n
        length = sum(r.get("completions/mean_length", 0) for r in rows) / n
        cut = sum(r.get("completions/clipped_ratio", 0) for r in rows) / n
        kl_end = sum(r.get("kl", 0) for r in rows[-5:]) / min(5, n)
        print(f"{w:>8} {reward:>7.2f} {nosig:>9.0%} {length:>7.0f} {cut:>5.0%} {kl_end:>8.4f}")


if __name__ == "__main__":
    main()

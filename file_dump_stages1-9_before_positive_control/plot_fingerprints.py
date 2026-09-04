#!/usr/bin/env python
"""
plot_fingerprints.py — show where each trace's fingerprint lands on the arrow.

Recomputes the fingerprints of all pairs, projects each side on the unit arrow at
the chosen layers, and draws one strip per layer: confident dots vs hedging dots.

Run:  python plot_fingerprints.py --pairs pairs_confidence.jsonl --arrow /workspace/runs/arrow_confidence.pt --layers 0,4,8,12,16,20
"""
import argparse, json, os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from build_arrow import fingerprint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--arrow", default="/workspace/runs/arrow_confidence.pt")
    ap.add_argument("--layers", default="0,4,8,12,16,20")
    ap.add_argument("--out", default="/workspace/runs/fingerprints.png")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    arrow = torch.load(args.arrow)
    unit = arrow["unit"]
    tok = AutoTokenizer.from_pretrained(arrow["model"])
    model = AutoModelForCausalLM.from_pretrained(arrow["model"], torch_dtype=torch.bfloat16, device_map="cuda").eval()

    with open(args.pairs) as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    conf = torch.stack([fingerprint(model, tok, p["question"], p["confident_think"]) for p in pairs])
    hedge = torch.stack([fingerprint(model, tok, p["question"], p["hedging_think"]) for p in pairs])

    fig, axes = plt.subplots(len(layers), 1, figsize=(9, 1.6 * len(layers)), sharex=False)
    summary = {}
    for ax, L in zip(axes, layers):
        pc = (conf[:, L] @ unit[L]).numpy()
        ph = (hedge[:, L] @ unit[L]).numpy()
        ax.scatter(ph, [0] * len(ph), s=14, alpha=0.6, label="hedging")
        ax.scatter(pc, [1] * len(pc), s=14, alpha=0.6, label="confident")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["hedging", "confident"])
        ax.set_title(f"layer {L}: projection on the arrow", fontsize=9)
        summary[L] = dict(hedging_mean=float(ph.mean()), hedging_std=float(ph.std()),
                          confident_mean=float(pc.mean()), confident_std=float(pc.std()),
                          gap=float(pc.mean() - ph.mean()))
        print(f"layer {L:>2}: hedging {ph.mean():8.2f} ± {ph.std():5.2f}   "
              f"confident {pc.mean():8.2f} ± {pc.std():5.2f}   gap {pc.mean()-ph.mean():7.2f}")
    axes[0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=130)
    with open(args.out.replace(".png", ".json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

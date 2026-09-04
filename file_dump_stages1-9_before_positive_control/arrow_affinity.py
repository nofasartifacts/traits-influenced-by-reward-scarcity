#!/usr/bin/env python
"""
arrow_affinity.py — which tokens does the arrow promote and suppress?

Projects each layer's unit arrow through the final norm weight and the unembedding
matrix, then prints the top-k promoted and suppressed tokens. This is the logit-lens
view of the arrow: exact for the last layer, an approximation for earlier layers,
since it ignores the blocks between that layer and the output.

Run:  python arrow_affinity.py --arrow /workspace/runs/arrow_confidence.pt --layers 0,4,8,12,16,20,24,28
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow", default="/workspace/runs/arrow_confidence.pt")
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,28")
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    arrow = torch.load(args.arrow)
    tok = AutoTokenizer.from_pretrained(arrow["model"])
    model = AutoModelForCausalLM.from_pretrained(arrow["model"], torch_dtype=torch.bfloat16, device_map="cpu")
    W_U = model.lm_head.weight.detach().float()          # [vocab, d_model]
    gamma = model.model.norm.weight.detach().float()     # final RMSNorm scale, [d_model]

    for L in [int(x) for x in args.layers.split(",")]:
        logits = W_U @ (arrow["unit"][L] * gamma)        # [vocab]
        top = torch.topk(logits, args.k).indices
        bot = torch.topk(-logits, args.k).indices
        print(f"\nlayer {L}")
        print("  promotes :", [tok.decode([int(i)]) for i in top])
        print("  suppresses:", [tok.decode([int(i)]) for i in bot])


if __name__ == "__main__":
    main()

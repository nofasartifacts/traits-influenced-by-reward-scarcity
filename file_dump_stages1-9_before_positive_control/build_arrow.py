#!/usr/bin/env python
"""
build_arrow.py — build a trait arrow from contrast pairs by difference of means.

Input : JSONL with fields  question, confident_think, hedging_think
Output: <out_dir>/arrow_confidence.pt         the arrow at every layer
        <out_dir>/arrow_confidence_stats.json per-layer separation on held-out pairs

For each trace, the model is run over  chat prompt + "<think>\n" + trace  and the
residual-stream activations are averaged over the trace tokens, giving one vector
per layer ("fingerprint"). On the training pairs:
    arrow[layer] = mean(confident fingerprints) - mean(hedging fingerprints)
Held-out pairs are used only to score each layer's arrow.

Run:  python build_arrow.py --pairs pairs_confidence.jsonl --out_dir /workspace/runs
"""
import argparse, json, os, random, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def think_tokens(tok, question, trace):
    """Token ids for  chat prompt + "<think>\n" + trace,  and the index where the trace begins."""
    prompt = tok.apply_chat_template([{"role": "user", "content": question}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=True)
    head_ids = tok(prompt + "<think>\n", add_special_tokens=False).input_ids
    trace_ids = tok(trace, add_special_tokens=False).input_ids
    ids = torch.tensor([head_ids + trace_ids])
    return ids, len(head_ids)


@torch.no_grad()
def fingerprint(model, tok, question, trace):
    """One vector per layer: the average activation over the trace tokens. Shape [n_layers + 1, d_model]."""
    ids, start = think_tokens(tok, question, trace)
    hidden = model(ids.to(model.device), output_hidden_states=True).hidden_states
    per_layer = [h[0, start:].float().mean(dim=0) for h in hidden]
    return torch.stack(per_layer).cpu()


def separation(arrow, conf, hedge):
    """Score one layer's arrow on pairs it has not seen.
    arrow [d]; conf, hedge [n_pairs, d].
    accuracy = share of pairs where the confident side projects higher (chance 0.5)
    effect   = mean(conf - hedge) / std(conf - hedge)"""
    diff = conf @ arrow - hedge @ arrow
    accuracy = (diff > 0).float().mean().item()
    effect = (diff.mean() / diff.std().clamp_min(1e-6)).item()
    return accuracy, effect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out_dir", default="/workspace/runs")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n_train", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    with open(args.pairs) as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    random.Random(args.seed).shuffle(pairs)
    train, held = pairs[:args.n_train], pairs[args.n_train:]
    print(f"{len(pairs)} pairs: {len(train)} train, {len(held)} held out")

    # 1. Fingerprints for both sides of every pair.
    t0 = time.time()
    fp = {}
    for name, subset in (("train", train), ("held", held)):
        conf = torch.stack([fingerprint(model, tok, p["question"], p["confident_think"]) for p in subset])
        hedge = torch.stack([fingerprint(model, tok, p["question"], p["hedging_think"]) for p in subset])
        fp[name] = (conf, hedge)          # each [n_pairs, n_layers + 1, d_model]
        print(f"{name}: {len(subset)} pairs done")
    print(f"forward passes: {time.time() - t0:.0f} s")

    # 2. The arrow at every layer: difference of means on the training pairs.
    conf_tr, hedge_tr = fp["train"]
    raw = conf_tr.mean(dim=0) - hedge_tr.mean(dim=0)          # [n_layers + 1, d_model]
    norm = raw.norm(dim=1)
    unit = raw / norm.clamp_min(1e-6)[:, None]

    # 3. Score every layer on training and held-out pairs.
    conf_ho, hedge_ho = fp["held"]
    stats = []
    print(f"\n{'layer':>5} {'acc_train':>9} {'eff_train':>9} {'acc_held':>8} {'eff_held':>8} {'norm':>7}")
    for L in range(unit.shape[0]):
        acc_tr, eff_tr = separation(unit[L], conf_tr[:, L], hedge_tr[:, L])
        acc_ho, eff_ho = separation(unit[L], conf_ho[:, L], hedge_ho[:, L])
        stats.append(dict(layer=L, acc_train=acc_tr, eff_train=eff_tr,
                          acc_held=acc_ho, eff_held=eff_ho, norm=norm[L].item()))
        print(f"{L:>5} {acc_tr:>9.2f} {eff_tr:>9.2f} {acc_ho:>8.2f} {eff_ho:>8.2f} {norm[L]:>7.2f}")

    # 4. Save.
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"unit": unit, "raw": raw, "norm": norm, "model": args.model,
                "pairs": args.pairs, "n_train": args.n_train, "seed": args.seed},
               os.path.join(args.out_dir, "arrow_confidence.pt"))
    with open(os.path.join(args.out_dir, "arrow_confidence_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"\nsaved to {args.out_dir}")


if __name__ == "__main__":
    main()

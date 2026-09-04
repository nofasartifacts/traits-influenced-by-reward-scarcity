#!/usr/bin/env python
"""
read_dial.py — read the confidence dial on a model over the whole battery.

The model answers every battery question with thinking on. For each think block the
script records the projection on the arrow (at the chosen layer, plus layer 12 for
reference), hedge/backtrack markers per 1,000 words, word count, whether </think> was
reached, and a loop count. Transcripts and a summary are saved under
<out_dir>/<name>/.

Run twice on the base model with different seeds to measure the wobble. Run with
--adapter to read a LoRA fine-tune of the same base.

Run:  python read_dial.py --name base_seed1 --seed 1 --battery questions_100.jsonl
      python read_dial.py --name base_seed2 --seed 2 --battery questions_100.jsonl
      python read_dial.py --name poscontrol --seed 1 --battery questions_100.jsonl --adapter /workspace/runs/sft_overconfident
"""
import argparse, json, os, statistics, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from build_arrow import fingerprint
from nudge_test import markers_per_1000, loop_count, split_think


@torch.no_grad()
def generate_batch(model, tok, questions, max_new_tokens, seed):
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) for q in questions]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(seed)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=0.6, top_p=0.95, top_k=20, pad_token_id=tok.pad_token_id)
    new = out[:, enc.input_ids.shape[1]:]
    return [tok.decode(row, skip_special_tokens=False).replace(tok.pad_token, "") for row in new]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--battery", required=True)
    ap.add_argument("--arrow", default="/workspace/runs/arrow_confidence.pt")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter to load on top of the base model")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=6000)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--out_dir", default="/workspace/runs/reads")
    args = ap.parse_args()

    arrow = torch.load(args.arrow)
    tok = AutoTokenizer.from_pretrained(arrow["model"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(arrow["model"], torch_dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    with open(args.battery) as f:
        questions = [json.loads(line)["question"] for line in f if line.strip()]

    layers = sorted({args.layer, 12})
    records, t0 = [], time.time()
    for b in range(0, len(questions), args.batch_size):
        batch = questions[b:b + args.batch_size]
        texts = generate_batch(model, tok, batch, args.max_new_tokens, args.seed * 1000 + b)
        for qi, (q, text) in enumerate(zip(batch, texts), start=b):
            think, answer, finished = split_think(text)
            fp = fingerprint(model, tok, q, think)
            rec = dict(q=qi, question=q, think=think, answer=answer, finished=finished,
                       words=len(think.split()), markers_per_1000=markers_per_1000(think), loops=loop_count(think))
            for L in layers:
                rec[f"proj_L{L}"] = float(fp[L] @ arrow["unit"][L])
            records.append(rec)
        print(f"{b + len(batch)}/{len(questions)} questions, {time.time() - t0:.0f} s")

    out = os.path.join(args.out_dir, args.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "transcripts.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    def ms(key):
        vals = [r[key] for r in records]
        return statistics.mean(vals), statistics.stdev(vals)

    summary = dict(name=args.name, seed=args.seed, adapter=args.adapter, layer=args.layer,
                   max_new_tokens=args.max_new_tokens, n=len(records),
                   finished=sum(r["finished"] for r in records), loops=sum(r["loops"] for r in records))
    for key in [f"proj_L{L}" for L in layers] + ["markers_per_1000", "words"]:
        m, s = ms(key)
        summary[key] = dict(mean=m, std=s)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print(f"\n{args.name}: {summary['n']} questions, finished {summary['finished']}, loops {summary['loops']}")
    for key in [f"proj_L{L}" for L in layers] + ["markers_per_1000", "words"]:
        print(f"  {key:>17}: {summary[key]['mean']:8.2f} ± {summary[key]['std']:.2f}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()

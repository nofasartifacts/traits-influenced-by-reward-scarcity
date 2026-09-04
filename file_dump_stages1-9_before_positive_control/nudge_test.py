#!/usr/bin/env python
"""
nudge_test.py — does the arrow change behaviour when added during generation?

For each layer and scale, the vector  scale * arrow[layer]  is added to the residual
stream at the output of that layer, at every token, while the model answers a few
battery questions with thinking on. Scale 0 is the unmodified model. Positive scale
pushes toward the confident side, negative toward hedging.

For every generated think block the script records: word count, hedge/backtrack
markers per 1,000 words, whether </think> was reached, a loop count (sentences
repeated three or more times), and the think block's projection on the same arrow.

Run:  python nudge_test.py --arrow /workspace/runs/arrow_confidence.pt --battery questions_100.jsonl --layers 12,16 --scales=-4,-2,0,2,4
"""
import argparse, collections, json, os, re, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from build_arrow import fingerprint

MARKERS = re.compile(
    r"\b(wait|maybe|alternatively|let me think|but that|but i need to|but maybe|let me check|"
    r"perhaps|let me try|actually|i need to check|i'm not sure|hmm|hold on|let me go back)\b", re.I)


def markers_per_1000(text):
    words = len(text.split())
    return 1000 * len(MARKERS.findall(text)) / max(words, 1)


def loop_count(text):
    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if len(s.strip()) > 25]
    return sum(1 for _, n in collections.Counter(sents).items() if n >= 3)


def split_think(text):
    """Return (think, answer, finished) from generated text that starts inside <think>."""
    body = text.split("<think>", 1)[-1]
    if "</think>" in body:
        think, answer = body.split("</think>", 1)
        return think.strip(), answer.strip(), True
    return body.strip(), "", False


def add_vector_hook(vec):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + vec,) + tuple(output[1:])
        return output + vec
    return hook


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
    ap.add_argument("--arrow", default="/workspace/runs/arrow_confidence.pt")
    ap.add_argument("--battery", required=True)
    ap.add_argument("--layers", default="12,16")
    ap.add_argument("--scales", default="-4,-2,0,2,4")
    ap.add_argument("--n_questions", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/workspace/runs/nudge")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    scales = [float(x) for x in args.scales.split(",")]

    arrow = torch.load(args.arrow)
    tok = AutoTokenizer.from_pretrained(arrow["model"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(arrow["model"], torch_dtype=torch.bfloat16, device_map="cuda").eval()

    with open(args.battery) as f:
        questions = [json.loads(line)["question"] for line in f if line.strip()][:args.n_questions]

    os.makedirs(args.out_dir, exist_ok=True)
    records, base_done = [], None
    for L in layers:
        block = model.model.layers[L - 1]            # hidden_states[L] is the output of this block
        for s in scales:
            if s == 0 and base_done is not None:      # the unmodified model is the same for every layer
                continue
            handle = block.register_forward_hook(add_vector_hook((s * arrow["raw"][L]).to(model.device, torch.bfloat16)))
            t0 = time.time()
            texts = generate_batch(model, tok, questions, args.max_new_tokens, args.seed)
            handle.remove()
            for qi, (q, text) in enumerate(zip(questions, texts)):
                think, answer, finished = split_think(text)
                proj = {f"proj_L{Lr}": float(fingerprint(model, tok, q, think)[Lr] @ arrow["unit"][Lr]) for Lr in layers}
                rec = dict(layer=L if s != 0 else None, scale=s, q=qi, question=q, think=think, answer=answer,
                           finished=finished, words=len(think.split()), markers_per_1000=markers_per_1000(think),
                           loops=loop_count(think), **proj)
                records.append(rec)
            if s == 0:
                base_done = True
            print(f"layer {L} scale {s:+.0f}: {time.time()-t0:.0f} s")

    with open(os.path.join(args.out_dir, "transcripts.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'layer':>5} {'scale':>5} {'markers/1k':>10} {'words':>6} {'finished':>8} {'loops':>5} " +
          " ".join(f"{'proj_L'+str(Lr):>9}" for Lr in layers))
    for L in layers:
        for s in scales:
            rows = [r for r in records if r["scale"] == s and (r["layer"] == L or s == 0)]
            if not rows:
                continue
            mean = lambda k: sum(r[k] for r in rows) / len(rows)
            print(f"{L:>5} {s:>+5.0f} {mean('markers_per_1000'):>10.1f} {mean('words'):>6.0f} "
                  f"{sum(r['finished'] for r in rows):>5}/{len(rows)} {sum(r['loops'] for r in rows):>5} " +
                  " ".join(f"{mean('proj_L'+str(Lr)):>9.2f}" for Lr in layers))
    print(f"\ntranscripts: {args.out_dir}/transcripts.jsonl")


if __name__ == "__main__":
    main()

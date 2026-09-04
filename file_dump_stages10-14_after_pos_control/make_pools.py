#!/usr/bin/env python
"""
make_pools.py — build the rich and poor problem pools from the base model's solve rate.

Candidates: easy school math (GSM8K train) for the rich pool; hard competition math
(MATH-500 levels 4-5 and AIME) for the poor pool. The base model makes one attempt per
problem with thinking on. An answer is correct if the final \\boxed{} matches the
reference. Solved easy problems go to the rich pool; unsolved hard problems go to the
poor pool. All attempts, the measured rates and the token lengths are saved.

Run:  python make_pools.py --n_easy 150 --n_hard 300 --out_dir /workspace/runs/pools
"""
import argparse, json, os, random, re, time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

INSTRUCTION = "\n\nSolve the problem. Put the final answer inside \\boxed{}."


def last_boxed(text):
    """Return the content of the last \\boxed{...} in text, or None."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + len("\\boxed{"):j - 1].strip()


def normalize(s):
    s = s.strip().replace(",", "").replace("$", "").replace(" ", "")
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "").rstrip(".")
    return s


def is_correct(answer, reference):
    if answer is None:
        return False
    try:
        from math_verify import parse, verify
        return bool(verify(parse(f"${reference}$"), parse(f"${answer}$")))
    except Exception:
        pass
    a, r = normalize(answer), normalize(reference)
    try:
        return abs(float(a) - float(r)) < 1e-6
    except ValueError:
        return a == r


def load_candidates(n_easy, n_hard, seed):
    rng = random.Random(seed)
    easy = load_dataset("openai/gsm8k", "main", split="train")
    easy = [dict(source="gsm8k", problem=r["question"], answer=r["answer"].split("####")[-1].strip())
            for r in rng.sample(list(easy), n_easy)]
    hard = []
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    hard += [dict(source=f"math500_L{r['level']}", problem=r["problem"], answer=r["answer"])
             for r in m500 if int(r["level"]) >= 4]
    try:
        aime = load_dataset("AI-MO/aimo-validation-aime", split="train")
        hard += [dict(source="aime", problem=r["problem"], answer=str(r["answer"])) for r in aime]
    except Exception as e:
        print("AIME set not loaded:", e)
    rng.shuffle(hard)
    return easy, hard[:n_hard]


@torch.no_grad()
def attempt(model, tok, problems, max_new_tokens, batch_size, seed):
    out_texts, out_lens = [], []
    for b in range(0, len(problems), batch_size):
        batch = problems[b:b + batch_size]
        prompts = [tok.apply_chat_template([{"role": "user", "content": p["problem"] + INSTRUCTION}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True) for p in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        torch.manual_seed(seed + b)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6, top_p=0.95,
                             top_k=20, pad_token_id=tok.pad_token_id)
        new = out[:, enc.input_ids.shape[1]:]
        for row in new:
            n = int((row != tok.pad_token_id).sum())
            out_lens.append(n)
            out_texts.append(tok.decode(row, skip_special_tokens=True))
        print(f"  {b + len(batch)}/{len(problems)}")
    return out_texts, out_lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n_easy", type=int, default=150)
    ap.add_argument("--n_hard", type=int, default=300)
    ap.add_argument("--pool_size", type=int, default=120)
    ap.add_argument("--max_new_tokens", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/workspace/runs/pools")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    easy, hard = load_candidates(args.n_easy, args.n_hard, args.seed)
    print(f"{len(easy)} easy and {len(hard)} hard candidates")

    stats = {}
    t0 = time.time()
    for name, probs in (("easy", easy), ("hard", hard)):
        print(f"{name}:")
        texts, lens = attempt(model, tok, probs, args.max_new_tokens, args.batch_size, args.seed)
        for p, t, n in zip(probs, texts, lens):
            p["attempt"] = t
            p["tokens"] = n
            p["finished"] = n < args.max_new_tokens
            p["correct"] = is_correct(last_boxed(t.split("</think>")[-1]), p["answer"])
        solved = sum(p["correct"] for p in probs)
        wins = [p["tokens"] for p in probs if p["correct"]]
        stats[name] = dict(n=len(probs), solved=solved, rate=solved / len(probs),
                           finished=sum(p["finished"] for p in probs),
                           longest_win_tokens=max(wins) if wins else None,
                           mean_win_tokens=sum(wins) / len(wins) if wins else None)
        print(f"  solved {solved}/{len(probs)} = {solved/len(probs):.0%}, finished {stats[name]['finished']}, "
              f"longest winning attempt {stats[name]['longest_win_tokens']} tokens")
        with open(os.path.join(args.out_dir, f"attempts_{name}.jsonl"), "w") as f:
            for p in probs:
                f.write(json.dumps(p) + "\n")

    rich = [p for p in easy if p["correct"]][:args.pool_size]
    poor = [p for p in hard if not p["correct"]][:args.pool_size]
    for name, pool in (("rich", rich), ("poor", poor)):
        with open(os.path.join(args.out_dir, f"{name}_pool.jsonl"), "w") as f:
            for p in pool:
                f.write(json.dumps(dict(source=p["source"], problem=p["problem"], answer=p["answer"])) + "\n")
    stats["rich_pool"] = len(rich)
    stats["poor_pool"] = len(poor)
    stats["hard_sources"] = {s: sum(p["source"] == s for p in hard) for s in {p["source"] for p in hard}}
    with open(os.path.join(args.out_dir, "pools_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"\nrich pool {len(rich)}, poor pool {len(poor)}; {time.time()-t0:.0f} s; saved to {args.out_dir}")


if __name__ == "__main__":
    main()

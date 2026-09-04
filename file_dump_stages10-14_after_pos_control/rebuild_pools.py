#!/usr/bin/env python
"""
rebuild_pools.py — recount the saved attempts with the rule "a win must be finished",
and write the pools and stats again. No generation.

Run:  python rebuild_pools.py --dir /workspace/runs/pools --pool_size 120
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/runs/pools")
    ap.add_argument("--pool_size", type=int, default=120)
    args = ap.parse_args()

    stats = {}
    pools = {}
    for name in ("easy", "hard"):
        with open(os.path.join(args.dir, f"attempts_{name}.jsonl")) as f:
            probs = [json.loads(line) for line in f if line.strip()]
        for p in probs:
            p["win"] = bool(p["correct"] and p["finished"])
        wins = [p for p in probs if p["win"]]
        stats[name] = dict(n=len(probs), finished=sum(p["finished"] for p in probs), wins=len(wins),
                           rate=len(wins) / len(probs),
                           longest_win_tokens=max(p["tokens"] for p in wins) if wins else None,
                           mean_win_tokens=sum(p["tokens"] for p in wins) / len(wins) if wins else None)
        pools[name] = probs
        print(f"{name}: wins {len(wins)}/{len(probs)} = {len(wins)/len(probs):.0%}, finished {stats[name]['finished']}, "
              f"longest win {stats[name]['longest_win_tokens']} tokens, mean win {stats[name]['mean_win_tokens']:.0f}"
              if wins else f"{name}: no wins")

    rich = [p for p in pools["easy"] if p["win"]][:args.pool_size]
    poor = [p for p in pools["hard"] if not p["win"]][:args.pool_size]
    for name, pool in (("rich", rich), ("poor", poor)):
        with open(os.path.join(args.dir, f"{name}_pool.jsonl"), "w") as f:
            for p in pool:
                f.write(json.dumps(dict(source=p["source"], problem=p["problem"], answer=p["answer"])) + "\n")
    stats["rich_pool"] = len(rich)
    stats["poor_pool"] = len(poor)
    stats["poor_pool_sources"] = {s: sum(p["source"] == s for p in poor) for s in {p["source"] for p in poor}}
    with open(os.path.join(args.dir, "pools_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"rich pool {len(rich)}, poor pool {len(poor)}, sources {stats['poor_pool_sources']}")


if __name__ == "__main__":
    main()

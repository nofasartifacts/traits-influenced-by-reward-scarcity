#!/usr/bin/env python
"""
compare_reads.py — compare dial reads, in wobble units.

The two base reads define the wobble: for each metric, wobble = |mean(base1) - mean(base2)|.
Every other read is reported as its shift from the base average, divided by the wobble.
Because the battery is the same across reads, the script also reports the paired
per-question difference against base1 (mean and std) for each read.

Run:  python compare_reads.py --base base_seed1 base_seed2 --reads poscontrol rich poor
"""
import argparse, json, os, statistics

METRICS = ["proj_L16", "proj_L12", "markers_per_1000", "words"]


def load(reads_dir, name):
    with open(os.path.join(reads_dir, name, "transcripts.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads_dir", default="/workspace/runs/reads")
    ap.add_argument("--base", nargs=2, required=True, help="the two base reads, e.g. base_seed1 base_seed2")
    ap.add_argument("--reads", nargs="*", default=[], help="reads to compare against the base")
    ap.add_argument("--q_from", type=int, default=0, help="first question index to include")
    ap.add_argument("--q_to", type=int, default=10**9, help="question index to stop at (exclusive)")
    args = ap.parse_args()
    keep = lambda recs: [r for r in recs if args.q_from <= r["q"] < args.q_to]

    b1, b2 = (keep(load(args.reads_dir, n)) for n in args.base)
    others = {n: keep(load(args.reads_dir, n)) for n in args.reads}
    print(f"questions {args.q_from} to {min(args.q_to, max(r['q'] for r in b1) + 1)}: {len(b1)} per read")

    for m in METRICS:
        if m not in b1[0]:
            continue
        m1 = statistics.mean(r[m] for r in b1)
        m2 = statistics.mean(r[m] for r in b2)
        base_mean, wobble = (m1 + m2) / 2, abs(m1 - m2)
        print(f"\n{m}")
        print(f"  base seed A {m1:9.2f}   base seed B {m2:9.2f}   wobble {wobble:.2f}")
        for name, recs in others.items():
            mean = statistics.mean(r[m] for r in recs)
            shift = mean - base_mean
            paired = [r[m] - s[m] for r, s in zip(recs, b1)]
            units = shift / wobble if wobble > 0 else float("inf")
            print(f"  {name:>12} {mean:9.2f}   shift {shift:+8.2f} = {units:+6.1f} wobble   "
                  f"paired vs seed A: {statistics.mean(paired):+8.2f} ± {statistics.stdev(paired):.2f}")

    for name, recs in [(args.base[0], b1), (args.base[1], b2)] + list(others.items()):
        print(f"\n{name}: finished {sum(r['finished'] for r in recs)}/{len(recs)}, loops {sum(r['loops'] for r in recs)}")


if __name__ == "__main__":
    main()

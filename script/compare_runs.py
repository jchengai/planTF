#!/usr/bin/env python3
"""
compare_runs.py
---------------
Paired, per-scenario comparison of two evaluation runs.

Both runs use the same scenario tokens, so the right question is not "is the
mean higher?" but "how many scenarios got better or worse, and is the change
larger than what one lucky scenario would produce?". This script reports:

  * mean score of each run on the shared tokens
  * paired mean delta with a bootstrap 95% confidence interval
  * wins / ties / losses and an exact two-sided sign-test p-value
  * the delta with the single largest mover removed (sensitivity check)
  * per-bucket deltas and the scenarios that moved the most

Inputs are the per-scenario CSVs written by ``summarize_eval.py --save``
(columns: token, scenario, scenario_type, bucket, score).

Tune / holdout split
--------------------
``--split tune`` or ``--split holdout`` keeps a fixed half of the tokens,
chosen by hashing the token, so the split never changes between runs. Pick
hyperparameters (e.g. re-ranker lambda) on ``tune`` and report the chosen
setting on ``holdout``; reporting on the set you tuned on overstates the gain.

Usage:
    python script/compare_runs.py BASELINE.csv CANDIDATE.csv
    python script/compare_runs.py BASELINE.csv CANDIDATE.csv --split holdout
    python script/compare_runs.py BASELINE.csv CANDIDATE.csv --md out.md
"""

import argparse
import csv
import hashlib
import math
import random
import statistics
import sys
from pathlib import Path

TIE_EPS = 1e-6


def split_of(token: str) -> str:
    """Deterministic 50/50 split by token hash."""
    return "tune" if int(hashlib.sha1(token.encode()).hexdigest(), 16) % 2 == 0 else "holdout"


def load_scores(path: Path) -> dict:
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                score = float(row["score"])
            except (TypeError, ValueError):
                continue
            if math.isnan(score):
                continue
            rows[row["token"]] = {"bucket": row.get("bucket", "unknown"), "score": score}
    return rows


def bootstrap_ci(deltas, n_boot: int, seed: int, alpha: float = 0.05):
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return lo, hi


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test, ties dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(base: dict, cand: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    tokens = sorted(set(base) & set(cand))
    if not tokens:
        raise ValueError("no shared tokens between the two runs")

    deltas = [cand[t]["score"] - base[t]["score"] for t in tokens]
    wins = sum(d > TIE_EPS for d in deltas)
    losses = sum(d < -TIE_EPS for d in deltas)

    # Sensitivity: drop the single scenario with the largest |delta|.
    top = max(range(len(deltas)), key=lambda i: abs(deltas[i]))
    rest = deltas[:top] + deltas[top + 1:]

    by_bucket = {}
    for t, d in zip(tokens, deltas):
        by_bucket.setdefault(base[t]["bucket"], []).append(d)

    return {
        "n": len(tokens),
        "only_base": sorted(set(base) - set(cand)),
        "only_cand": sorted(set(cand) - set(base)),
        "base_mean": statistics.mean(base[t]["score"] for t in tokens),
        "cand_mean": statistics.mean(cand[t]["score"] for t in tokens),
        "delta_mean": statistics.mean(deltas),
        "ci": bootstrap_ci(deltas, n_boot, seed),
        "wins": wins,
        "ties": len(deltas) - wins - losses,
        "losses": losses,
        "sign_p": sign_test_p(wins, losses),
        "top_token": tokens[top],
        "top_delta": deltas[top],
        "delta_without_top": statistics.mean(rest) if rest else float("nan"),
        "by_bucket": {
            b: (len(v), statistics.mean(v), sum(x > TIE_EPS for x in v), sum(x < -TIE_EPS for x in v))
            for b, v in sorted(by_bucket.items())
        },
        "movers": sorted(
            ((t, base[t]["bucket"], base[t]["score"], cand[t]["score"], d) for t, d in zip(tokens, deltas)),
            key=lambda r: -abs(r[4]),
        ),
    }


def render(r: dict, base_name: str, cand_name: str, split: str, top_k: int) -> str:
    lo, hi = r["ci"]
    verdict = (
        "CI excludes 0" if lo > 0 or hi < 0 else "CI includes 0 (not distinguishable from noise)"
    )
    out = [
        f"# Paired comparison ({split} split, N={r['n']})",
        "",
        f"- Baseline:  `{base_name}`",
        f"- Candidate: `{cand_name}`",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Baseline mean | {r['base_mean']:.4f} |",
        f"| Candidate mean | {r['cand_mean']:.4f} |",
        f"| Paired mean delta | {r['delta_mean']:+.4f} |",
        f"| Bootstrap 95% CI | [{lo:+.4f}, {hi:+.4f}] - {verdict} |",
        f"| Wins / ties / losses | {r['wins']} / {r['ties']} / {r['losses']} |",
        f"| Sign test p (two-sided) | {r['sign_p']:.3f} |",
        f"| Largest single mover | `{r['top_token']}` ({r['top_delta']:+.4f}) |",
        f"| Mean delta without it | {r['delta_without_top']:+.4f} |",
        "",
        "## Per bucket",
        "",
        "| Bucket | N | Mean delta | Wins | Losses |",
        "|---|---|---|---|---|",
    ]
    for b, (n, d, w, l) in r["by_bucket"].items():
        out.append(f"| {b} | {n} | {d:+.4f} | {w} | {l} |")

    movers = [m for m in r["movers"] if abs(m[4]) > TIE_EPS][:top_k]
    if movers:
        out += ["", "## Largest movers", "", "| Token | Bucket | Baseline | Candidate | Delta |", "|---|---|---|---|---|"]
        for t, b, s0, s1, d in movers:
            out.append(f"| `{t}` | {b} | {s0:.4f} | {s1:.4f} | {d:+.4f} |")

    if r["only_base"] or r["only_cand"]:
        out += [
            "",
            f"Tokens only in baseline: {len(r['only_base'])}; only in candidate: "
            f"{len(r['only_cand'])} (excluded from all numbers above).",
        ]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Paired per-scenario comparison of two runs")
    ap.add_argument("baseline", type=Path, help="Per-scenario CSV of the reference run")
    ap.add_argument("candidate", type=Path, help="Per-scenario CSV of the run being evaluated")
    ap.add_argument("--split", choices=["all", "tune", "holdout"], default="all")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=10, help="Number of largest movers to list")
    ap.add_argument("--md", type=Path, default=None, help="Also write the report to this file")
    args = ap.parse_args()

    base = load_scores(args.baseline)
    cand = load_scores(args.candidate)
    if args.split != "all":
        base = {t: v for t, v in base.items() if split_of(t) == args.split}
        cand = {t: v for t, v in cand.items() if split_of(t) == args.split}

    try:
        result = compare(base, cand, args.n_boot, args.seed)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    report = render(result, args.baseline.name, args.candidate.name, args.split, args.top)
    print(report)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(report)
        print(f"Report saved: {args.md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
summarize_eval.py
-----------------
Aggregate simulation results from nuPlan runs, broken down by scenario bucket.

Usage:
    python script/summarize_eval.py [--challenge CHALLENGE] [--run-dir DIR] [--v1-dir DIR]

Arguments:
    --challenge   Simulation challenge type (default: closed_loop_nonreactive_agents)
    --run-dir     Explicit path to a simulation output directory.
                  If omitted, finds the most recent run under datasets/nuplan/exp/.
    --v1-dir      Path to mini_v1 baseline directory for comparison
                  (default: benchmarks/mini_v1).
    --save        Save summary to outputs/eval_v2/ as CSV + MD.
    --tag         Label added to the saved file names, e.g. "baseline" or
                  "rerank_l0.3" -> eval_v2_<tag>_<timestamp>.csv

Examples:
    # Latest run, default challenge
    python script/summarize_eval.py

    # Compare against v1 baseline
    python script/summarize_eval.py --v1-dir benchmarks/mini_v1

    # Save results
    python script/summarize_eval.py --save
"""

import argparse
import csv
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Bucket mapping: scenario_type → bucket name ─────────────────────────────
# Maps nuPlan scenario_type strings to our mini_eval_v2 coverage buckets.
# The parquet 'scenario_type' column is used directly for bucketing.
SCENARIO_TYPE_TO_BUCKET = {
    # intersections — matches mini_eval_v2.yaml bucket: intersections
    "traversing_intersection":                  "intersections",
    "traversing_traffic_light_intersection":    "intersections",
    "starting_unprotected_cross_turn":          "intersections",
    "on_traffic_light_intersection":            "intersections",
    "starting_left_turn":                       "intersections",
    "starting_right_turn":                      "intersections",
    "on_all_way_stop_intersection":             "intersections",
    # lane_change — matches mini_eval_v2.yaml bucket: lane_change
    "changing_lane":            "lane_change",
    "changing_lane_to_left":    "lane_change",
    "changing_lane_to_right":   "lane_change",
    "ego_lane_change":          "lane_change",
    # lead_vehicle — matches mini_eval_v2.yaml bucket: lead_vehicle
    "following_lane_with_lead":       "lead_vehicle",
    "following_lane_with_slow_lead":  "lead_vehicle",
    "stopping_with_lead":             "lead_vehicle",
    # pedestrian — matches mini_eval_v2.yaml bucket: pedestrian
    "near_pedestrian_on_crosswalk":             "pedestrian",
    "near_pedestrian_on_crosswalk_with_ego":    "pedestrian",
    "stopping_at_crosswalk":                    "pedestrian",
    "traversing_crosswalk":                     "pedestrian",
    "waiting_for_pedestrian_to_cross":          "pedestrian",
    # stop_and_go — matches mini_eval_v2.yaml bucket: stop_and_go
    "stationary":                                   "stop_and_go",
    "stationary_at_traffic_light_with_lead":        "stop_and_go",
    "stationary_at_traffic_light_without_lead":     "stop_and_go",
    "stationary_in_traffic":                        "stop_and_go",
    "accelerating_at_traffic_light":                "stop_and_go",
    "stopping_at_stop_sign_without_lead":           "stop_and_go",
    # high_medium_speed — matches mini_eval_v2.yaml bucket: high_medium_speed
    "high_magnitude_speed":     "high_medium_speed",
    "medium_magnitude_speed":   "high_medium_speed",
    "low_magnitude_speed":      "high_medium_speed",
    # near_long_vehicle — matches mini_eval_v2.yaml bucket: near_long_vehicle
    "near_long_vehicle":        "near_long_vehicle",
    "near_multiple_vehicles":   "near_long_vehicle",
    "near_high_speed_vehicle":  "near_long_vehicle",
    # edge_cases — matches mini_eval_v2.yaml bucket: edge_cases
    "high_lateral_acceleration":    "edge_cases",
    "near_trafficcone_on_driveable": "edge_cases",
    "near_construction_zone_sign":  "edge_cases",
    "near_multiple_pedestrians":    "edge_cases",
    "traversing_pickup_dropoff":    "edge_cases",
    "on_pickup_dropoff":            "edge_cases",
}

# ── Legacy token-based bucket map (kept for reference / future use) ──────────
# These are the scenario_tag.token values from mini_eval_v2.yaml build time.
# The parquet uses simulation output tokens, so SCENARIO_TYPE_TO_BUCKET is used.
BUCKET_MAP = {
    # intersections
    "dafbae80126553c0": "intersections", "876f639b4d6b56d7": "intersections",
    "40b1b4111e825738": "intersections", "79a7aa1ea826527a": "intersections",
    "ff3158605b005f8a": "intersections", "defe34b1b1c25d8b": "intersections",
    "3e72ce75ba8a547e": "intersections", "a3b4b8e47696534e": "intersections",
    "e4d09605e2b6563e": "intersections", "806653c428055cae": "intersections",
    "02a9c48371f359e1": "intersections", "a484ac489ac3568a": "intersections",
    "748004463b3751d2": "intersections", "d4bda6cefeb559fa": "intersections",
    # lane_change
    "d788bef2e5a35a40": "lane_change",   "9b66f41e37515c7e": "lane_change",
    "81977df61cba5954": "lane_change",   "c11d38ebe41151d8": "lane_change",
    "6141acb01ce15d57": "lane_change",   "2c7f92e91f9b59ad": "lane_change",
    "963481dd224b5cc7": "lane_change",
    # lead_vehicle
    "cd5120b17ec65044": "lead_vehicle",  "90962439c5e457c4": "lead_vehicle",
    "4608ec6a4a885b11": "lead_vehicle",  "23e4777b457c5b2e": "lead_vehicle",
    "bfb3087c5acc5b42": "lead_vehicle",  "69fb195d3e925f6f": "lead_vehicle",
    "c14271a9d7055eef": "lead_vehicle",  "addefbd481c55e24": "lead_vehicle",
    "45b00c28db7655a8": "lead_vehicle",  "012bd0fe22fe5c43": "lead_vehicle",
    # pedestrian
    "19e3ccd6f88e5f2d": "pedestrian",    "e1081d3d91325a8b": "pedestrian",
    "f3c40bb00f995f1f": "pedestrian",    "9128019da5645c2d": "pedestrian",
    "85fd87a7b2675bd5": "pedestrian",    "57cdaea36e5e53db": "pedestrian",
    "61dee24849dd52be": "pedestrian",    "62b7575a98845999": "pedestrian",
    "6ae71824b3695919": "pedestrian",    "a947b48d32fb586e": "pedestrian",
    "5315f7c68a0a573c": "pedestrian",
    # stop_and_go
    "6f37482faaa253f5": "stop_and_go",   "049d2977cf375747": "stop_and_go",
    "d04a2b028893526b": "stop_and_go",   "7078aaddf7cf5a9d": "stop_and_go",
    "2df83b76f0bc57f1": "stop_and_go",   "22664cd54c3559d9": "stop_and_go",
    "4e4feb7b27275423": "stop_and_go",   "6dcbe6aa9cb958ed": "stop_and_go",
    "06acbd06452a5247": "stop_and_go",   "957e84d53b915519": "stop_and_go",
    "8cd9fd808301594c": "stop_and_go",   "6a2f8c80ec835585": "stop_and_go",
    # high_medium_speed
    "54b7fcdba4cb58e4": "high_medium_speed", "4809dcaca8eb5b7f": "high_medium_speed",
    "0b516f5e5721573a": "high_medium_speed", "322ab7c0c5705175": "high_medium_speed",
    "fa19af6a17ea5710": "high_medium_speed", "1ae336d305ac5bc0": "high_medium_speed",
    "29992920c5155625": "high_medium_speed", "2bbe09fee03a5fed": "high_medium_speed",
    "b422c67738115bd7": "high_medium_speed", "10ce2cfd0ea65bd2": "high_medium_speed",
    "9664db5423505d01": "high_medium_speed", "d203fe1453fc5084": "high_medium_speed",
    # near_long_vehicle
    "19c81bdb96cf5b98": "near_long_vehicle", "70c126cc4cc45784": "near_long_vehicle",
    "f2e57048bead5a94": "near_long_vehicle", "637e54086e1a5595": "near_long_vehicle",
    "05c996ef58b45e78": "near_long_vehicle", "e993edfdb97456ea": "near_long_vehicle",
    "5ab340df26095e7a": "near_long_vehicle", "5cdea93101965af9": "near_long_vehicle",
    "fed2cb4d633e5532": "near_long_vehicle", "c6f3cedaa6aa5580": "near_long_vehicle",
    # edge_cases
    "c92c4811752c560f": "edge_cases",    "e2997b0f6f395853": "edge_cases",
    "34c824411daa597d": "edge_cases",    "911806075c2758f5": "edge_cases",
    "ea0efbdc8f6b50bd": "edge_cases",    "f79024c835ac58be": "edge_cases",
    "37065a044922533d": "edge_cases",    "612f44dab2fc593e": "edge_cases",
}

BUCKET_ORDER = [
    "intersections", "lane_change", "lead_vehicle", "pedestrian",
    "stop_and_go", "high_medium_speed", "near_long_vehicle", "edge_cases",
]

V1_TOKENS = {
    "38b71d6e2dc65a1d", "0d48c024a6455ace", "1dbcee5b8ba55210",
    "c037d6199bb25375", "a3b4b8e47696534e", "b391c0fa8349515b",
    "10460185498255a9", "00699cdbd1a051bd", "e4d09605e2b6563e",
    "3cb701313ff0518e",
}


def _find_latest_run(challenge: str, base: str) -> Path:
    # Recursive: experiment_uid may be nested (e.g. mini_eval_v2/planTF/<tag>).
    pattern = os.path.join(
        base, "nuplan/exp/exp/simulation", challenge, "**/aggregator_metric/*.parquet"
    )
    files = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(
            f"No aggregator parquet found for challenge '{challenge}' under {base}\n"
            f"  pattern: {pattern}"
        )
    return Path(files[-1])


def _load_parquet(path: Path):
    try:
        import pandas as pd
        return pd.read_parquet(path)
    except ImportError:
        print("ERROR: pandas not installed. Run: pip install pandas pyarrow", file=sys.stderr)
        sys.exit(1)


def _extract_token(scenario_name: str) -> str:
    """nuPlan scenario names include the token as a hex suffix."""
    parts = scenario_name.split("_")
    for part in reversed(parts):
        if len(part) == 16 and all(c in "0123456789abcdef" for c in part.lower()):
            return part.lower()
    return ""


def main():
    ap = argparse.ArgumentParser(description="Summarize eval results by bucket")
    ap.add_argument("--challenge", default="closed_loop_nonreactive_agents")
    ap.add_argument("--run-dir", default=None, help="Path to aggregator_metric/")
    ap.add_argument("--v1-dir", default="benchmarks/mini_v1", help="mini_v1 baseline dir")
    ap.add_argument("--save", action="store_true", help="Save results to outputs/eval_v2/")
    ap.add_argument("--tag", default=None, help="Label for saved file names")
    ap.add_argument("--datasets-root", default="datasets", help="Root of datasets dir")
    args = ap.parse_args()

    # ── Load latest run ──────────────────────────────────────────────────────
    if args.run_dir:
        parquet_files = list(Path(args.run_dir).glob("*.parquet"))
        if not parquet_files:
            print(f"ERROR: No parquet files in {args.run_dir}")
            sys.exit(1)
        parquet_path = max(parquet_files, key=os.path.getmtime)
    else:
        parquet_path = _find_latest_run(args.challenge, args.datasets_root)

    print(f"\nLoading: {parquet_path}")
    df = _load_parquet(parquet_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)[:8]}...")

    # ── Extract score per scenario ───────────────────────────────────────────
    score_col = None
    for candidate in ["score", "final_score", "weighted_average_score"]:
        if candidate in df.columns:
            score_col = candidate
            break

    scenario_col = next((c for c in df.columns if "scenario" in c.lower()), None)

    if score_col is None:
        print("\nAvailable columns:", list(df.columns))
        print("WARNING: Could not find score column. Showing raw data summary.")
        print(df.head(10).to_string())
        return

    rows = []
    import re
    for _, row in df.iterrows():
        scen = str(row.get(scenario_col, ""))
        scen_type = str(row.get("scenario_type", ""))
        # Only include per-scenario rows: scenario column is a 16-char hex token
        if not re.match(r'^[0-9a-f]{16}$', scen.lower()):
            continue
        tok = scen.lower()
        # Primary: use scenario_type for bucket assignment (reliable)
        bucket = SCENARIO_TYPE_TO_BUCKET.get(scen_type)
        if bucket is None:
            # Fallback: try legacy token-based map
            bucket = BUCKET_MAP.get(tok, "unknown")
        score = float(row[score_col]) if row[score_col] is not None else float("nan")
        rows.append({
            "token": tok, "scenario": scen, "scenario_type": scen_type,
            "bucket": bucket, "score": score,
        })

    # ── Overall stats ────────────────────────────────────────────────────────
    import statistics
    all_scores = [r["score"] for r in rows if not (r["score"] != r["score"])]
    overall = statistics.mean(all_scores) if all_scores else 0.0
    failures = sum(1 for s in all_scores if s < 0.5)

    print(f"\n{'='*62}")
    print(f"  EVAL RESULTS — {args.challenge}")
    print(f"{'='*62}")
    print(f"  Overall mean score : {overall:.4f}")
    print(f"  Scenarios  scored  : {len(all_scores)}")
    print(f"  Failures (<0.5)    : {failures}")
    print()

    # ── Per-bucket breakdown ─────────────────────────────────────────────────
    by_bucket = {}
    for r in rows:
        by_bucket.setdefault(r["bucket"], []).append(r["score"])

    print(f"  {'Bucket':<22} {'N':>4}  {'Mean':>7}  {'Min':>7}  {'Fails':>6}")
    print(f"  {'-'*22}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*6}")

    bucket_stats = {}
    for bucket in BUCKET_ORDER + sorted(set(by_bucket.keys()) - set(BUCKET_ORDER)):
        scores = by_bucket.get(bucket, [])
        if not scores:
            continue
        mean_s = statistics.mean(scores)
        min_s = min(scores)
        fails = sum(1 for s in scores if s < 0.5)
        bucket_stats[bucket] = {"mean": mean_s, "min": min_s, "n": len(scores), "fails": fails}
        flag = " (!)" if fails > 0 else ""
        print(f"  {bucket:<22} {len(scores):>4}  {mean_s:>7.4f}  {min_s:>7.4f}  {fails:>6}{flag}")

    # ── Lane-change coverage note ────────────────────────────────────────────
    print()
    print("  NOTE: lane_change bucket has only 7 tokens (all available in mini).")
    print("        Scores are not statistically representative. Use val14 for coverage.")

    # ── Zero-score scenarios ─────────────────────────────────────────────────
    zeros = [r for r in rows if r["score"] == 0.0]
    if zeros:
        print(f"\n  Zero-score scenarios ({len(zeros)}):")
        for r in zeros:
            v1 = " [also in v1]" if r["token"] in V1_TOKENS else ""
            print(f"    {r['token']}  {r['bucket']:<22} {r['scenario_type'][:45]}{v1}")

    # ── V1 comparison ────────────────────────────────────────────────────────
    v1_scores = {r["token"]: r["score"] for r in rows if r["token"] in V1_TOKENS}
    if v1_scores:
        print(f"\n  V1-overlap tokens ({len(v1_scores)}/10):")
        for tok, score in sorted(v1_scores.items(), key=lambda x: x[1]):
            print(f"    {tok}  score={score:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.tag:
            ts = f"{args.tag}_{ts}"
        out_dir = Path("outputs/eval_v2")
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / f"eval_v2_{ts}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["token", "scenario", "scenario_type", "bucket", "score"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  CSV saved: {csv_path}")

        md_path = out_dir / f"eval_v2_{ts}.md"
        with open(md_path, "w") as f:
            f.write(f"# mini_eval_v2 Results — {ts}\n\n")
            f.write(f"**Challenge:** {args.challenge}  \n")
            f.write(f"**Source:** `{parquet_path}`  \n")
            f.write(f"**Overall mean score:** {overall:.4f}  \n")
            f.write(f"**Scenarios:** {len(all_scores)}  \n")
            f.write(f"**Failures (<0.5):** {failures}  \n\n")
            f.write("## Per-Bucket Summary\n\n")
            f.write("| Bucket | N | Mean | Min | Fails |\n")
            f.write("|---|---|---|---|---|\n")
            for bucket, s in bucket_stats.items():
                f.write(f"| {bucket} | {s['n']} | {s['mean']:.4f} | {s['min']:.4f} | {s['fails']} |\n")
        print(f"  MD  saved: {md_path}")

    print()


if __name__ == "__main__":
    main()

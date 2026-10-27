#!/usr/bin/env python3
"""
build_scenario_inventory.py
---------------------------
Scans all nuplan-v1.1_mini SQLite log databases and produces:
  outputs/mini_scenario_inventory.csv  — full type × log count matrix
  outputs/mini_scenario_inventory.md   — human-readable summary
  outputs/selected_tokens.csv          — coverage-driven token selection for mini_eval_v2

Usage:
  conda run -n plantf python script/build_scenario_inventory.py
  conda run -n plantf python script/build_scenario_inventory.py --mini-dir /path/to/mini
  conda run -n plantf python script/build_scenario_inventory.py --seed 42 --target 10

  # Validate that every token in a scenario_filter yaml will resolve at sim time:
  conda run -n plantf python script/build_scenario_inventory.py --validate config/scenario_filter/mini_eval_v2.yaml

Token-matching rules (mirrors nuplan devkit get_scenarios_from_db):
  - Tokens must be lidar_pc.token values (NOT scenario_tag.token PKs).
  - The lidar_pc frame must belong to a "valid" scene: at least 2 scenes before
    and 2 after it in the log (devkit valid_scenes CTE).
  - Tokens that fail either check are silently dropped by the scenario builder.

This script is the canonical source for the data in docs/mini_scenario_inventory.md
and the tokens in config/scenario_filter/mini_eval_v2.yaml.
Re-run if the mini dataset changes.
"""

import argparse
import csv
import os
import random
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


# ── Devkit-compatible SQL ──────────────────────────────────────────────────────
# Must stay in sync with nuplan-devkit get_scenarios_from_db().
# Mirrors the valid_scenes CTE exactly: excludes scenes in the first 2 or last 2
# positions of the log so the scenario has enough context frames on both sides.
_VALID_SCENARIOS_SQL = """
    WITH ordered_scenes AS (
        SELECT  token,
                ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num
        FROM scene
    ),
    num_scenes AS (
        SELECT COUNT(*) AS cnt FROM scene
    ),
    valid_scenes AS (
        SELECT o.token
        FROM ordered_scenes AS o CROSS JOIN num_scenes AS n
        WHERE o.row_num >= 3 AND o.row_num < n.cnt - 1
    )
    SELECT  lower(hex(lp.token)) AS token,
            st.type             AS scenario_type
    FROM lidar_pc AS lp
    INNER JOIN scenario_tag  AS st ON lp.token = st.lidar_pc_token
    INNER JOIN lidar         AS ld ON ld.token = lp.lidar_token
    INNER JOIN log           AS l  ON ld.log_token = l.token
    INNER JOIN valid_scenes  AS vs ON lp.scene_token = vs.token
    {where_clause}
"""


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MINI_DIR = (
    Path(__file__).parent.parent.parent
    / "datasets/nuplan/nuplan-v1.1_mini/data/cache/mini"
)
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Bucket → list of scenario type substrings to match
BUCKET_DEFS = {
    "intersections":      ["traversing_intersection", "on_intersection", "tl_intersection",
                           "near_intersection"],
    "lane_change":        ["changing_lane", "lane_change"],
    "lead_vehicle":       ["following_lane_with_lead", "starting_", "stopping_",
                           "behind_long_vehicle"],
    "pedestrian":         ["near_pedestrian", "waiting_for_pedestrian"],
    "stop_and_go":        ["high_magnitude_jerk", "low_magnitude_speed",
                           "stationary_in_traffic", "on_stopline"],
    "high_medium_speed":  ["high_magnitude_speed", "medium_magnitude_speed"],
    "near_long_vehicle":  ["near_long_vehicle"],
    "edge_cases":         ["on_pickup_dropoff", "traversing_narrow_lane",
                           "ego_at_pudo", "near_trafficcone"],
}


def assign_bucket(scenario_type: str) -> str:
    for bucket, patterns in BUCKET_DEFS.items():
        for p in patterns:
            if p in scenario_type:
                return bucket
    return "other"


def scan_databases(mini_dir: Path, verbose: bool = False):
    """
    Return {scenario_type: {log_name: [token, ...]}}.

    Only tokens that satisfy the nuPlan devkit's valid_scenes constraint are
    included — i.e. frames whose scene has at least 2 scenes before and after
    it in the log.  This matches the SQL in get_scenarios_from_db() so tokens
    selected here are guaranteed to resolve at simulation time.
    """
    db_files = sorted(mini_dir.glob("*.db"))
    if not db_files:
        raise FileNotFoundError(f"No .db files found in {mini_dir}")
    print(f"Found {len(db_files)} DB files in {mini_dir}")

    data: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    total = 0

    query = _VALID_SCENARIOS_SQL.format(where_clause="")
    for i, db_path in enumerate(db_files):
        log_name = db_path.stem
        if verbose:
            print(f"  [{i+1}/{len(db_files)}] {log_name}")
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute(query)
            for token, stype in cur.fetchall():
                data[stype][log_name].append(token)
                total += 1
            conn.close()
        except Exception as e:
            print(f"  WARNING: skipped {log_name}: {e}")

    print(f"Total valid scenario_tag entries: {total:,}")
    return data, db_files


def validate_yaml(yaml_path: Path, mini_dir: Path) -> int:
    """
    Validate that every scenario_tokens entry in a scenario_filter yaml will
    resolve through the nuPlan scenario builder.  Uses the exact same SQL as
    the devkit (valid_scenes CTE + lidar_pc.token match).

    Exits with code 0 if all tokens resolve, 1 otherwise.
    """
    if not yaml_path.exists():
        print(f"ERROR: yaml not found: {yaml_path}")
        return 1

    text = yaml_path.read_text()
    # Extract quoted 16-char hex strings under scenario_tokens:
    tokens = re.findall(r'^\s*-\s*"([0-9a-f]{16})"', text, re.MULTILINE)
    if not tokens:
        print(f"No scenario_tokens found in {yaml_path}")
        return 0

    print(f"Validating {len(tokens)} tokens from {yaml_path.name} ...")

    db_files = sorted(mini_dir.glob("*.db"))
    if not db_files:
        print(f"ERROR: No .db files in {mini_dir}")
        return 1

    placeholders = ",".join(["?"] * len(tokens))
    where = f"WHERE lower(hex(lp.token)) IN ({placeholders})"
    query = _VALID_SCENARIOS_SQL.format(where_clause=where)
    args = [t for t in tokens]   # plain hex strings; SQL uses lower(hex(lp.token))

    found: dict[str, str] = {}   # token → scenario_type
    for db_path in db_files:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(query, args)
        for tok, stype in cur.fetchall():
            if tok not in found:
                found[tok] = stype
        conn.close()

    missing = [t for t in tokens if t not in found]
    resolved = len(tokens) - len(missing)

    if missing:
        print(f"FAIL: {resolved}/{len(tokens)} tokens resolve.")
        print(f"  {len(missing)} token(s) will be silently dropped by the scenario builder:")
        for t in missing:
            print(f"    {t}")
        print()
        print("Run with --mini-dir to regenerate replacements, or check:")
        print("  - Token is a lidar_pc.token value (not scenario_tag.token PK)")
        print("  - Token's scene is not in the first/last 2 scenes of its log")
        return 1
    else:
        print(f"OK: {resolved}/{len(tokens)} tokens resolve.")
        return 0


def select_tokens(data, seed: int, target_per_bucket: int):
    """
    For each bucket, pick `target_per_bucket` tokens spread across different logs.
    Strategy: round-robin over logs that have tokens for relevant types.
    Returns {bucket: [(token, log, type), ...]}
    """
    rng = random.Random(seed)

    # Flatten per bucket
    bucket_pool: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for stype, log_map in data.items():
        bucket = assign_bucket(stype)
        for log, tokens in log_map.items():
            for t in tokens:
                bucket_pool[bucket].append((t, log, stype))

    selected: dict[str, list[tuple[str, str, str]]] = {}
    for bucket, pool in bucket_pool.items():
        if bucket == "other":
            continue
        # Shuffle then spread across logs
        rng.shuffle(pool)
        seen_logs: set[str] = set()
        picked = []
        # First pass: one per log
        for t, log, stype in pool:
            if log not in seen_logs and len(picked) < target_per_bucket:
                picked.append((t, log, stype))
                seen_logs.add(log)
        # Second pass: fill remainder if < target
        if len(picked) < target_per_bucket:
            for t, log, stype in pool:
                if (t, log, stype) not in picked:
                    picked.append((t, log, stype))
                    if len(picked) >= target_per_bucket:
                        break
        selected[bucket] = picked

    return selected


def write_csv(data, db_files, out_path: Path):
    """Write full type × log count matrix."""
    logs = [p.stem for p in db_files]
    all_types = sorted(data.keys())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_type", "total"] + logs)
        for stype in all_types:
            row = [stype]
            total = sum(len(v) for v in data[stype].values())
            row.append(total)
            for log in logs:
                row.append(len(data[stype].get(log, [])))
            writer.writerow(row)
    print(f"Wrote: {out_path}")


def write_markdown(data, out_path: Path):
    """Write human-readable markdown summary with bucket assignments."""
    rows = []
    for stype, log_map in data.items():
        total = sum(len(v) for v in log_map.values())
        logs = len(log_map)
        bucket = assign_bucket(stype)
        rows.append((total, logs, stype, bucket))
    rows.sort(reverse=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Mini Dataset Scenario Inventory\n\n")
        f.write("*Generated by `script/build_scenario_inventory.py`*\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Scenario types: {len(rows)}\n")
        f.write(f"- Total tags: {sum(r[0] for r in rows):,}\n\n")
        f.write("## Type Distribution\n\n")
        f.write("| Rank | Scenario Type | Total Tokens | Logs | Bucket |\n")
        f.write("|------|---------------|-------------|------|--------|\n")
        for rank, (total, logs, stype, bucket) in enumerate(rows, 1):
            f.write(f"| {rank} | `{stype}` | {total:,} | {logs} | {bucket} |\n")
        f.write("\n## Bucket Coverage\n\n")
        bucket_totals: dict[str, int] = defaultdict(int)
        for total, logs, stype, bucket in rows:
            bucket_totals[bucket] += total
        for bucket, total in sorted(bucket_totals.items(), key=lambda x: -x[1]):
            f.write(f"- **{bucket}**: {total:,} total tokens\n")
    print(f"Wrote: {out_path}")


def write_tokens_csv(selected, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bucket", "token", "log", "scenario_type"])
        for bucket, picks in selected.items():
            for t, log, stype in picks:
                writer.writerow([bucket, t, log, stype])
    print(f"Wrote: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan mini dataset and build scenario inventory, or validate a yaml."
    )
    parser.add_argument("--mini-dir", type=Path, default=DEFAULT_MINI_DIR,
                        help="Path to nuplan-v1.1_mini/data/cache/mini/")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for token selection")
    parser.add_argument("--target", type=int, default=10,
                        help="Target tokens per bucket in selection")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-DB progress")
    parser.add_argument(
        "--validate", type=Path, metavar="YAML",
        help="Validate all scenario_tokens in YAML resolve at sim time; exit 0=pass, 1=fail"
    )
    args = parser.parse_args()

    if not args.mini_dir.exists():
        print(f"ERROR: mini dir not found: {args.mini_dir}")
        print("  Set --mini-dir or check datasets symlink")
        return 1

    if args.validate:
        return validate_yaml(args.validate, args.mini_dir)

    data, db_files = scan_databases(args.mini_dir, verbose=args.verbose)

    write_csv(data, db_files, OUTPUT_DIR / "mini_scenario_inventory.csv")
    write_markdown(data, OUTPUT_DIR / "mini_scenario_inventory.md")

    selected = select_tokens(data, seed=args.seed, target_per_bucket=args.target)
    write_tokens_csv(selected, OUTPUT_DIR / "selected_tokens.csv")

    print("\nDone. To regenerate mini_eval_v2.yaml from selected_tokens.csv,")
    print("review outputs/selected_tokens.csv and update config/scenario_filter/mini_eval_v2.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

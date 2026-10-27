# Evaluation Plan

## Purpose

This document defines the two-tier benchmark strategy for planTF evaluation on
the nuplan-v1.1_mini dataset, explains the reasoning behind scenario selection,
clarifies what claims are valid after each benchmark, and documents the hard
limits of mini-split evaluation.

---

## Benchmark Tiers

### Tier 1 — mini_benchmark_v1 (Fast Dev Benchmark)

| Property | Value |
|---|---|
| Config | `config/scenario_filter/mini_benchmark.yaml` |
| Script | `bash script/mini_benchmark.sh` |
| Scenarios | 10 (frozen, token-pinned) |
| Run time | ~3–5 minutes |
| Purpose | Fast regression check; repeatable across commits |
| Baseline | `benchmarks/mini_v1/baseline.md` (final score = 0.7716) |

**Use for**: detecting regressions between commits. A green v1 run means nothing
regressed on these 10 specific scenarios. It does NOT imply broad correctness.

**Do NOT** use v1 results to claim general planner robustness. The 10-scenario
sample includes 2 zero-score hard cases that disproportionately move the aggregate.

### Tier 2 — mini_eval_v2 (Coverage-Driven Validation Benchmark)

| Property | Value |
|---|---|
| Config | `config/scenario_filter/mini_eval_v2.yaml` |
| Script | `bash script/mini_eval_v2.sh` |
| Scenarios | 84 (coverage-bucketed, token-pinned) |
| Run time | ~15–20 minutes |
| Purpose | Stronger validation across 8 scenario buckets |
| Summarize | `python script/summarize_eval.py --save` |

**Use for**: claiming per-category planner behaviour, identifying systematic
weaknesses, comparing changes that are expected to affect specific scenario types.

---

## Scenario Bucket Design Philosophy

### Why token-pinned instead of random sampling?

Random sampling from the mini split biases toward the most common scenario types
(`stationary`, `on_intersection` — together >30% of all tags). Rare but important
scenario types (lane changes, unprotected turns, pedestrian yielding) would be
chronically under-represented or absent.

Token-pinned benchmarks are **reproducible** and **interpretable**: the same 84
scenarios run every time, making result comparisons exact and transparent.

### Coverage requirements

Each bucket targets:
- Minimum 7 scenarios (hard limit imposed by lane_change availability)
- Maximum 14 scenarios (intersections, due to known failure diversity)
- Spread across different log files (geographic/temporal diversity within mini)
- Mix of easy, medium, and known-hard scenarios within each bucket

### Bucket rationale

| Bucket | Why included | Known model behaviour |
|---|---|---|
| `intersections` | Highest-risk manoeuvre class; 2 v1 zero-scorers | Struggles with traversing_tl_intersection |
| `lane_change` | Important for highway/urban safety | **Mini data insufficient** (see below) |
| `lead_vehicle` | Core urban following behaviour | Generally handles well (v1 scores ~0.99) |
| `pedestrian` | Safety-critical; yielding correctness | v1 shows 0.95 on near_ped; waiting_for_ped unknown |
| `stop_and_go` | Traffic light compliance | v1 shows 0.94–0.94; stationary_in_traffic unknown |
| `high_medium_speed` | Comfort + compliance at speed | v1 shows 0.96–0.99 |
| `near_long_vehicle` | Occlusion / spatial awareness | v1 shows 0.94 |
| `edge_cases` | Model stress-test (construction, dense peds, lateral accel) | Unknown — not in v1 |

---

## Limits of Mini-Split Evaluation

### Lane change coverage — explicitly insufficient

The mini split contains **44 total lane-change tagged frames** across all 64 logs:
- `changing_lane`: 22
- `changing_lane_to_left`: 15
- `changing_lane_to_right`: 7

All 7 distinct tokens used in `mini_eval_v2` represent the **entire available pool**.
This is a sample-size issue with the dataset, not with the benchmark design.

**Claim validity**: no statistical claim about lane-change performance is valid
from mini results. All 7 tokens must pass; any failure warrants investigation;
but a pass rate of 7/7 does not imply generalization.

### Geographic coverage

The mini split primarily covers Las Vegas Strip (US-NV), with smaller coverage
of Pittsburgh and Boston. Model behaviour on Singapore (SG) and other geographies
in full nuPlan requires the full dataset split.

### Scenario density

The mini split concentrates on structured urban driving. Ramp merges, motorway
following, and true highway scenarios are largely absent. Speed claims made from
`high_magnitude_speed` tokens in mini may not transfer to motorway contexts.

### Reactive agents

`closed_loop_nonreactive_agents` (the default challenge) does not test agent
reactivity. The planner navigates among agents that follow fixed logs. A passing
score here is necessary but not sufficient for real-world deployment.
`closed_loop_reactive_agents` reveals additional failure modes when agents respond
to the ego vehicle. Run both challenges for a complete picture.

---

## What Is Already Proven (after v1 + deployment work)

- **Patched model fidelity** — 5 ONNX compatibility patches are numerically equivalent to the original on all non-NATTEN operations (max Δ < 1e-5 on CPU).
- **NATTEN approximation cost**:
  - Worst-case trajectory divergence: 0.86 m max, mean 0.35 m/step
  - TRT FP32 path (Patched PyTorch vs TRT FP32): 1.6 mm — TRT matches the patched model (the patched model itself is 0.35 m from the original checkpoint)
- **TensorRT latency** (RTX 3060, batch=1):
  - FP16: **1.2 ms** (16.2× over CPU)
  - FP32: **1.7 ms**, max Δ = 1.6 mm vs ORT — verified fidelity
- **mini_benchmark_v1** (10 scenarios): final NR-CLS = **0.7716**
- **mini_eval_v2 baseline** (84 scenarios, NR-CLS): **0.8401** — intersections bucket weakest at 0.7055; TL compliance identified as primary weakness
- **Mode re-ranker** (λ=0.3, `src/planners/mode_reranker.py`): **0.8478** (+0.77%); main gain `stop_and_go` +0.036; regression `pedestrian` −0.009 under investigation. Driven by one recovered scenario; not yet confirmed on held-out tokens (see `script/compare_runs.py`)
- **C++ runtime** (RTX 3060): ORT matches Python ORT < 0.001 mm; TRT FP16 0.770 ms (1298 QPS), FP32 1.344 ms (744 QPS)

## What Is Still Pending

- **Lane-change robustness** — mini data insufficient (7 tokens); val14 required for statistical validity.
- **Reactive agent behaviour** — `closed_loop_reactive_agents` not yet benchmarked with v2.
- **NATTEN faithful path** — original model cannot be exported without NATTEN patch; FP16 LayerNorm drift (59 mm) unresolved pending opset 17 re-export.
- **Traffic-light compliance** — intersections bucket 0.7055; re-ranking leaves it unchanged because all 7 failures produce no geometrically-correct mode in K=6.
- **Pedestrian bucket regression** — re-ranking causes −0.009 drop; root cause: coverage heuristic penalises lateral pedestrian-avoidance moves that deviate from the lane centreline. Fix: skip re-ranking when the nearest on-route polygon is type CROSSWALK.
- **Geography generalisation** — full val14/test14 not yet run.

---

## Suggested Evaluation Cadence

| When | What to run | Why |
|---|---|---|
| Every commit | `script/mini_benchmark.sh` (v1, 10 scenarios) | Fast regression catch |
| Before a merge to main | `script/mini_eval_v2.sh` (v2, 84 scenarios) | Broader validation |
| Before a deployment claim | v2 on both NR and Reactive challenges | Reactivity check |
| For a paper/publication claim | Full val14 + test14 on full dataset | Statistical validity |

---

## Output Path Conventions

```
outputs/
  eval_v2/
    eval_v2_YYYYMMDD_HHMMSS.csv    ← per-scenario results
    eval_v2_YYYYMMDD_HHMMSS.md     ← summary by bucket
  tensorrt/
    trt_benchmark_*.json
    trt_latency_*.csv
benchmarks/
  mini_v1/
    baseline.md                    ← frozen v1 reference (do not overwrite)
    latency_baseline.csv           ← per-step latency from v1 run
```

Do not overwrite `benchmarks/mini_v1/baseline.md`. New run results go to
`outputs/eval_v2/`. The v1 baseline is the permanent frozen reference.

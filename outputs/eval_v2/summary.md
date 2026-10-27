# mini_eval_v2 Benchmark Results

**Date:** 2026-03-27 (supersedes 2026-03-24 summary; see archive note below)
**Challenge:** closed\_loop\_nonreactive\_agents  
**Planner:** PlanTF (state6+SDE) — `checkpoints/planTF.ckpt`  
**Scenario filter:** `config/scenario_filter/mini_eval_v2.yaml`  
**Hardware:** ASUS ROG Zephyrus G14 · RTX 3060 · Ryzen 9 5900HS  
**Simulation time:** ~20 minutes (84 scenarios, sequential worker)

---

## Run Index

| Run dir (under `.../closed_loop_nonreactive_agents/`) | Scenarios | Config | Overall | Output file |
|---|---|---|---|---|
| `2026.03.25.16.06.57` | 84 | baseline (argmax) | **0.8401** | `eval_v2_20260327_012254.md` (not committed) |
| `2026.03.26.16.39.15` | 84 | re-ranked (λ=0.3) | **0.8478** | `eval_v2_20260327_012317.md` (not committed) |

> **Missing files.** The per-scenario CSV/MD files for the two authoritative
> runs above were never committed; only the stale 2026-03-24 run is in this
> directory. Until they are added (`eval_v2_20260327_012254.{csv,md}` and
> `eval_v2_20260327_012317.{csv,md}`), the 0.8401 / 0.8478 numbers cannot be
> checked from the repo, and `script/compare_runs.py` cannot be run on them.

Stale runs: `2026.03.24` (`mini_eval_v2/planTF/v2_run1/`): 71/84 scenarios ran; score 0.8166. `eval_v2_20260327_011722` is a March-27 re-run of that same stale directory — also 71 scenarios, 0.8166. Both superseded.

---

## Baseline Run — 2026.03.25.16.06.57 (84 scenarios)

| Metric | Value |
|--------|-------|
| **Overall mean score** | **0.8401** |
| Scenarios evaluated | 84 |
| Failures (score = 0.000) | 10 |

### Per-Bucket Summary

| Bucket | N | Mean | Fails |
|--------|---|------|-------|
| intersections | 24 | 0.7055 | 7 |
| lane\_change | 5 | 0.7079 | 1 |
| lead\_vehicle | 7 | 0.9918 | 0 |
| pedestrian | 7 | 0.9666 | 0 |
| stop\_and\_go | 19 | 0.9127 | 1 |
| high\_medium\_speed | 9 | 0.9680 | 0 |
| near\_long\_vehicle | 5 | 0.9031 | 0 |
| edge\_cases | 8 | 0.7278 | 2 |

Zero-score tokens: `0136ce79864c53e1`, `4a275c4d8e125040`, `48a1b05d9c6f5fd7`, `d9ee9cf40a84520f`, `9e30155b8bb55fd9`, `a3b4b8e47696534e`, `eaf0200162f557cd`, `e4d09605e2b6563e`, `cc1d648ab526530a`, `edb1d6776eb25f5d`

---

## Re-ranked Run — 2026.03.26.16.39.15 (84 scenarios, λ=0.3)

| Metric | Value |
|--------|-------|
| **Overall mean score** | **0.8478** (+0.77%) |
| Scenarios evaluated | 84 |
| Failures (score = 0.000) | 9 |

### Per-Bucket Summary

| Bucket | N | Baseline | Re-ranked | Delta |
|--------|---|----------|-----------|-------|
| intersections | 24 | 0.7055 | 0.7055 | 0.000 |
| lane\_change | 5 | 0.7079 | 0.7062 | −0.002 |
| lead\_vehicle | 7 | 0.9918 | 0.9917 | −0.001 |
| pedestrian | 7 | 0.9666 | 0.9578 | −0.009 |
| stop\_and\_go | 19 | 0.9127 | 0.9486 | **+0.036** |
| high\_medium\_speed | 9 | 0.9680 | 0.9686 | +0.001 |
| near\_long\_vehicle | 5 | 0.9031 | 0.9033 | +0.000 |
| edge\_cases | 8 | 0.7278 | 0.7311 | +0.003 |

Recovered token: `0136ce79864c53e1` (stop\_and\_go; was zero-score in baseline).

---

## Notes on Bucket N vs yaml Selection

The `Selected` column in `docs/mini_scenario_inventory.md` shows how many tokens were placed in each bucket in the yaml. Run N differs because `summarize_eval.py` buckets by `scenario_type` from the nuPlan aggregator parquet, which assigns the primary simulation tag — this can differ from the tag used to select the token. The total (84) is always consistent.

---

## Files

| File | Run | Scen | Score | Notes |
|------|-----|------|-------|-------|
| [eval_v2_20260324_173928.md](eval_v2_20260324_173928.md) | 2026-03-24 | 71 | 0.8166 | Stale — 13 tokens absent from mini snapshot |
| `eval_v2_20260327_011722.md` (not committed) | 2026-03-27 re-run | 71 | 0.8166 | Stale — same old dir, redundant |
| `eval_v2_20260327_012254.md` (not committed) | 2026-03-27 **baseline** | 84 | **0.8401** | **Authoritative baseline** |
| `eval_v2_20260327_012317.md` (not committed) | 2026-03-27 **re-ranked** | 84 | **0.8478** | **Authoritative re-ranked (λ=0.3)** |

---

## Reproducibility

```bash
# Baseline (re-ranker is off by default)
bash script/mini_eval_v2.sh closed_loop_nonreactive_agents
python script/summarize_eval.py --save --tag baseline

# Re-ranked
bash script/mini_eval_v2.sh closed_loop_nonreactive_agents \
    planner.imitation_planner.use_mode_reranker=true \
    planner.imitation_planner.reranker_lambda=0.3
python script/summarize_eval.py --save --tag rerank_l0.3

# Paired comparison (bootstrap CI, sign test); use --split holdout for the
# half of the tokens not used to choose lambda
python script/compare_runs.py outputs/eval_v2/eval_v2_baseline_*.csv \
    outputs/eval_v2/eval_v2_rerank_l0.3_*.csv
```

> `mini_eval_v2.sh` sets `worker=sequential` (Ray workers don't inherit
> PYTHONPATH) and `scenario_builder.data_root` from `$NUPLAN_DATA_ROOT`.

## Statistical caveat

The +0.0077 overall delta equals what a single scenario going from 0.0 to
~0.65 contributes (0.65 / 84), and exactly one zero-score scenario
(`0136ce79864c53e1`) was recovered. No other bucket moved by more than 0.01.
λ was picked on the 21 TL-intersection scenarios, which are a subset of these
84. Treat the gain as unconfirmed until `compare_runs.py --split holdout`
shows a confidence interval that excludes zero.


---

# Archive: stale 2026-03-24 run (71 / 84 scenarios)

Everything below describes the superseded `v2_run1` run, kept for reference.
It is the run whose CSV/MD are committed here
(`eval_v2_20260324_173928.*`). Do not compare it with the 84-scenario runs
above.

## Overall Results

| Metric | Value |
|--------|-------|
| **Overall mean score** | **0.8166** |
| Scenarios evaluated | 71 / 84 (13 tokens not in mini split) |
| Failures (score < 0.5) | 11 |
| Median score | 0.9724 |
| Min score | 0.0000 |

---

## Per-Bucket Summary

| Bucket | N | Mean | Min | Fails | Notes |
|--------|---|------|-----|-------|-------|
| **intersections** | 21 | 0.6742 | 0.0000 | 7 | (!) Low: TL intersection failures |
| **lane\_change** | 4 | 0.6527 | 0.0000 | 1 | Only 4 of 7 resolved in mini |
| **lead\_vehicle** | 5 | 0.9928 | 0.9806 | 0 | OK Strong |
| **pedestrian** | 6 | 0.9611 | 0.8249 | 0 | OK Strong |
| **stop\_and\_go** | 16 | 0.8964 | 0.0000 | 1 | OK Good |
| **high\_medium\_speed** | 8 | 0.9710 | 0.8694 | 0 | OK Strong |
| **near\_long\_vehicle** | 2 | 0.9796 | 0.9593 | 0 | OK (small N) |
| **edge\_cases** | 9 | 0.7120 | 0.0000 | 2 | (!) Some failures |

---

## Zero-Score Scenarios

| Token | Bucket | Scenario Type | In V1? |
|-------|--------|---------------|--------|
| `a3b4b8e47696534e` | intersections | traversing\_traffic\_light\_intersection | ok |
| `e4d09605e2b6563e` | intersections | on\_traffic\_light\_intersection | ok |
| `48a1b05d9c6f5fd7` | intersections | traversing\_traffic\_light\_intersection | |
| `d9ee9cf40a84520f` | intersections | traversing\_traffic\_light\_intersection | |
| `eaf0200162f557cd` | intersections | traversing\_traffic\_light\_intersection | |
| `edb1d6776eb25f5d` | intersections | traversing\_traffic\_light\_intersection | |
| `0136ce79864c53e1` | stop\_and\_go | stationary\_at\_traffic\_light\_with\_lead | |
| `9e30155b8bb55fd9` | lane\_change | changing\_lane\_to\_left | |
| `4a275c4d8e125040` | edge\_cases | on\_pickup\_dropoff | |
| `cc1d648ab526530a` | edge\_cases | traversing\_pickup\_dropoff | |

**Pattern:** Most failures (7/10) are in **traffic-light intersection scenarios**. The planner gets zero score consistently in these situations, suggesting a systematic issue with traffic light signal handling or unprotected turns. This matches known limitations of purely imitation-based planners without explicit traffic-rule modules.

---

## Comparison: V1 Overlap Tokens

Tokens `a3b4b8e47696534e` and `e4d09605e2b6563e` are shared with the V1 10-scenario benchmark.
Both score **0.0** in this run (both are traffic-light intersection scenarios).

V1 benchmark targets: [benchmarks/mini_v1/baseline.md](../benchmarks/mini_v1/baseline.md)

---

## Bucket Coverage Notes

- **intersections** (21 scenarios): Heavy representation due to 14 tokens mapped to this bucket and nuplan classifying many scenarios as `traversing_traffic_light_intersection`
- **near\_long\_vehicle** (2 scenarios): Only 2 of 10 planned tokens resolved in mini split
- **lane\_change** (4 scenarios): Only 4 of 7 planned tokens resolved

The 13 unresolved tokens are from logs not present in the mini dataset split.

---

## Files

| File | Contents |
|------|---------|
| [eval_v2_20260324_173928.csv](eval_v2_20260324_173928.csv) | Per-scenario CSV (token, scenario_type, bucket, score) |
| [eval_v2_20260324_173928.md](eval_v2_20260324_173928.md) | Per-bucket summary markdown |

Aggregator parquet:
```
datasets/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/
  mini_eval_v2/planTF/v2_run1/aggregator_metric/
    closed_loop_nonreactive_agents_weighted_average_metrics_2026.03.24.17.00.57.parquet
```

---

## Reproducibility

```bash
# Run simulation
cd /path/to/planTF
PYTHONPATH=/path/to/planTF:$PYTHONPATH \
NUPLAN_DATA_ROOT=/path/to/datasets/nuplan \
conda run -n plantf --no-capture-output python run_simulation.py \
    +simulation=closed_loop_nonreactive_agents planner=planTF \
    scenario_builder=nuplan \
    "scenario_builder.data_root=$NUPLAN_DATA_ROOT/nuplan-v1.1_mini/data/cache/mini" \
    scenario_filter=mini_eval_v2 worker=sequential \
    experiment_uid="mini_eval_v2/planTF/v2_run1" verbose=true \
    "planner.imitation_planner.planner_ckpt=/path/to/planTF/checkpoints/planTF.ckpt"

# Summarize results
conda run -n plantf python script/summarize_eval.py \
    --challenge closed_loop_nonreactive_agents \
    --run-dir datasets/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/mini_eval_v2/planTF/v2_run1/aggregator_metric \
    --save
```

> **Note:** `worker=sequential` is required (Ray workers don't inherit PYTHONPATH).  
> `scenario_builder.data_root` must point to the mini-split cache directory.

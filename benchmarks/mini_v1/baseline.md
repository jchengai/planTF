# PlanTF Mini Benchmark — Baseline

**Run:** `2026.03.21.17.07.22`  
**Checkpoint:** `checkpoints/planTF.ckpt` (released weights, unchanged)  
**Scenario filter:** `config/scenario_filter/mini_benchmark.yaml`  
**Challenge:** `closed_loop_nonreactive_agents`  
**Num scenarios:** 10  

> NOTE: Do NOT change scenario tokens, checkpoint, or planner config. This is the frozen reference for all future comparisons.

---

## Quality Metrics

| Metric | Value |
|---|---|
| **Final score** | **0.7716** |
| drivable_area_compliance | 0.9 |
| driving_direction_compliance | 0.9 |
| ego_is_comfortable | 0.9 |
| no_ego_at_fault_collisions | 1.0 |
| time_to_collision_within_bound | 0.9 |
| speed_limit_compliance | 1.0 |
| ego_is_making_progress | 1.0 |

### Per-type scores

| scenario_type | score | note |
|---|---|---|
| stationary | 0.9976 | |
| medium_magnitude_speed | 0.9958 | |
| following_lane_with_slow_lead | 0.9906 | |
| high_magnitude_speed | 0.9674 | |
| near_pedestrian_on_crosswalk | 0.9478 | |
| near_long_vehicle | 0.9449 | |
| stationary_at_traffic_light_with_lead | 0.9371 | |
| traversing_intersection | 0.9347 | |
| traversing_traffic_light_intersection | **0.0000** | driving_direction_compliance = 0 |
| on_traffic_light_intersection | **0.0000** | ego_is_comfortable = 0, time_to_collision = 0 |

> The two zero-score scenarios drag the final score from ~0.97 → 0.77. See note below.

---

## Latency (1494 steps, CPU-only inference)

| Metric | Mean | P95 | Max |
|---|---|---|---|
| `feature_build_ms` | 112.78 | — | 1530 (step 0 warmup) |
| `forward_ms` (model inference) | **21.78** | **32.54** | 56.67 |
| `total_planner_ms` | **134.59** | **187.44** | 1542 (step 0 warmup) |
| `gpu_mem_mb` | 0.0 | 0.0 | 0.0 |

### Warmup
Step 0 of the first scenario spikes to ~1400 ms (`feature_build_ms` = 1376 ms). From step 1 onward, steady-state is ~120–130 ms feature build + ~22 ms inference. This is normal JIT/cache warmup — **exclude step 0 when comparing across runs**.

### GPU note
`gpu_mem_mb` = 0 throughout. The sequential worker runs on **CPU only**. Expect `forward_ms` to drop significantly (~5–10×) once GPU inference is enabled (Phase 2 / ONNX export).

---

## Zero-score scenarios — are they a problem?

**Yes, but not a setup bug.** Both scenarios are traffic-light intersection types:

- `traversing_traffic_light_intersection` — `driving_direction_compliance = 0`: the ego drives against the expected direction at some point in the scenario. Likely a mini dataset edge case where the expert trajectory turns unexpectedly.
- `on_traffic_light_intersection` — `ego_is_comfortable = 0` and `time_to_collision_within_bound = 0`: hard deceleration or near-miss. The multiplicative scoring formula zeros the full scenario score if **any** mandatory metric is zero.

**Impact on comparisons:** These two scenarios will likely remain zero (or near-zero) for any checkpoint trained without specific augmentation for these edge cases. They are **not blocking** for Phase 2 (ONNX export / inference inference), but should be noted when comparing fine-tuned checkpoints — improvement here will look large even for small real gains.

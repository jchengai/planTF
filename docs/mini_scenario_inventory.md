# Mini Dataset — Scenario Inventory

**Dataset:** nuplan-v1.1_mini  
**DB files:** 64 log files  
**Method:** `scenario_tag` table queried across all DBs; tokens are 8-byte hex identifiers  
**Date compiled:** 2026-03-24  
**Script:** raw query at `outputs/mini_scenario_inventory.csv`

---

## Type Distribution

All scenario types found in the mini split, sorted by frequency:

| Scenario Type | Total Tokens | Logs | Eval Bucket | Notes |
|---|---|---|---|---|
| `stationary` | 188,367 | 64 | — | Too trivial; not useful for eval |
| `on_intersection` | 84,376 | ~50 | — | Positional tag only; use `traversing_*` instead |
| `on_pickup_dropoff` | 78,646 | ~50 | — | Domain-specific; not prioritized |
| `near_pedestrian_on_crosswalk` | 74,407 | ~55 | **pedestrian** | Well represented; use diverse logs |
| `traversing_intersection` | 57,786 | ~55 | **intersections** | Core eval type |
| `on_traffic_light_intersection` | 57,415 | ~55 | — | Positional; known failure case in v1 |
| `medium_magnitude_speed` | 51,613 | ~60 | **high_medium_speed** | Good coverage |
| `traversing_traffic_light_intersection` | 50,622 | ~55 | **intersections** | Known failure in v1 (score=0.000) |
| `stationary_in_traffic` | 42,400 | ~50 | **stop_and_go** | Congestion scenarios |
| `traversing_pickup_dropoff` | 39,329 | ~45 | — | Not prioritized |
| `high_magnitude_speed` | 39,329 | ~55 | **high_medium_speed** | Good coverage |
| `near_pedestrian_at_pickup_dropoff` | 24,800 | ~45 | — | Subset of pedestrian; not prioritized |
| `on_stopline_traffic_light` | 21,460 | ~50 | — | Positional; covered by `stationary_at_tl_*` |
| `stationary_at_traffic_light_without_lead` | 16,020 | ~50 | **stop_and_go** | Useful: stop light, no lead |
| `near_long_vehicle` | 10,793 | ~50 | **near_long_vehicle** | Good coverage |
| `low_magnitude_speed` | 8,344 | ~45 | **high_medium_speed** | Slow ego; parking-lot like |
| `traversing_crosswalk` | 7,855 | ~45 | — | Covered by pedestrian bucket |
| `on_stopline_stop_sign` | 5,161 | ~40 | — | Covered by `starting_straight_stop_sign_*` |
| `stationary_at_traffic_light_with_lead` | 3,922 | ~40 | **stop_and_go** | Useful: stop light, with lead |
| `near_trafficcone_on_driveable` | 3,691 | ~35 | **edge_cases** | Construction / obstacle |
| `near_high_speed_vehicle` | 3,647 | ~35 | **near_long_vehicle** | Occlusion / speed differential |
| `near_multiple_vehicles` | 2,860 | ~35 | **near_long_vehicle** | Dense traffic |
| `following_lane_with_slow_lead` | 2,559 | ~40 | **lead_vehicle** | Key eval type; model known to handle well |
| `on_all_way_stop_intersection` | 2,485 | ~35 | **intersections** | All-way stop |
| `near_construction_zone_sign` | 1,457 | ~30 | **edge_cases** | Construction zone |
| `following_lane_without_lead` | 1,190 | ~30 | — | Too easy; not prioritized |
| `starting_right_turn` | 604 | ~30 | **intersections** | Turn behavior |
| `starting_left_turn` | 600 | ~30 | **intersections** | Unprotected left turn |
| `high_lateral_acceleration` | 600 | ~25 | **edge_cases** | Aggressive cornering |
| `starting_protected_noncross_turn` | 494 | ~25 | — | Covered by other turn types |
| `near_barrier_on_driveable` | 438 | ~20 | — | Similar to trafficcone |
| `accelerating_at_traffic_light` | 386 | ~25 | **stop_and_go** | Light-to-motion transition |
| `starting_unprotected_cross_turn` | 353 | ~25 | **intersections** | Higher difficulty; suspected struggle |
| `starting_straight_traffic_light_intersection_traversal` | 340 | ~20 | — | Covered by traversing_tl_intersection |
| `waiting_for_pedestrian_to_cross` | 316 | ~20 | **pedestrian** | Active yielding required |
| `starting_straight_stop_sign_intersection_traversal` | 266 | ~20 | — | Stop sign; not prioritized |
| `near_pedestrian_on_crosswalk_with_ego` | 233 | ~20 | **pedestrian** | Harder: pedestrian in ego path |
| `starting_protected_cross_turn` | 207 | ~15 | — | Protected; lower interest |
| `stopping_at_crosswalk` | 169 | ~15 | **pedestrian** | Reactive stop behavior |
| `stopping_with_lead` | 159 | ~15 | **lead_vehicle** | Emergency/comfort braking |
| `starting_unprotected_noncross_turn` | 150 | ~15 | — | Less common |
| `near_multiple_pedestrians` | 120 | ~15 | **edge_cases** | Dense pedestrian scenario |
| `changing_lane` | 22 | ~10 | **lane_change** | LIMITED: only 22 tokens |
| `stopping_at_traffic_light_without_lead` | 69 | ~15 | — | Covered by stationary_at_tl |
| `stopping_at_traffic_light_with_lead` | 18 | ~10 | — | Low count |
| `changing_lane_to_left` | 15 | ~8 | **lane_change** | VERY LIMITED: 15 tokens |
| `changing_lane_to_right` | 7 | ~5 | **lane_change** | VERY LIMITED: 7 tokens |
| `high_magnitude_jerk` | 7 | ~5 | — | Too few |
| `behind_bike` | 2 | ~2 | — | Too few |
| `traversing_narrow_lane` | 1 | ~1 | — | Too few (single instance) |

---

## Lane Change Availability — Explicit Note

**Lane change coverage is severely limited in mini.**

| Type | Count | Logs |
|---|---|---|
| `changing_lane` | 22 | ~10 |
| `changing_lane_to_left` | 15 | ~8 |
| `changing_lane_to_right` | 7 | ~5 |
| **Total lane-change tagged** | **44** | — |

The mini split does not contain substantive lane-change scenarios. The dataset
was recorded on structured urban routes where lane changes are infrequent.
The `mini_eval_v2` benchmark includes the **7 available tokens** across all
three subtypes but makes **no coverage claims** about lane-change performance.
Full-dataset evaluation (`val14`, `test14`) is required for lane-change generalization claims.

---

## Buckets Selected for mini_eval_v2

| Bucket | Types Used | Selected | Run N\* | Limited? |
|---|---|---|---|---|
| intersections | traversing_intersection, traversing_tl_intersection, unprotected_cross, left/right turn, all_way_stop | 14 | 24 | No — ample supply |
| lane_change | changing_lane, to_left, to_right | 7 | 5 | **Yes — all available tokens used** |
| lead_vehicle | following_slow_lead, following_lead, stopping_with_lead | 10 | 7 | No |
| pedestrian | near_ped_crosswalk, waiting_for_ped, near_ped_with_ego, stopping_at_crosswalk | 11 | 7 | No |
| stop_and_go | stationary_at_tl_lead, stationary_at_tl_no_lead, stationary_in_traffic, accel_at_tl | 12 | 19 | No |
| high_medium_speed | high_magnitude_speed, medium_magnitude_speed, low_magnitude_speed | 12 | 9 | No |
| near_long_vehicle | near_long_vehicle, near_multiple_vehicles, near_high_speed_vehicle | 10 | 5 | No |
| edge_cases | high_lateral_accel, near_trafficcone, near_construction, near_multiple_ped | 8 | 8 | No |
| **Total** | | **84** | **84** | |

\* **Run N** = bucket count as reported by `summarize_eval.py`, not the yaml token count. nuPlan assigns each scenario its primary type at simulation time, which may differ from the yaml selection bucket (e.g. a token placed under `lead_vehicle` may be tagged `traversing_traffic_light_intersection` and land in `intersections`). The per-bucket split can therefore shift between yaml and summarizer; the total is always 84.

---

## Known Hard Scenarios (from mini_benchmark_v1)

These two tokens scored 0.000 in the 10-scenario v1 benchmark and are deliberately
included in the expanded benchmark:

| Token | Type | Failure Mode |
|---|---|---|
| `e4d09605e2b6563e` | `on_traffic_light_intersection` | `ego_is_comfortable=0`, `time_to_collision=0` |
| `a3b4b8e47696534e` | `traversing_traffic_light_intersection` | `driving_direction_compliance=0` |

Both are in the `intersections` bucket in `mini_eval_v2`.

# Branching Guide

## Branch Overview

```
main  ──────────────────────────────────────────────── all feature branches merged
  (expanded-eval   merged March 2026, commit 46792b4)
  (cpp-runtime     merged March 2026, commit 93b367b)
  (onnx-deployment legacy, content in inference/)
  (faithful-natten legacy, content in inference/faithful/)
```

---

## What Belongs Where

### `main` — Stable baseline
- **Always runnable**: `bash script/initial_sanity_check.sh` and `bash script/mini_benchmark.sh` must pass.
- **Only merge via `--no-ff`** from a feature branch after review.
- **No in-progress work** — nothing half-implemented lives here.
- **Contains**: the 10-scenario `mini_benchmark.yaml` and its frozen `baseline.md` results.

### `expanded-eval` — Broader scenario benchmark
**Purpose**: coverage-driven evaluation beyond the 10-scenario dev benchmark.

Files that belong here:
- `config/scenario_filter/mini_eval_v2.yaml`
- `script/mini_eval_v2.sh`
- `script/summarize_eval.py`
- `docs/mini_scenario_inventory.md` / `outputs/mini_scenario_inventory.csv`
- `docs/EVAL_PLAN.md`
- Results under `outputs/eval_v2/`

**Do NOT mix in**:
- C++ code of any kind
- Changes to `inference/common/patches.py` or any planner logic
- Deployment artifacts (`.trt`, `.onnx`)

### `cpp-runtime` — Native C++ inference
**Purpose**: ONNX Runtime C++ first, TensorRT C++ second.

Files that belong here:
- `cpp/onnx_runtime_infer.cpp`
- `cpp/tensorrt_infer.cpp`
- `cpp/tensorrt_utils.h` / `.cpp`
- `cpp/CMakeLists.txt`
- `cpp/README.md`
- `cpp/dummy_input_writer.py` (helper to write test inputs)

**Do NOT mix in**:
- Any change to planner behaviour or `imitation_planner.py`
- Scenario filter changes
- Faithful/patch research
- Result claims from expanded eval

**Uses**: `inference/planTF.onnx` (patched export) — must never be confused
with the original unpatched model.

### `onnx-deployment` — Python ONNX/TRT benchmarking (legacy)
All previous Python TensorRT work is here. Keep stable. New deployment work
may start here or in `cpp-runtime`.

### `faithful-natten` — NATTEN fidelity research (legacy)
For improving the NATTEN patch accuracy. The NATTEN attention replacement is
the **only remaining source of divergence** from the original model. This
branch is for investigating opset 17 LayerNorm, RPB alternatives, or partial
faithful export paths.

---

## What Must NOT Be Mixed

| Cross-contamination | Why |
|---|---|
| Eval scenario changes in `cpp-runtime` | Benchmark integrity — eval must be independent of runtime |
| Patch logic changes in `expanded-eval` | Fidelity claims must not shift under an eval run |
| Model architecture changes in `onnx-deployment` | Would invalidate the existing ONNX/TRT engine cache |
| Deployment claims (latency, speedup) in `faithful-natten` | The patched model ≠ the faithful model |
| C++ code in `expanded-eval` | Runtime work must be independently verifiable |

---

## Suggested Merge Order

```
Phase 1:  expanded-eval   → main  [MERGED March 2026, commit 46792b4]
Phase 2:  cpp-runtime     → main  [MERGED March 2026, commit 93b367b]
Phase 3:  faithful-natten         [legacy — content lives in inference/faithful/;
                                   no formal merge commit; FIDELITY_REPORT.md is the record]
```

Merge policy:
```bash
git checkout main
git merge --no-ff expanded-eval   -m "merge: expanded evaluation benchmark v2"
git merge --no-ff cpp-runtime     -m "merge: native C++ ORT + TRT inference"
```

Never fast-forward merge to main — the merge commit is the audit trail.

---

## Guard Rules (enforced by convention, may be scripted)

1. `main` tag must pass `bash script/initial_sanity_check.sh` before merge.
2. `expanded-eval` must include an updated `docs/mini_scenario_inventory.md` before merge.
3. `cpp-runtime` must demonstrate latency parity (within 2×) of Python path before merge.
4. Deployment claims (latency numbers) are only valid against the **patched** ONNX model.
   Claims about the original faithful model require `faithful-natten` work.

# planTF — Inference & ONNX Export

Scripts in this directory run planTF inference and ONNX export
**without the nuPlan simulation framework**.

Two parallel development tracks are maintained from this folder:

- **Deploy track** (`deploy/`) — ONNX export, TensorRT, performance benchmarking
- **Faithful track** (`faithful/`) — NATTEN approximation, RPB research, accuracy

Shared utilities live in `common/`.

---

## Project Structure

```
inference/
  run_pipeline.py          # top-level CLI: --mode deploy | faithful
  verify_refactor.py       # validates folder layout and public API contracts
  README.md
  FIDELITY_REPORT.md       # numeric record of all fidelity experiments
  common/
    run_inference.py       # load_model() + make_dummy_input() helpers (shared)
    inspect_model.py       # print model architecture + parameter count
    patches.py             # all model-patching helpers (NATTEN, bool-index, atan2, ...)
  deploy/
    export_onnx.py         # ONNX export with 5 compatibility patches
    benchmark_latency.py   # mean/p95 latency: Original / Patched / ONNX
    benchmark_tensorrt.py  # TensorRT FP32/FP16 benchmark + engine build
    TENSORRT_REPORT.md     # full latency + fidelity results
  faithful/
    compare_outputs.py     # three-way fidelity: Original vs Patched vs ONNX
    ablate_patches.py      # isolate which patch causes divergence
```

`planTF.onnx` is a generated artifact produced by `deploy/export_onnx.py`; it is not stored in the repository.

---

## Branch Status

`onnx-deployment` and `faithful-natten` are legacy branches; their content has been merged to `main`. All inference work is developed on `main`. See [BRANCHING.md](../BRANCHING.md) for the full branch history and guard rules.

---

## Development Guidelines

### DO
- Use `common/run_inference.py` as the single source of truth for model loading
- Use `common/patches.py` for all patching logic; do not duplicate patch code in `deploy/` or `faithful/`
- Record numeric results in `FIDELITY_REPORT.md` before merging back to `main`

### DO NOT
- Modify original model behaviour (`src/models/`) in the deployment scripts
- Break the 5 ONNX patches — they are the foundation of the deploy track
- Confuse the patched ONNX model with the original checkpoint; latency claims are only valid for the patched model

### Faithful scripts may experiment with
- Local-window attention replacing global MHA in `patch_natten_for_onnx`
- Relative positional bias (RPB) approximations
- Custom ONNX ops / TensorRT plugins for NATTEN

---

## Scripts

| Script | Track | Purpose |
|---|---|---|
| `run_pipeline.py` | both | `--mode deploy` or `--mode faithful` end-to-end runner |
| `verify_refactor.py` | shared | Validate folder layout and public API contracts |
| `common/inspect_model.py` | shared | Load checkpoint, print architecture and parameter count |
| `common/run_inference.py` | shared | Standalone PyTorch forward pass with dummy inputs |
| `common/patches.py` | shared | All model-patching helpers (NATTEN, bool-index, atan2, MHA wrappers) |
| `deploy/export_onnx.py` | deploy | Export to ONNX with compatibility patches |
| `deploy/benchmark_latency.py` | deploy | Compare mean/p95 latency: Original / Patched / ONNX Runtime |
| `deploy/benchmark_tensorrt.py` | deploy | Build TRT engine, benchmark FP32/FP16, compare outputs |
| `faithful/compare_outputs.py` | faithful | Fidelity validation: Original vs Patched vs ONNX |
| `faithful/ablate_patches.py` | faithful | Isolate which patch(es) cause output divergence |

---

## Quick Start

```bash
# From repo root, with conda env activated

# Inspect model
python inference/common/inspect_model.py

# Single inference
python inference/common/run_inference.py

# Full deploy pipeline (ONNX export + latency benchmark)
python inference/run_pipeline.py --mode deploy

# Full faithful pipeline (fidelity comparison + ablation)
python inference/run_pipeline.py --mode faithful

# Or run individual scripts directly:
python inference/deploy/export_onnx.py
python inference/deploy/benchmark_latency.py --runs 50 --save
python inference/faithful/compare_outputs.py --reexport --save
python inference/faithful/ablate_patches.py
```

---

## Refactor Validation

The folder reorganisation into `common/` / `deploy/` / `faithful/` was verified
by `verify_refactor.py`. Run at any time with:

```bash
python inference/verify_refactor.py          # summary
python inference/verify_refactor.py --verbose # show detail on every check
```

### What was moved

| Before | After | Track |
|---|---|---|
| `inference/run_inference.py` | `inference/common/run_inference.py` | shared |
| `inference/inspect_model.py` | `inference/common/inspect_model.py` | shared |
| `inference/export_onnx.py` | `inference/deploy/export_onnx.py` | deploy |
| `inference/benchmark_latency.py` | `inference/deploy/benchmark_latency.py` | deploy |
| `inference/compare_outputs.py` | `inference/faithful/compare_outputs.py` | faithful |
| `inference/ablate_patches.py` | `inference/faithful/ablate_patches.py` | faithful |

### Invariants checked by verify_refactor.py

| Category | Checks | Result |
|---|---|---|
| `[FILE]` | All scripts + docs present; no stale root-level scripts | PASS |
| `[IMPORT]` | All 6 modules import cleanly in isolated subprocess | PASS |
| `[API]` | `load_model`, `make_dummy_input` signatures intact; output shapes correct | PASS |
| `[DEDUP]` | `load_model` / `make_dummy_input` defined **only** in `common/` | PASS |
| `[ORCH]` | `run_pipeline.py` contains no model logic, no patches, no `torch.onnx` calls | PASS |
| `[PATH]` | All `--ckpt` and `--onnx` defaults use `_repo_root`/`_inference` (not CWD) | PASS |
| `[OUTDIR]` | `outputs/` resolves to `<repo_root>/outputs` regardless of invocation directory | PASS |
| `[DEPS]` | `deploy/` does not import `faithful/`; `faithful→deploy` is intentional and noted | PASS |
| `[ONNX]` | `planTF.onnx` lives at `inference/planTF.onnx`, not inside any subdir | PASS |

### Known risks (and their mitigations)

| Risk | Mitigation |
|---|---|
| `sys.path` manipulation per-script | Paths computed via `__file__`, not `os.getcwd()` — CWD-independent |
| CWD-relative argument defaults | Hardened to absolute paths using `_repo_root` / `_inference` variables |
| `faithful/` importing `deploy/export_onnx` (cross-track) | Intentional and one-directional; deploy↛faithful enforced by `[DEPS]` check |
| ONNX file at `inference/planTF.onnx` (not inside `deploy/`) | Shared artifact consumed by both deploy and faithful; location enforced by `[ONNX]` check |
| `outputs/` directory created on first save | Resolved to `<repo_root>/outputs/` in all scripts; `[OUTDIR]` check verifies this |

---

## What Is Currently Faithful

| Component | Faithful? | Notes |
|---|---|---|
| ONNX vs Patched PyTorch | **Yes** | Max error 8e-6 — pure float rounding |
| `PointsEncoder` bool-index rewrite | **Yes** | Identical in eval mode (frozen BN) |
| `AgentEncoder` bool-index rewrite | **Yes** | Zero-gate is exact for valid agents |
| `atan2` substitute | **Yes** | Max heading error < 1e-6 rad |
| `F.interpolate` size→ fix | **Yes** | Functionally identical |

## Exact NATTEN Export (default)

`patch_natten_exact` replaces `NeighborhoodAttention1D` with `_ExactNA1D`, a
gather-based re-implementation of NATTEN's 1D kernel for dilation 1. It keeps
the relative positional bias and NATTEN's edge handling: tokens near either
end of the sequence keep a full window of `kernel_size` real neighbours
because the window shifts inward. The neighbour and bias indices depend only
on the sequence length (21 → 11 → 6 for the agent history encoder), so they
are baked into the ONNX graph as constants. Only standard ops are used
(Gather, Mul, ReduceSum, Softmax).

`tests/test_onnx_patches.py` checks it against the real NATTEN 0.14.6 package
and runs in CI (`onnx-parity` job). Measured on CPU with the released
architecture, random weights (RPB scaled to std 1 so it matters) and padded
inputs (20/33 agents, 100/152 polygons valid):

| Comparison vs original PyTorch + NATTEN | Max position error |
|---|---|
| `_ExactNA1D` module vs `NeighborhoodAttention1D` (all PlanTF lengths) | < 2e-7 |
| PyTorch, all export patches, `--natten exact` | ~1e-6 m |
| ONNX Runtime, exported with `--natten exact` | ~1e-6 m |
| PyTorch, all export patches, `--natten global` (legacy) | ~0.08 m |
| Legacy export (global NATTEN + dropped `key_padding_mask`) | ~0.45 m |

Exactness costs no latency: in the C++ ONNX Runtime benchmark (`cpp/ort_infer`,
CPU, 4 vCPU container, 200 runs) the exact graph ran at p50 ≈ 38.5 ms vs
≈ 40 ms for the legacy global-MHA graph, i.e. within noise.

These are random-weight numbers. Re-run `faithful/compare_outputs.py` and
`faithful/ablate_patches.py` with the released checkpoint to replace the
March figures in FIDELITY_REPORT.md.

## What Is Approximate

| Component | Approximation | Error introduced |
|---|---|---|
| NATTEN → global MHA (`patch_natten_for_onnx`, `--natten global`) | **Major** | Windowed+RPB → global, no RPB. Max trajectory error ≈ **0.86** on the checkpoint |
| NATTEN → local-window MHA (`patch_natten_faithful`) | **Minor** | Windowed, no RPB, zero-padded edge windows. Superseded by `patch_natten_exact` |

`key_padding_mask` is no longer dropped (see Patch 2).

---

## ONNX Export Details

The model uses several constructs incompatible with ONNX opset ≤14 in
PyTorch 1.12. `export_onnx.py` applies five in-place patches on a freshly
loaded model copy before calling `torch.onnx.export`. The original model
files are **never modified**.

### Patch 1 — `patch_natten_for_onnx`

| | |
|---|---|
| **What** | Replaces every `NeighborhoodAttention1D` (NATTEN) inside `NATLayer` blocks with `nn.MultiheadAttention` (`batch_first=False`) |
| **Scope** | 6 blocks in `NATSequenceEncoder` (agent history encoder) |
| **Weight transfer** | `qkv.weight`, `qkv.bias`, `proj.weight`, `proj.bias` copied exactly |
| **Dropped** | Relative positional bias (`rpb`) — not representable in ONNX |
| **Classification** | **Approximate / behaviour-changing** |
| **Risk** | NATTEN uses local (windowed) neighborhood attention; `nn.MultiheadAttention` computes full global attention. Output distributions will differ. The RPB encodes distance-dependent attention priors — dropping it further changes behaviour. |

### Patch 2 — `patch_mha_for_onnx`

| | |
|---|---|
| **What** | Replaces `nn.MultiheadAttention(batch_first=True)` modules in `TransformerEncoderLayer` and `StateAttentionEncoder` with `_Bf1MHAWrapper`, which computes attention explicitly from the same weights |
| **Scope** | 4 `TransformerEncoderLayer` blocks + 1 `StateAttentionEncoder` = 5 modules |
| **Mask** | `key_padding_mask` is applied as an additive float mask (−1e9), which avoids the PyTorch 1.12 exporter's `bool → Expand` bug. `StateAttentionEncoder` only builds a mask in training, so it has none at inference |
| **Classification** | **Safe** (matches original to float precision, padded inputs included) |
| **History** | Earlier versions dropped `key_padding_mask`. That is invisible with the all-valid dummy input that the March ablation used, but lets padded agents/polygons into the encoder's attention in real scenes (~0.45 m max position error with padded inputs, random weights). `test_padding_values_do_not_leak` guards against this. |

### Patch 3 — `patch_boolean_indexing_for_onnx`

| | |
|---|---|
| **What** | Replaces `x[bool_mask]`/`x[bool_mask] = y` in `PointsEncoder` and `MapEncoder` |
| **How** | `PointsEncoder`: run MLP on all inputs, zero-gate outputs for invalid tokens. `MapEncoder`: replace conditional embedding scatter with `torch.where` |
| **Classification** | **Safe transformation** |
| **Risk** | `BatchNorm1d` inside `PointsEncoder` now sees zero-padded invalid rows, which slightly shifts batch statistics. With `model.eval()` (frozen running stats) this is a no-op — batch stats are not updated. Zero-gating gives identical results to the original masked path for valid tokens. |

### Patch 4 — `patch_agent_encoder_for_onnx`

| | |
|---|---|
| **What** | Replaces `x_agent[valid_agent_mask] = encoder_output` in `AgentEncoder` |
| **How** | Run the history encoder on all agents (invalid ones receive zero features), multiply output by a float validity gate |
| **Classification** | **Safe transformation** |
| **Risk** | Numerically identical for valid agents. Invalid agents produce non-zero intermediate activations inside the encoder (due to biases), but these are zeroed before any downstream computation. |

### Patch 5 — `patch_planning_model_for_onnx`

| | |
|---|---|
| **What** | Replaces `torch.atan2(sin, cos)` in `PlanningModel.forward` |
| **How** | `atan(y/x) + quadrant_offset`, using a small epsilon for numerical stability |
| **Classification** | **Safe transformation** |
| **Risk** | Equivalent to `atan2` everywhere except near `x=0` (vertical headings), where the epsilon `1e-7` introduces a tiny error. Max heading error < `1e-6` rad in practice. |

---

## Source File Change

**`src/models/planTF/layers/embedding.py`** — `F.interpolate(scale_factor=...)` changed to
`F.interpolate(size=[...])`. Functionally identical; required because PyTorch 1.12's ONNX
shape-inference pass cannot handle a dynamic scalar `scale_factor`. Safe for all code paths.

---

## Known Risks Summary

| Patch | Classification | Expected output delta |
|---|---|---|
| NATTEN → exact NA1D (default) | **Safe** | Float precision (< 1e-6) |
| NATTEN → global MHA (legacy `--natten global`) | **Approximate** | Moderate — windowed vs global attention, no RPB |
| `key_padding_mask` as additive mask | **Safe** | Float precision |
| `PointsEncoder` bool-index | **Safe** | Negligible (eval mode BN, zero-gated output) |
| `AgentEncoder` bool-index | **Safe** | Negligible (zero-gate after encoder) |
| `atan2` substitute | **Safe** | < 1e-6 rad |

---

## Decision Flowchart

```
Run faithful/compare_outputs.py
        │
        ▼
Original vs ONNX: max_abs < 1e-2?
    ├── YES → proceed to TensorRT
    └── NO  → isolate divergence
                │
                ├── Original vs Patched large?
                │       → patch logic is wrong (check Patch 1 / Patch 2)
                └── Patched vs ONNX large?
                        → ONNX tracing differs from patch forward
                          (check tracer warnings in export output)
```

---

## Input / Output Shapes (batch=1)

```
agent.position          [1, A, 21, 2]    float32   A ≤ 33
agent.heading           [1, A, 21]       float32
agent.velocity          [1, A, 21, 2]    float32
agent.shape             [1, A, 21, 2]    float32
agent.category          [1, A]           int64
agent.valid_mask        [1, A, 21]       bool
map.point_position      [1, M, 3, 20, 2] float32  M ≤ 152
map.polygon_center      [1, M, 3]        float32
map.polygon_type        [1, M]           int64
map.polygon_on_route    [1, M]           bool
map.polygon_tl_status   [1, M]           int64
map.polygon_has_speed_limit [1, M]       bool
map.polygon_speed_limit [1, M]           float32
map.valid_mask          [1, M, 20]       bool
current_state           [1, 7]           float32

output_trajectory       [1, 80, 3]       float32   (x, y, heading) at 0.1s intervals
```

---

## C++ Runtime

ORT/TensorRT inference programs are in [`cpp/`](../cpp/). These consume the same
`planTF.onnx` generated by `deploy/export_onnx.py` and produce outputs directly
comparable to Python ORT.

See [`cpp/README.md`](../cpp/README.md) for build instructions and
[`cpp/CPP_RUNTIME_REPORT.md`](../cpp/CPP_RUNTIME_REPORT.md) for benchmark results.

# planTF ONNX Export — Fidelity Report

**Date**: March 24, 2026  
**Model**: planTF checkpoint (`checkpoints/planTF.ckpt`)  
**Input**: A=33 agents, M=152 polygons, `torch.manual_seed(0)` dummy tensors  
**Metric**: `output_trajectory` shape `[1, 80, 3]` (x, y, heading at 0.1 s intervals)

---

## Update (October 2026): both sources of divergence fixed

The March results below were measured with the legacy export and **all-valid**
dummy inputs. Two changes since then:

1. **NATTEN is now exported exactly.** `patch_natten_exact` (`_ExactNA1D`)
   re-implements NATTEN-1D with gather ops, keeping the RPB and NATTEN's edge
   windows, so option C below ("register a custom op") is not needed. It matches
   the real NATTEN 0.14.6 kernel to < 2e-7 and is the default for
   `export_onnx.py` (`--natten global` keeps the legacy approximation).
2. **`key_padding_mask` was not harmless.** The conclusion in section 4 that
   dropping it "has zero effect" holds only for all-valid inputs. With padded
   inputs (20/33 agents, 100/152 polygons valid, random weights) the legacy
   export drifts ~0.45 m from the original, more than the NATTEN approximation
   (~0.08 m). `_Bf1MHAWrapper` now applies the mask additively.

With both fixes, the exported ONNX model matches original PyTorch + NATTEN to
~1e-6 m on padded inputs (random weights; `tests/test_onnx_patches.py`, CI job
`onnx-parity`). `compare_outputs.py` and `ablate_patches.py` now use padded
inputs by default. **Still to do:** re-run both with the released checkpoint,
rebuild the TensorRT engines from the new ONNX file, and re-measure latency;
the latency and C++ numbers in this repo are for the legacy graph.

---

## 1. Summary of Results

| Comparison | Max abs error | Mean abs error | Mean pos error | Mean hdg error |
|---|---|---|---|---|
| Original (PyTorch) vs Patched (PyTorch) | **0.8595** | 0.1939 | 0.350 m | 0.109 rad |
| Original (PyTorch) vs ONNX (Runtime)   | **0.8595** | 0.1939 | 0.350 m | 0.109 rad |
| Patched (PyTorch)  vs ONNX (Runtime)   | **8.1e-6** | 2.0e-6 | 3e-6 m | 1e-6 rad |

---

## 2. Key Findings

### ONNX export is faithful to the patched model
Patched PyTorch vs ONNX Runtime max error is **8e-6** — pure floating-point rounding.
The `torch.onnx.export` + onnxruntime execution path introduces no additional semantic error.
The ONNX file is a correct serialisation of the patched model.

### Patched model diverges significantly from original
Max error between original and patched (or ONNX) is **0.86**, with mean position error
of **0.35 m** per timestep growing to **1.16 m** at the final (t=8 s) step.

This is **not** an ONNX export bug. It is a consequence of the compatibility patches
applied to make the model exportable.

---

## 3. Root-Cause Analysis

### Primary cause: NATTEN → global MultiheadAttention (Patch 1)

**Suspected contribution: ~85–100% of the divergence.**

The original model uses `NeighborhoodAttention1D` (NATTEN) — windowed local attention
where each token only attends to its `kernel_size` nearest neighbours plus a learnable
relative positional bias (RPB). The export patch replaces this with standard
`nn.MultiheadAttention` (full global attention, no RPB).

| | NATTEN | Patch replacement |
|---|---|---|
| Attention pattern | Local window (kernel_size = 3, 3, 5) | Full global |
| Relative positional bias | Yes (learned `rpb` parameter) | No (dropped) |
| Weight transfer | qkv + proj copied exactly | Yes |

**Observable at level 2**: sequence length = 5, kernel_size = 5 → already global.
**Levels 0 and 1**: sequence lengths 20 and 10 with kernel_size 3 — genuinely local.
These levels are where the approximation error is introduced.

### Secondary cause: dropped `key_padding_mask` (Patch 2)

**Suspected contribution: small with mostly-valid inputs.**

The cross-agent and agent×map transformer blocks receive a `key_padding_mask` marking
which agent/polygon slots are zero-padded. The export patch drops this mask (PyTorch 1.12
cannot serialize the bool→Expand ONNX op it generates).

With the dummy input (all slots valid), this has zero effect. In real scenes with sparse
agent counts the effect is larger, as invalid tokens dilute attention.

### Patches 3–5: verified safe

- `patch_boolean_indexing_for_onnx` — functionally identical, BN running stats frozen
- `patch_agent_encoder_for_onnx` — zero-gate is exactly correct for valid agents
- `patch_planning_model_for_onnx` (atan2) — max error < 1e-6 rad, verified safe

---

## 4. Ablation Results (from `ablate_patches.py`)

| Patch variant | Max abs error | Mean pos error | Mean hdg error |
|---|---|---|---|
| original self-check | 0.000 | 0.000 m | 0.000 rad |
| only `patch_planning_model` (atan2) | 7.2e-7 | 0.000 m | 0.000 rad |
| only `patch_boolean_indexing` | 3.3e-6 | 0.000 m | 0.000 rad |
| only `patch_agent_encoder` | 0.000 | 0.000 m | 0.000 rad |
| only `patch_mha` (drop key_padding_mask) | 2.4e-6 | 0.000 m | 0.000 rad |
| only `patch_natten` (global MHA) | **0.860** | **0.350 m** | **0.109 rad** |
| only `patch_natten_faithful` (local MHA) | **0.299** | **0.098 m** | **0.050 rad** |
| all patches (ONNX export config) | **0.860** | **0.350 m** | **0.109 rad** |

**Key finding**: NATTEN is the **sole** meaningful contributor.
Patches 1–4 are verified safe (all < 4e-6). The `key_padding_mask` drop has zero
effect on all-valid dummy inputs.

`patch_natten_faithful` (local-window MHA, same kernel_size, no RPB) reduces the
max error from 0.860 → **0.299** and position error from 0.350 → **0.098 m** vs
the original. It is closer to the original but still diverges due to missing RPB.

---

## 5. Latency Benchmark (from `benchmark_latency.py`, 50 runs, CPU)

| Variant | Mean | Median | p95 |
|---|---|---|---|
| Original PyTorch (NATTEN) | 27.0 ms | 26.1 ms | 34.9 ms |
| Patched PyTorch (global MHA) | 21.1 ms | 21.2 ms | 23.2 ms |
| ONNX Runtime | 21.1 ms | 20.8 ms | 25.5 ms |

NATTEN custom ops are **1.28× slower** than the patched (global MHA) equivalent on CPU.
ONNX Runtime matches Patched PyTorch exactly in mean latency, confirming the export
adds no runtime overhead.

---

## 6. Next Steps

| Option | When | Action |
|---|---|---|
| **A — Accept approximation** | Throughput benchmarking / latency profiling only | Proceed to TensorRT with current ONNX |
| **B — Improved NATTEN approximation** | Need closer output fidelity in PyTorch | Use `patch_natten_faithful` (local-window MHA, no RPB) — see `export_onnx.py` |
| **C — Exact replication** | Production deployment needs exact trajectory match | Register NATTEN as custom ONNX/TRT op; see [SHI-Labs/NATTEN](https://github.com/SHI-Labs/NATTEN) |

Run `ablate_patches.py` to quantitatively confirm which individual patch contributes
most to divergence before choosing an option.

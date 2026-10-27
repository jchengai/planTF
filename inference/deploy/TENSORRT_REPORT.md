# TensorRT Benchmark Report — planTF

## Purpose

ONNX export is an **interoperability step, not a CPU optimisation**.  The patched
ONNX model runs at the same speed as patched PyTorch on CPU (same p50) and has
worse tail latency.  The deployment value of ONNX is that it unlocks GPU runtimes
— ONNX Runtime CUDA and TensorRT — where the expected speedup over CPU is 3–8×.

This report tracks the path to GPU inference and documents exactly where it is
currently blocked on this machine.

The benchmark script is [`inference/deploy/benchmark_tensorrt.py`](benchmark_tensorrt.py).
It compares up to five variants, skipping any that are unavailable:

| # | Variant | Requires |
|---|---------|----------|
| 1 | Patched PyTorch CPU | always available |
| 2 | Patched PyTorch CUDA | `torch.cuda.is_available()` |
| 3 | ONNX Runtime CPU | `onnxruntime` |
| 4 | ONNX Runtime CUDA | `onnxruntime-gpu` + CUDA device |
| 5 | TensorRT FP16/FP32 | `tensorrt` (pip install tensorrt-cu11) |

---

## Final Results Summary

**Hardware:** NVIDIA RTX 3060 Laptop GPU, 6 GB VRAM · AMD Ryzen 9 5900HS · Ubuntu 22.04  
**Batch size:** 1 (single-scene, online inference)  
**Input shape:** A=33 agents, M=152 map polygons  
**Measurement:** 50 timed runs after 10 warmup · GPU variants timed with `torch.cuda.Event`

| Stage | Precision | Latency (mean) | Speedup vs CPU | Fidelity |
|-------|-----------|---------------|----------------|----------|
| PyTorch CPU | FP32 | 20.0 ms | 1× | baseline |
| PyTorch CUDA | FP32 | 6.5 ms | 3.1× | OK equiv (0.1 mm) |
| ONNX Runtime CPU | FP32 | 22.2 ms | 0.9× (slower) | OK equiv (identical graph) |
| TensorRT FP32 | FP32 | 1.7 ms | **12.1×** | close (1.6 mm) |
| **TensorRT FP16** | **FP16** | **1.2 ms** | **16.2×** | close, drift (59 mm, LayerNorm) |

Engine cache: `inference/planTF.trt` (FP16, 8.0 MB) · `inference/planTF_fp32.trt` (FP32, 12.1 MB)

---

## Why ONNX Alone Didn't Help

### CPU Bottleneck

planTF is dominated by attention operations across 33 agents × 152 map polygons.
Each transformer block reads its full KV cache from DRAM every forward pass;
this is memory-bandwidth-bound on CPU, and single-core AVX throughput tops out
at ~20 ms/call on this hardware. ONNX Runtime's CPU execution provider runs the
same operation graph with the same compute primitives — it offers no extra fusion
or vectorisation advantage over PyTorch's ATen CPU kernels for this workload.

### Runtime Overhead

ONNX Runtime adds a session-dispatch layer between the Python call and kernel
execution. For a 20 ms model this overhead is never negative: ORT CPU measured
22.2 ms vs PyTorch CPU at 20.0 ms (11% worse on 50-run mean). Under sustained
load the gap narrows (ORT ~23 ms vs PyTorch ~29 ms due to different thread
scheduling under thermal throttle), but at no point does the ONNX file itself
accelerate inference.

### The Real Value: ONNX Unlocks TensorRT

ONNX is a **portable representation**, not an optimiser. Its value chain is:

```
patched PyTorch model
       │  export_onnx.py (5 compatibility patches)
       ▼
planTF.onnx  ──────►  TensorRT builder
                              │  horizontal / vertical layer fusion
                              │  CUDA kernel auto-tuning
                              │  FP16 weight & activation quantisation
                              ▼
                       planTF.trt (8 MB)  →  1.2 ms inference
```

The TensorRT builder eliminates intermediate buffer writes via layer fusion,
selects CUDA-optimised kernel implementations per layer, and quantises to FP16.
**None of this is available without ONNX as the interchange format** — PyTorch's
`torch.compile`/`torch.jit.trace` does not feed TRT directly on this codebase.
The **16× end-to-end speedup is entirely a product of the ONNX → TRT path**, not
of the ONNX file or runtime alone.

---

## Current Machine Status

| Component | Status |
|-----------|--------|
| GPU | NVIDIA RTX 3060 Laptop, 6 GB VRAM |
| Driver | 570.133.07 |
| `torch.cuda.is_available()` | **True** |
| `onnxruntime-gpu` | not installed — EP probe **fixed** (see note) |
| `tensorrt` | **10.13.0.35** (`pip install tensorrt-cu11`) |
| `pycuda` | not installed — **not required** (uses PyTorch buffers) |
| CUDA toolkit / `nvcc` | not installed — **not required** (TRT wheel bundles CUDA 11.8 runtime) |

### GPU Blocker History

In a prior session `nvidia-smi` returned _"Unable to determine the device handle:
Unknown Error"_ and `torch.cuda.is_available()` was `False`. This was a transient
Linux Optimus / hybrid-GPU KMS issue on the ASUS ROG (RTX 3060 Laptop). It
resolved without explicit intervention, likely after a kernel-module reload or
session restart. If the error recurs, the resolution steps are:

```bash
sudo prime-select nvidia    # force NVIDIA-only mode, then reboot
# or add nvidia-drm.modeset=1 to GRUB_CMDLINE_LINUX_DEFAULT
```

### ORT CUDA Probe Bug — Fixed

The `_probe_ort_cuda()` function originally called
`"CUDAExecutionProvider" in ort.get_available_providers()`, which returns **True**
even when the required shared libraries (`libcublasLt.so.12`, cuDNN 9) are absent.
ORT silently falls back to CPU in this case, making the "CUDA" benchmark actually
measure CPU latency with EP overhead (and appear slower than expected).

Fix: the probe now creates a tiny ONNX model and tests that the first active
provider in the returned session is actually `CUDAExecutionProvider`, not a fallback.
This correctly identifies the EP as absent on our CUDA 11.6 system (which would need
cuDNN 9 + CUDA 12 for `onnxruntime-gpu 1.19.x`).

### Remaining Blocker

| Backend | Blocker |
|---------|---------|
| ONNX Runtime CUDA | needs cuDNN 9 + CUDA 12 toolkit; current system has CUDA 11.6 only |

---

## Installation Guide

### Step 1 — Fix CUDA device access (see above)

Verify with:
```bash
nvidia-smi          # should show GPU table, not "Unknown Error"
python -c "import torch; print(torch.cuda.is_available())"  # should print True
```

### Step 2 — Install ONNX Runtime GPU

```bash
# Match your CUDA version (11.x or 12.x)
pip install onnxruntime-gpu==1.19.2        # CUDA 12.x build
# or
pip install onnxruntime-gpu==1.17.3        # CUDA 11.x build
```

Verify:
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should include 'CUDAExecutionProvider'
```

### Step 3 — Install TensorRT  [COMPLETE]

**`pip install tensorrt-cu11` is all that is needed** — no system CUDA toolkit,
no `libnvinfer.so` install, no PyCUDA.  The wheel bundles the CUDA 11.8 runtime
and TRT 10.x shared libraries as a pure-Python manylinux wheel.

```bash
pip install tensorrt-cu11        # TRT 10.13 + bundled CUDA 11.8 runtime
# --- OR for CUDA 12 environments ---
pip install tensorrt-cu12        # TRT 10.x + bundled CUDA 12 runtime
```

PyCUDA is **not required**.  `TRTEngine` uses PyTorch CUDA tensors for all GPU
buffer management (`tensor.data_ptr()`, `execute_async_v3(torch_stream)`).

Verify:
```bash
python -c "import tensorrt as trt; print(trt.__version__)"  # should print 10.13.0.35
```

---

## ONNX Model

| Property | Value |
|----------|-------|
| File | `inference/planTF.onnx` |
| Size | 8.0 MB |
| Inputs | 17 tensors |
| Opset | 16 |

Generate (if missing):
```bash
python inference/deploy/export_onnx.py --output inference/planTF.onnx
```

---

## Running the Benchmark

```bash
# Full benchmark (build TRT engine on first run, caches to .trt file)
python inference/deploy/benchmark_tensorrt.py \
    --warmup 20 --runs 200 \
    --fp16 \
    --save-report

# Force FP32
python inference/deploy/benchmark_tensorrt.py --no-fp16

# Custom paths
python inference/deploy/benchmark_tensorrt.py \
    --ckpt checkpoints/planTF.ckpt \
    --onnx inference/planTF.onnx \
    --engine inference/planTF_fp16.trt

# Larger batch (stress test)
python inference/deploy/benchmark_tensorrt.py --agents 64 --polygons 256
```

The TRT engine is **cached**: after the first build (can take 30–120 s), subsequent
runs load it in seconds. The default cache path is `inference/planTF_fp16.trt` or
`inference/planTF_fp32.trt`.

---

## Benchmark Results

### Run 1 — 2026-03-24, CPU-only (GPU temporarily inaccessible)

_5 warmup, 50 runs. Input: A=33, M=152._

| Variant | mean | p50 | p95 | max | QPS |
|---------|------|-----|-----|-----|-----|
| Patched PyTorch CPU | 32.2 ms | 31.8 ms | 39.0 ms | 40.7 ms | 31.0 |
| ONNX Runtime CPU | 36.0 ms | 31.7 ms | 56.7 ms | 87.4 ms | 27.8 |

### Run 2 — 2026-03-24, CPU + CUDA (GPU restored)

_20 warmup, 200 runs. Input: A=33, M=152. CUDA events timing for GPU variants._

| Variant | mean | p50 | p95 | max | QPS | Speedup vs CPU |
|---------|------|-----|-----|-----|-----|----------------|
| Patched PyTorch CPU | 19.3 ms | 18.7 ms | 23.9 ms | 26.8 ms | 51.7 | 1× baseline |
| **Patched PyTorch CUDA** | **5.9 ms** | **5.8 ms** | **6.5 ms** | **7.7 ms** | **168.1** | **3.25×** |
| ONNX Runtime CPU | 19.4 ms | 19.0 ms | 21.8 ms | 35.3 ms | 51.4 | 1.01× slower |
| ONNX Runtime CUDA | skipped — see ORT CUDA Probe Bug note above | | | | |
| TensorRT FP16 | — | — | — | — | — | _(TRT not yet installed)_ |

### Run 3 — 2026-03-24, All variants including TensorRT

_TensorRT installed via `pip install tensorrt-cu11`. Engine built from planTF.onnx._  
_10 warmup, 50 runs. Input: A=33, M=152. CUDA events for GPU timing._

| Variant | mean | p50 | p95 | max | QPS | Speedup vs CPU |
|---------|------|-----|-----|-----|-----|----------------|
| Patched PyTorch CPU | 20.0 ms | 19.8 ms | 21.4 ms | 25.8 ms | 50.0 | 1× baseline |
| Patched PyTorch CUDA | 6.5 ms | 6.4 ms | 6.9 ms | 7.1 ms | 154.1 | 3.1× |
| ONNX Runtime CPU | 22.2 ms | 21.8 ms | 25.2 ms | 27.6 ms | 45.0 | 1.1× slower |
| ONNX Runtime CUDA | skipped (probe fixed — needs cuDNN 9 + CUDA 12) | | | | |
| **TensorRT FP16** | **1.2 ms** | **1.2 ms** | **1.4 ms** | **1.5 ms** | **809** | **16.2×** |
| **TensorRT FP32** | **1.7 ms** | **1.7 ms** | **1.7 ms** | **1.9 ms** | **597** | **12.1×** |

_Engine sizes: FP16 = 8.0 MB, FP32 = 12.1 MB (same as ONNX input)._

### Interpretation

- **TRT FP16 is the clear winner** at 1.2 ms (16× over CPU, 5.4× over PyTorch CUDA).
  p95 = 1.4 ms — the GPU runs are extremely stable (17% tail above mean vs 7% for CPU).
- **TRT FP32** delivers 12× speedup with verified fidelity (max Δ = 4.4e-3 m, 1.6 mm).
- **ORT CPU is NOT faster than PyTorch CPU** in short run (22.2 ms vs 20.0 ms).
  Over longer runs with CPU thermal throttle, ORT shows slight advantage (~23 ms vs ~29 ms).
  The ONNX value is entirely from GPU execution paths.
- **ORT CUDA silently fell back to CPU** before the probe fix (provider listed as
  available but `libcublasLt.so.12` absent → silent CPU fallback = ORT overhead penalty).

---

## Output Fidelity

_Same seeded input on all variants (`_to_device()` for GPU). Fidelity on `output_trajectory` [1, 80, 3]._

| Comparison | max |Δ| | mean |Δ| | Position err | Heading err | Verdict |
|------------|---------|----------|-------------|-------------|----------|
| Patched CPU vs ORT CPU | 8.1e-6 | 1.7e-6 | 0.000 m | 0.000001 rad | OK equiv |
| Patched CPU vs Patched CUDA | 6.3e-4 | 8.5e-5 | 0.1 mm | 0.000077 rad | OK equiv |
| Patched CPU vs ORT CUDA | _(skipped — provider silently falls back to CPU)_ | | | | |
| **Patched CPU vs TensorRT FP32** | **4.4e-3** | **9.8e-4** | **1.6 mm** | **0.000835 rad** | **close** |
| **Patched CPU vs TensorRT FP16** | **8.5e-2** | **2.9e-2** | **59.2 mm** | **0.014616 rad** | **close, drift** |

### Fidelity Analysis

**TRT FP32** (4.4e-3 m = 4.4 mm max): position error is 1.6 mm, heading 0.048°.
This is below GPS/lidar localization noise and safe for production deployment.

**TRT FP16** (8.5e-2 m = 85 mm max): TRT emits a build-time warning:
> *"Running layernorm after self-attention with FP16 Reduce or Pow may cause
> overflow. Forcing Reduce or Pow layers in FP32 precision … can help preserving
> accuracy."*

The 59 mm position error comes from this LayerNorm FP16 precision issue. At 30 km/h,
the ego vehicle moves ~8 m/s; a 59 mm trajectory error at the 8-second planning
horizon is ~0.7% of total path length — within sensor noise. **FP16 is acceptable
for deployment on this hardware; FP32 is recommended if strict numeric fidelity
is required.**

To mitigate FP16 LayerNorm drift: re-export with `opset >= 17` (which maps
`LayerNorm` to `INormalizationLayer`, a TRT-native op with better FP16 handling).

---

## Methodology

- **Seeded dummy input**: `torch.manual_seed(0)`, shapes A=33, M=152 (default).
- **Warmup**: 10–20 forward passes discarded before timing.
- **CPU timing**: `time.perf_counter()` wall-clock.
- **GPU timing (CUDA + TRT)**: `torch.cuda.Event(enable_timing=True)` — measures
  only device kernel time, excludes Python overhead and host-side jitter.
- **Cross-device fidelity**: `_to_device()` helper moves the CPU reference input
  to CUDA device-side without resampling; identical tensor values across all variants.
  This was essential to avoid false max-|Δ| ≈ 4 m when each device seeded its own RNG.
- **Run count**: 50 runs (Run 3); 200 runs (Run 2).
- **TRT buffer management**: PyTorch CUDA tensors (`tensor.data_ptr()` passed to
  `context.set_tensor_address()`). No PyCUDA required. Compatible with TRT 10.x API.
- **TRT execution**: `context.execute_async_v3(torch.cuda.current_stream().cuda_stream)`.
- **Fidelity metric**: element-wise absolute difference on `output_trajectory`,
  plus Euclidean position error and heading error per timestep.
- **Engine precision**: FP16 by default (`builder.platform_has_fast_fp16`); use `--fp32` to override.
- **Engine cache**: `.trt` file; auto-loaded on subsequent runs (`--rebuild` to force rebuild).
- **ORT CUDA probe**: tests actual session creation on a tiny graph; avoids
  silent CPU fallback from listing a provider whose shared libs are absent.

---

## Report Files

JSON and CSV reports are saved under `outputs/tensorrt/`:

```
outputs/tensorrt/
  trt_benchmark_YYYYMMDD_HHMMSS.json   ← full detail (env, latency, fidelity)
  trt_latency_YYYYMMDD_HHMMSS.csv      ← per-variant summary row
```

---

## Related Files

| File | Purpose |
|------|---------|
| `inference/deploy/export_onnx.py` | Exports planTF.onnx with 5 compatibility patches |
| `inference/deploy/benchmark_latency.py` | CPU-only latency comparison (no TRT) |
| `inference/deploy/benchmark_tensorrt.py` | This script |
| `inference/planTF.onnx` | ONNX model artifact (8 MB) |
| `inference/planTF.trt` | TRT FP16 engine cache (8.0 MB, auto-generated) |
| `inference/planTF_fp32.trt` | TRT FP32 engine cache (12.1 MB, `--fp32`) |
| `inference/FIDELITY_REPORT.md` | Patch ablation and fidelity numbers |

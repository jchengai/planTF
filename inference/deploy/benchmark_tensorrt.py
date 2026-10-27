"""
benchmark_tensorrt.py
---------------------
Benchmark planTF inference latency across five variants on the
DEPLOYMENT TRACK (patched model only — no NATTEN fidelity experiments here):

  1. Patched PyTorch (CPU)           — always available baseline
  2. Patched PyTorch (CUDA)          — requires torch.cuda.is_available()
  3. ONNX Runtime CPU                — always available
  4. ONNX Runtime CUDA               — requires onnxruntime-gpu
  5. TensorRT FP16/FP32              — requires tensorrt (pip install tensorrt-cu11)

All variants use the SAME seeded dummy input via _to_device() so fidelity
comparisons are numerically meaningful across CPU and GPU.

The TRT engine is built from planTF.onnx the first time and cached to disk.
On subsequent runs it is loaded from cache.  FP16 precision is used when the
GPU supports it; otherwise falls back to FP32.

GPU inference (CUDA + TRT) uses CUDA events for timing and PyTorch tensors
for GPU buffer management.  PyCUDA is NOT required.

Usage (from repo root):
    python inference/deploy/benchmark_tensorrt.py
    python inference/deploy/benchmark_tensorrt.py --fp32       # force FP32
    python inference/deploy/benchmark_tensorrt.py --rebuild    # force engine rebuild
    python inference/deploy/benchmark_tensorrt.py --runs 200 --warmup 20
    python inference/deploy/benchmark_tensorrt.py --save-report
    python inference/deploy/benchmark_tensorrt.py --cpu-only   # skip TRT/CUDA ORT

Environment requirements for full TRT benchmark:
    pip install tensorrt-cu11        # TRT 10.x with bundled CUDA 11.8 runtime
    pip install tensorrt-cu12        # alternative for CUDA 12 environments
    pip install onnxruntime-gpu      # for ONNX Runtime CUDA variant

When TensorRT is absent the script runs the PyTorch + ORT CPU variants and
prints a clear setup guide.  Exit code is 0 in all cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time

_here       = os.path.dirname(os.path.abspath(__file__))
_inference = os.path.dirname(_here)
_repo_root  = os.path.dirname(_inference)
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_inference, "common"))

import numpy as np
import torch

from run_inference import load_model, make_dummy_input, _metrics
from patches import apply_onnx_export_patches


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability probes
# ---------------------------------------------------------------------------

def _probe_trt():
    """Return (tensorrt_module, trt_version_str) or (None, reason_string)."""
    try:
        import tensorrt as trt
        return trt, trt.__version__
    except ImportError:
        return None, "not installed"
    except Exception as e:
        return None, str(e)


def _probe_pycuda():
    try:
        import pycuda.driver as drv
        drv.init()
        if drv.Device.count() == 0:
            return None, "no CUDA devices"
        return drv, drv.Device.count()
    except ImportError:
        return None, "pycuda not installed"
    except Exception as e:
        return None, str(e)


def _probe_cuda_torch():
    """True if torch.cuda is functional (device accessible)."""
    try:
        if not torch.cuda.is_available():
            return False
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


def _probe_ort_cuda():
    """True only if ORT CUDAExecutionProvider can actually create a session.

    `ort.get_available_providers()` lists CUDA even when the required libs
    (libcublasLt, cuDNN) are absent; we test actual session creation to avoid
    a silent CPU fallback during benchmarking.
    """
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
        # Runtime test: create a tiny ONNX model and run it on CUDA EP
        import numpy as np, io
        import onnx
        from onnx import helper, TensorProto
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        node = helper.make_node("Relu", inputs=["x"], outputs=["y"])
        graph = helper.make_graph([node], "test", [x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        buf = model.SerializeToString()
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # suppress warnings
        sess = ort.InferenceSession(buf, sess_opts,
                                    providers=["CUDAExecutionProvider",
                                               "CPUExecutionProvider"])
        # Check the first provider is actually CUDA (not silently fallen back)
        active = sess.get_providers()
        return len(active) > 0 and active[0] == "CUDAExecutionProvider"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Input flattening helpers
# ---------------------------------------------------------------------------

def _build_trt_feeds(data: dict) -> dict:
    """Flatten the structured dummy-input dict to a flat {name: np.ndarray} dict.

    This is the canonical feed-dict builder used by both ORT and TRT.  TRT
    receives the full dict directly; ORT filters it via _build_ort_feeds.
    """
    ag, mp = data["agent"], data["map"]
    return {
        "a_position":                ag["position"].numpy(),
        "a_heading":                 ag["heading"].numpy(),
        "a_velocity":                ag["velocity"].numpy(),
        "a_shape":                   ag["shape"].numpy(),
        "a_category":                ag["category"].numpy(),
        "a_valid_mask":              ag["valid_mask"].numpy(),
        "a_target":                  ag["target"].numpy(),
        "m_point_position":          mp["point_position"].numpy(),
        "m_point_vector":            mp["point_vector"].numpy(),
        "m_point_orientation":       mp["point_orientation"].numpy(),
        "m_point_side":              mp["point_side"].numpy(),
        "m_polygon_center":          mp["polygon_center"].numpy(),
        "m_polygon_position":        mp["polygon_position"].numpy(),
        "m_polygon_orientation":     mp["polygon_orientation"].numpy(),
        "m_polygon_type":            mp["polygon_type"].numpy(),
        "m_polygon_on_route":        mp["polygon_on_route"].numpy(),
        "m_polygon_tl_status":       mp["polygon_tl_status"].numpy(),
        "m_polygon_has_speed_limit": mp["polygon_has_speed_limit"].numpy(),
        "m_polygon_speed_limit":     mp["polygon_speed_limit"].numpy(),
        "m_valid_mask":              mp["valid_mask"].numpy(),
        "current_state":             data["current_state"].numpy(),
        "origin":                    data["origin"].numpy(),
        "angle":                     data["angle"].numpy(),
    }


def _build_ort_feeds(sess, data: dict) -> dict:
    """Filter _build_trt_feeds to only the names the ORT session expects."""
    pool = _build_trt_feeds(data)
    return {inp.name: pool[inp.name] for inp in sess.get_inputs() if inp.name in pool}


# ---------------------------------------------------------------------------
# TensorRT engine  —  TRT 10.x, PyTorch GPU buffers (no pycuda required)
# ---------------------------------------------------------------------------

def _np_to_torch_dtype(np_dtype: np.dtype) -> torch.dtype:
    """Map a numpy scalar dtype to its closest torch equivalent."""
    _map = {
        np.float32: torch.float32,
        np.float16: torch.float16,
        np.int32:   torch.int32,
        np.int64:   torch.int64,
        np.int8:    torch.int8,
        np.bool_:   torch.bool,
    }
    return _map.get(np_dtype, torch.float32)


class TRTEngine:
    """
    TensorRT 10.x engine wrapper using PyTorch CUDA tensors for buffer
    management.  No PyCUDA required — PyTorch's allocator and stream
    infrastructure is used throughout.

    Handles:
      - building from ONNX with FP16 / FP32 (TRTEngine.build)
      - loading a cached serialised engine from disk
      - synchronous inference: H→D copy, execute_async_v3, D→H copy

    Static input shapes only (planTF's patched ONNX is fully static).
    """

    def __init__(self, engine_path: str):
        import tensorrt as trt

        # Ensure a live CUDA context through PyTorch before TRT Runtime init
        _dev = torch.device("cuda")
        torch.zeros(1, device=_dev)

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime    = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        # Allocate I/O tensors as CUDA PyTorch tensors
        self._inputs   = {}   # name → torch.Tensor on cuda
        self._outputs  = {}   # name → torch.Tensor on cuda
        self._np_types = {}   # name → np.dtype  (for H→D casting)

        for i in range(self.engine.num_io_tensors):
            name     = self.engine.get_tensor_name(i)
            shape    = tuple(self.engine.get_tensor_shape(name))
            np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            t_dtype  = _np_to_torch_dtype(np_dtype)
            t        = torch.zeros(shape, dtype=t_dtype, device="cuda")

            self._np_types[name] = np_dtype
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._inputs[name]  = t
            else:
                self._outputs[name] = t

        # Register tensor addresses once (static shapes → pointers are stable)
        for name, t in {**self._inputs, **self._outputs}.items():
            self.context.set_tensor_address(name, t.data_ptr())

    def infer(self, feeds: dict) -> dict:
        """
        feeds: {name: np.ndarray}  — must cover all engine input names.
        Returns {name: np.ndarray} for every output tensor.
        """
        # H → D
        for name, arr in feeds.items():
            if name not in self._inputs:
                continue
            t = self._inputs[name]
            t.copy_(torch.as_tensor(
                arr.reshape(t.shape).astype(self._np_types[name])
            ))

        # Execute and sync
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)
        torch.cuda.synchronize()

        # D → H
        return {name: t.cpu().numpy() for name, t in self._outputs.items()}

    @staticmethod
    def build(
        onnx_path: str,
        engine_path: str,
        fp16: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Build a TensorRT engine from an ONNX file and write to engine_path.
        Uses PyTorch's CUDA context — no PyCUDA required.
        """
        import tensorrt as trt

        # Activate CUDA context via PyTorch before TRT builder init
        torch.zeros(1, device="cuda")

        trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network()
        parser  = trt.OnnxParser(network, trt_logger)

        print(f"  Parsing ONNX: {onnx_path} ...", end=" ", flush=True)
        with open(onnx_path, "rb") as f:
            ok = parser.parse(f.read())
        if not ok:
            for i in range(parser.num_errors):
                print(f"\n  Parser error {i}: {parser.get_error(i)}")
            raise RuntimeError("TensorRT ONNX parse failed")
        print("OK")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        fp16_supported = builder.platform_has_fast_fp16
        use_fp16 = fp16 and fp16_supported
        if fp16 and not fp16_supported:
            print("  WARNING: GPU does not support fast FP16 — falling back to FP32")
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  Precision: FP16")
        else:
            print("  Precision: FP32")

        print(f"  Building engine (this may take 1–5 min) ...", end=" ", flush=True)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build returned None")

        os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(bytes(serialized))
        size_mb = os.path.getsize(engine_path) / 1e6
        print(f"OK ({size_mb:.1f} MB)")
        print(f"  Saved engine: {engine_path}")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_pytorch(model, data, warmup, runs) -> np.ndarray:
    with torch.no_grad():
        for _ in range(warmup):
            model(data)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(data)
            times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def _time_pytorch_cuda(model, data_cuda, warmup, runs) -> np.ndarray:
    with torch.no_grad():
        for _ in range(warmup):
            model(data_cuda)
            torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(data_cuda)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def _time_ort(sess, feeds, warmup, runs) -> np.ndarray:
    for _ in range(warmup):
        sess.run(None, feeds)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def _time_trt(engine: "TRTEngine", feeds, warmup, runs) -> np.ndarray:
    """Time TRT inference with CUDA events (no pycuda required)."""
    s_start = torch.cuda.Event(enable_timing=True)
    s_end   = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(warmup):
            engine.infer(feeds)

        times = []
        for _ in range(runs):
            s_start.record()
            engine.infer(feeds)
            s_end.record()
            s_end.synchronize()
            times.append(s_start.elapsed_time(s_end))  # ms, GPU-side
    return np.array(times)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _stats(times: np.ndarray) -> dict:
    return {
        "mean_ms":       float(times.mean()),
        "p50_ms":        float(np.percentile(times, 50)),
        "p95_ms":        float(np.percentile(times, 95)),
        "min_ms":        float(times.min()),
        "max_ms":        float(times.max()),
        "throughput_qps": float(1000.0 / times.mean()),
        "n_runs":        len(times),
    }


def _to_device(data: dict, device: torch.device) -> dict:
    """Recursively move a nested dict of tensors to `device`.

    Use this instead of make_dummy_input(device=...) so that CPU and GPU
    variants receive numerically identical inputs and fidelity comparisons
    are meaningful.
    """
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


# ---------------------------------------------------------------------------
# Pretty-printers
# ---------------------------------------------------------------------------

_W_LABEL = 28
_W_COL   = 9


def _print_latency_table(results: dict) -> None:
    cols = ["mean_ms", "p50_ms", "p95_ms", "max_ms", "throughput_qps"]
    hdrs = ["mean", "p50", "p95", "max", "qps"]
    print()
    print("=" * ((_W_LABEL + 2) + len(cols) * (_W_COL + 2)))
    header = f"  {'Variant':{_W_LABEL}}"
    for h in hdrs:
        header += f"  {h:>{_W_COL}}"
    print(header)
    print("  " + "-" * (_W_LABEL + len(cols) * (_W_COL + 2)))
    baseline = None
    for label, stats in results.items():
        row = f"  {label:{_W_LABEL}}"
        for c in cols:
            v = stats[c]
            if c == "throughput_qps":
                row += f"  {v:>{_W_COL}.1f}"
            else:
                row += f"  {v:>{_W_COL - 2}.1f}ms"
        print(row)
        if baseline is None:
            baseline = stats["mean_ms"]
    print("=" * ((_W_LABEL + 2) + len(cols) * (_W_COL + 2)))

    # Speedup column
    if len(results) > 1:
        baseline_label = list(results.keys())[0]
        print(f"\n  Speedup vs '{baseline_label}':")
        for label, stats in results.items():
            if label == baseline_label:
                continue
            ratio = baseline / stats["mean_ms"]
            direction = f"{ratio:.2f}× faster" if ratio >= 1 else f"{1/ratio:.2f}× slower"
            print(f"    {label}: {direction}")


def _print_fidelity_table(fidelity: dict) -> None:
    print()
    print("=" * 78)
    header = f"  {'Comparison':<35}  {'max |Δ|':>10}  {'mean |Δ|':>10}  {'pos err':>9}  {'hdg err':>9}"
    print(header)
    print("  " + "-" * 76)
    for label, m in fidelity.items():
        row = f"  {label:<35}  {m['max_abs']:>10.4e}  {m['mean_abs']:>10.4e}  {m['pos_err_m']:>8.4f}m  {m['hdg_err_rad']:>8.6f}r"
        flag = ""
        if m["max_abs"] < 1e-3:
            flag = "  OK equiv"
        elif m["max_abs"] < 0.1:
            flag = "  ~ close"
        else:
            flag = "  diff"
        print(row + flag)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------

def _save_report(
    out_dir: str,
    latency: dict,
    fidelity: dict,
    env: dict,
    agents: int,
    polygons: int,
    warmup: int,
    runs: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # JSON — full detail
    report = {
        "timestamp": ts,
        "environment": env,
        "input_config": {"agents": agents, "polygons": polygons, "warmup": warmup, "runs": runs},
        "latency": latency,
        "fidelity": fidelity,
    }
    json_path = os.path.join(out_dir, f"trt_benchmark_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  JSON report → {json_path}")

    # CSV — latency summary
    csv_path = os.path.join(out_dir, f"trt_latency_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "mean_ms", "p50_ms", "p95_ms",
                                               "min_ms", "max_ms", "throughput_qps", "n_runs"])
        writer.writeheader()
        for label, stats in latency.items():
            writer.writerow({"variant": label, **stats})
    print(f"  CSV  report → {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _default_ckpt   = os.path.join(_repo_root, "checkpoints", "planTF.ckpt")
    _default_onnx   = os.path.join(_inference, "planTF.onnx")
    _default_engine = os.path.join(_inference, "planTF.trt")
    _default_outdir = os.path.join(_repo_root, "outputs", "tensorrt")

    parser = argparse.ArgumentParser(
        description="TensorRT + ONNX Runtime + PyTorch latency comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ckpt",        default=_default_ckpt)
    parser.add_argument("--onnx",        default=_default_onnx)
    parser.add_argument("--engine",      default=_default_engine,
                        help="Path to TRT engine cache (built if absent)")
    parser.add_argument("--agents",      type=int,  default=33)
    parser.add_argument("--polygons",    type=int,  default=152)
    parser.add_argument("--natten",   choices=["exact", "global"], default="exact",
                        help="NATTEN replacement: exact (default) or legacy global MHA")
    parser.add_argument("--warmup",      type=int,  default=10)
    parser.add_argument("--runs",        type=int,  default=100)
    parser.add_argument("--fp32",        action="store_true",
                        help="Force FP32 (skip FP16 even if GPU supports it)")
    parser.add_argument("--rebuild",     action="store_true",
                        help="Rebuild TRT engine even if cache exists")
    parser.add_argument("--cpu-only",    action="store_true",
                        help="Skip TRT and CUDA ORT (CPU variants only)")
    parser.add_argument("--save-report", action="store_true",
                        help=f"Write JSON + CSV to {_default_outdir}")
    parser.add_argument("--out-dir",     default=_default_outdir,
                        help="Override output directory for --save-report")
    args = parser.parse_args()

    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Environment probe
    # ------------------------------------------------------------------
    trt_mod, trt_info  = _probe_trt()
    pycuda_drv, cuda_info = _probe_pycuda()
    cuda_torch   = _probe_cuda_torch()
    ort_cuda_ok  = _probe_ort_cuda()

    # TRT now uses PyTorch GPU buffers — pycuda is NOT required
    trt_ok   = trt_mod is not None and cuda_torch and not args.cpu_only
    cuda_ort = ort_cuda_ok and not args.cpu_only
    cuda_pt  = cuda_torch and not args.cpu_only

    env = {
        "trt_available":    trt_mod is not None,
        "trt_version":      trt_info,
        "pycuda_available": pycuda_drv is not None,  # informational only
        "cuda_info":        str(cuda_info),
        "cuda_torch":       cuda_torch,
        "ort_cuda":         ort_cuda_ok,
        "fp16_requested":   not args.fp32,
    }

    print("\n" + "=" * 60)
    print("  planTF TensorRT Benchmark — Environment")
    print("=" * 60)
    print(f"  PyTorch:          {torch.__version__}  (CUDA: {torch.version.cuda})")
    print(f"  torch.cuda OK:    {cuda_torch}")
    _ort_ver = "?"
    try:
        import onnxruntime as ort
        _ort_ver = ort.__version__
    except Exception:
        pass
    print(f"  ORT version:      {_ort_ver}")
    print(f"  ORT CUDA EP:      {ort_cuda_ok}")
    print(f"  TensorRT:         {trt_info}")
    print(f"  PyCUDA:           {cuda_info if pycuda_drv is None else 'OK  (' + str(cuda_info) + ' device(s))'}  (optional)")
    print(f"  FP16 mode:        {'FP32 (forced)' if args.fp32 else 'FP16 (auto-detect)'}")
    print()

    if not trt_ok:
        _print_setup_guide(trt_mod, cuda_torch)

    # ------------------------------------------------------------------
    # Shared dummy input
    # ------------------------------------------------------------------
    cpu_data = make_dummy_input(A=args.agents, M=args.polygons)

    print(f"Input  A={args.agents}  M={args.polygons}  warmup={args.warmup}  runs={args.runs}\n")

    latency  = {}
    outputs  = {}   # label → np.ndarray [1, 80, 3]
    fidelity = {}

    # ------------------------------------------------------------------
    # 1. Patched PyTorch CPU  (always runs)
    # ------------------------------------------------------------------
    print("── 1. Patched PyTorch (CPU) ──────────────────────────────────")
    model_p = load_model(args.ckpt, torch.device("cpu"))
    apply_onnx_export_patches(model_p, natten=args.natten)
    with torch.no_grad():
        outputs["Patched PyTorch CPU"] = model_p(cpu_data)["output_trajectory"].cpu().numpy()
    print(f"  Timing ({args.warmup} warmup, {args.runs} runs) ...", end=" ", flush=True)
    t = _time_pytorch(model_p, cpu_data, args.warmup, args.runs)
    latency["Patched PyTorch CPU"] = _stats(t)
    print(f"done  mean={t.mean():.1f}ms  p50={np.percentile(t,50):.1f}ms  p95={np.percentile(t,95):.1f}ms")
    del model_p

    # ------------------------------------------------------------------
    # 2. Patched PyTorch CUDA  (if CUDA available)
    # ------------------------------------------------------------------
    if cuda_pt:
        print("── 2. Patched PyTorch (CUDA) ─────────────────────────────────")
        cuda_device = torch.device("cuda")
        model_pc = load_model(args.ckpt, cuda_device)
        apply_onnx_export_patches(model_pc, natten=args.natten)
        model_pc = model_pc.to(cuda_device)  # move patch wrappers to CUDA
        cuda_data = _to_device(cpu_data, cuda_device)
        with torch.no_grad():
            outputs["Patched PyTorch CUDA"] = model_pc(cuda_data)["output_trajectory"].cpu().numpy()
        print(f"  Timing ({args.warmup} warmup, {args.runs} runs) ...", end=" ", flush=True)
        t = _time_pytorch_cuda(model_pc, cuda_data, args.warmup, args.runs)
        latency["Patched PyTorch CUDA"] = _stats(t)
        print(f"done  mean={t.mean():.1f}ms  p50={np.percentile(t,50):.1f}ms  p95={np.percentile(t,95):.1f}ms")
        del model_pc
    else:
        print("── 2. Patched PyTorch (CUDA) — SKIPPED (CUDA not available) ──")

    # ------------------------------------------------------------------
    # 3. ONNX Runtime CPU  (always runs if ONNX file exists)
    # ------------------------------------------------------------------
    print("── 3. ONNX Runtime (CPU) ─────────────────────────────────────")
    _onnx_ok = os.path.isfile(args.onnx)
    if not _onnx_ok:
        print(f"  ONNX file not found: {args.onnx}")
        print("  Run: python inference/deploy/export_onnx.py  to generate it")
    else:
        import onnxruntime as ort
        sess_cpu = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        feeds = _build_ort_feeds(sess_cpu, cpu_data)
        outputs["ONNX Runtime CPU"] = sess_cpu.run(None, feeds)[0]
        print(f"  Timing ({args.warmup} warmup, {args.runs} runs) ...", end=" ", flush=True)
        t = _time_ort(sess_cpu, feeds, args.warmup, args.runs)
        latency["ONNX Runtime CPU"] = _stats(t)
        print(f"done  mean={t.mean():.1f}ms  p50={np.percentile(t,50):.1f}ms  p95={np.percentile(t,95):.1f}ms")

    # ------------------------------------------------------------------
    # 4. ONNX Runtime CUDA  (if ort-gpu available)
    # ------------------------------------------------------------------
    if cuda_ort and _onnx_ok:
        print("── 4. ONNX Runtime (CUDA) ────────────────────────────────────")
        import onnxruntime as ort
        sess_gpu = ort.InferenceSession(
            args.onnx,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        feeds_gpu = _build_ort_feeds(sess_gpu, cpu_data)
        outputs["ONNX Runtime CUDA"] = sess_gpu.run(None, feeds_gpu)[0]
        print(f"  Timing ({args.warmup} warmup, {args.runs} runs) ...", end=" ", flush=True)
        t = _time_ort(sess_gpu, feeds_gpu, args.warmup, args.runs)
        latency["ONNX Runtime CUDA"] = _stats(t)
        print(f"done  mean={t.mean():.1f}ms  p50={np.percentile(t,50):.1f}ms  p95={np.percentile(t,95):.1f}ms")
    elif not cuda_ort and not args.cpu_only:
        print("── 4. ONNX Runtime (CUDA) — SKIPPED (CUDAExecutionProvider absent) ──")

    # ------------------------------------------------------------------
    # 5. TensorRT
    # ------------------------------------------------------------------
    if trt_ok and _onnx_ok:
        print("── 5. TensorRT ───────────────────────────────────────────────")
        engine_path = args.engine
        use_fp16    = not args.fp32

        need_build = args.rebuild or not os.path.isfile(engine_path)
        if need_build:
            reason = "forced rebuild" if args.rebuild else "no cache found"
            print(f"  Building TRT engine ({reason}) ...")
            TRTEngine.build(args.onnx, engine_path, fp16=use_fp16)
        else:
            size_mb = os.path.getsize(engine_path) / 1e6
            print(f"  Loaded cached engine: {engine_path}  ({size_mb:.1f} MB)")

        engine    = TRTEngine(engine_path)
        feeds_trt = _build_trt_feeds(cpu_data)   # full flat dict, no ORT session needed

        trt_out  = engine.infer(feeds_trt)
        # The ONNX model has a single output tensor named "output_trajectory"
        trt_traj = trt_out.get("output_trajectory", list(trt_out.values())[0])
        trt_label = "TensorRT FP16" if use_fp16 else "TensorRT FP32"
        outputs[trt_label] = trt_traj

        print(f"  Timing ({args.warmup} warmup, {args.runs} runs, CUDA events) ...", end=" ", flush=True)
        t = _time_trt(engine, feeds_trt, args.warmup, args.runs)
        latency[trt_label] = _stats(t)
        print(f"done  mean={t.mean():.1f}ms  p50={np.percentile(t,50):.1f}ms  p95={np.percentile(t,95):.1f}ms")
    elif not trt_ok:
        print("── 5. TensorRT — SKIPPED (see setup guide above) ────────────")

    # ------------------------------------------------------------------
    # Fidelity comparison  (all vs Patched PyTorch CPU as reference)
    # ------------------------------------------------------------------
    ref_label = "Patched PyTorch CPU"
    ref = outputs.get(ref_label)
    if ref is not None and len(outputs) > 1:
        for label, arr in outputs.items():
            if label == ref_label:
                continue
            fidelity[f"{ref_label}  vs  {label}"] = _metrics(ref, arr)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  LATENCY RESULTS")
    print("=" * 60)
    _print_latency_table(latency)

    if fidelity:
        print("\n" + "=" * 60)
        print("  OUTPUT FIDELITY  (vs Patched PyTorch CPU, output_trajectory)")
        print("=" * 60)
        _print_fidelity_table(fidelity)

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    if args.save_report:
        _save_report(
            args.out_dir, latency, fidelity, env,
            args.agents, args.polygons, args.warmup, args.runs,
        )

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print()
    trt_label = next((k for k in latency if k.startswith("TensorRT")), None)
    if trt_ok and trt_label:
        trt_mean   = latency[trt_label]["mean_ms"]
        pt_mean    = latency["Patched PyTorch CPU"]["mean_ms"]
        speedup    = pt_mean / trt_mean
        trt_fid    = fidelity.get(f"Patched PyTorch CPU  vs  {trt_label}", {})
        equiv      = trt_fid.get("max_abs", 1.0) < 1e-2
        print(f"  {trt_label} speedup vs Patched CPU: {speedup:.2f}×")
        print(f"  TRT fidelity vs patched model:   {'OK: equivalent (max<0.01)' if equiv else 'NOTE: check FP16 drift'}")
    else:
        print("  TensorRT was not benchmarked on this machine.")
        print("  Install TensorRT (pip install tensorrt-cu11) and run again.")
        print("  See inference/deploy/TENSORRT_REPORT.md for setup instructions.")
    print()


# ---------------------------------------------------------------------------
# Setup guide (shown when TRT unavailable)
# ---------------------------------------------------------------------------

def _print_setup_guide(trt_mod, cuda_torch) -> None:
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│  TensorRT benchmark requires additional setup               │")
    print("└─────────────────────────────────────────────────────────────┘")
    if trt_mod is None:
        print("  · Install TensorRT (bundled CUDA 11 runtime, no nvcc needed):")
        print("      pip install tensorrt-cu11")
        print("    For CUDA 12 environments:")
        print("      pip install tensorrt-cu12")
    if not cuda_torch:
        print("  · CUDA device not accessible (driver/runtime mismatch).")
        print("    Check: nvidia-smi  and  torch.cuda.is_available()")
        print("    If running in a container, ensure --gpus all is set.")
    print()
    print("  Note: PyCUDA is NOT required — TRT uses PyTorch GPU buffers.")
    print("  CPU variants (Patched PyTorch + ONNX Runtime) will still run.")
    print()


if __name__ == "__main__":
    main()

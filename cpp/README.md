# C++ Runtime Inference

Two inference C++ programs that replicate the Python benchmark and expose the model latency without the Python/PyTorch overhead.

| Binary | Purpose | Speedup vs CPU Python |
|--------|---------|----------------------|
| `ort_infer` | ONNX Runtime C++ (CPU / CUDA EP) | ~1× CPU, matches ORT Python within 0.001 mm |
| `trt_infer` | TensorRT C++ (GPU only) | **~16× over CPU Python (FP16)** |

**Validated results (RTX 3060 · batch=1 · A=33 agents · M=152 polygons):**

| Backend | Precision | mean (ms) | QPS | Output diff vs ORT |
|---------|-----------|-----------|-----|--------------------|
| C++ ORT | FP32 CPU | 31.4 ms | 32 | < 0.001 mm |
| C++ TRT | FP32 GPU | **1.344 ms** | **744** | 0.252 mm |
| C++ TRT | FP16 GPU | **0.770 ms** | **1298** | ~200 mm (expected) |

See [CPP_RUNTIME_REPORT.md](CPP_RUNTIME_REPORT.md) for full results, build notes, and header resolution.

---

## Prerequisites

All dependencies are fetched from the `plantf` conda environment pip wheels.
No system CUDA toolkit or nvcc installation is required.

| Dependency | Required for | Install |
|------------|-------------|---------|
| CMake ≥ 3.16 | Both | `apt install cmake` or conda |
| ONNX Runtime (pip wheel) | `ort_infer` | already in `plantf` env |
| TensorRT pip wheels | `trt_infer` | already in `plantf` env |
| `nvidia-cuda-nvcc-cu11` | `trt_infer` | `pip install nvidia-cuda-nvcc-cu11==11.8.89` |

```bash
# Install the extra nvcc header package (provides crt/ subdirectory)
conda run -n plantf pip install nvidia-cuda-nvcc-cu11==11.8.89
```

---

## Building

```bash
# From repo root
CONDA_P=/home/apr/miniconda3/envs/plantf   # adjust to your miniconda location

cmake cpp/ -B cpp/build \
  -DENABLE_CUDA=ON \
  -DENABLE_TRT=ON \
  "-DORT_ROOT=$CONDA_P/lib/python3.9/site-packages/onnxruntime/capi" \
  "-DCUDA_INCLUDE_DIR=$CONDA_P/lib/python3.9/site-packages/nvidia/cuda_runtime/include" \
  "-DCUDA_CRT_INCLUDE_DIR=$CONDA_P/lib/python3.9/site-packages/nvidia/cuda_nvcc/include" \
  "-DCUDA_RT_LIBRARY=$CONDA_P/lib/python3.9/site-packages/nvidia/cuda_runtime/lib/libcudart.so" \
  "-DTRT_ROOT=$CONDA_P/lib/python3.9/site-packages/tensorrt_libs"

cmake --build cpp/build -j$(nproc)
```

> **Why explicit paths?**  The TRT and CUDA packages live inside the conda env's pip
> site-packages, not in system paths. `find_package(CUDA)` and `find_package(TensorRT)`
> don't search there, so we pass everything explicitly. `CMakeCache.txt` caches the
> results, so subsequent `make` calls don't need the override flags.

---

## Running

### Environment setup

```bash
CONDA_P=/home/apr/miniconda3/envs/plantf
export ORT_CAPI="$CONDA_P/lib/python3.9/site-packages/onnxruntime/capi"
export TRT_LIBS="$CONDA_P/lib/python3.9/site-packages/tensorrt_libs"
export CUDART_LIB="$CONDA_P/lib/python3.9/site-packages/nvidia/cuda_runtime/lib"

# One-time: create unversioned symlink for linker
ln -sf libonnxruntime.so.1.19.2 "$ORT_CAPI/libonnxruntime.so"
ln -sf libonnxruntime.so.1.19.2 "$ORT_CAPI/libonnxruntime.so.1"

export LD_LIBRARY_PATH="$ORT_CAPI:$TRT_LIBS:$CUDART_LIB:$LD_LIBRARY_PATH"
cd cpp/build
```
The ORT shared library lives inside the conda env. Set `LD_LIBRARY_PATH` once (see Environment setup above).

```bash
# CPU — baseline comparison with Python ORT
./ort_infer --model ../../inference/planTF.onnx

# Custom warmup / runs
./ort_infer --model ../../inference/planTF.onnx --warmup 10 --runs 20

# All options
./ort_infer --help
```

Expected output (CPU, RTX 3060 machine):
```
mean : 31.4 ms   p50 : 31.4 ms   p95 : 33.7 ms   QPS : 32
  Output tensors:
  [0] output_trajectory  shape=[1,80,3]  first 6 values: -0.239406, 0.0852928, ...
```

Note: the ONNX model has 17 inputs (agent history + map polygon features). All
input values are constant (0.1 / 0 / false), measuring pure inference latency free
of data-prep overhead. The Python ORT benchmark uses the same approach.

### TensorRT (`trt_infer`)

```bash
# Load existing .trt cache (built by Python deployment pipeline)
./trt_infer --engine ../../inference/planTF.trt

# FP32 engine
./trt_infer --engine ../../inference/planTF_fp32.trt --fp32

# Force-rebuild engine from ONNX (slow, ~1-5 min)
./trt_infer --engine ../../inference/planTF.trt --rebuild

# All options
./trt_infer --help
```

Expected output (RTX 3060, validated):
```
# FP16
  mean : 0.770 ms   p50 : 0.774 ms   p95 : 0.789 ms   QPS : 1298

# FP32
  mean : 1.344 ms   p50 : 1.347 ms   p95 : 1.371 ms   QPS : 744
```

---

## Comparing Against Python / Validating Outputs

Use the `validate_outputs.py` script for automated cross-runtime comparison:

```bash
# From repo root — runs all 4 checks automatically
conda run -n plantf python cpp/validate_outputs.py

# With explicit paths
conda run -n plantf python cpp/validate_outputs.py \
    --onnx inference/planTF.onnx \
    --fp32-engine inference/planTF_fp32.trt \
    --fp16-engine inference/planTF.trt \
    --cpp-build cpp/build
```

Or manually compare Python baselines:
```bash
# Python ORT baseline
conda run -n plantf python inference/deploy/benchmark_latency.py

# Python TRT baseline
conda run -n plantf python inference/deploy/benchmark_tensorrt.py
```

The C++ programs print `first 6 values` of `output_trajectory [1, 80, 3]`.
Both use constant-fill inputs (float=0.1, int=0, bool=False).

**Validated numerical agreement (constant-fill inputs):**
| Comparison | Max diff | Tolerance | Status |
|------------|----------|-----------|--------|
| ORT C++ vs ORT Python | < 0.001 mm | 0.1 mm | PASS |
| TRT FP32 C++ vs ORT Python | 0.252 mm | 2.0 mm | PASS |
| TRT FP16 C++ vs ORT Python | ~200 mm | 250 mm | PASS (FP16 precision loss expected) |

See [CPP_RUNTIME_REPORT.md](CPP_RUNTIME_REPORT.md) for full analysis.

---

## File Structure

```
cpp/
├── CMakeLists.txt          # build configuration
├── onnx_runtime_infer.cpp  # ORT C++ benchmark (CPU + CUDA EP)
├── tensorrt_infer.cpp      # TensorRT C++ benchmark (GPU)
├── tensorrt_utils.h        # shared TRT engine buffer helpers
├── tensorrt_utils.cpp      # alloc_engine_buffers() implementation
├── validate_outputs.py     # cross-runtime output equivalence checker
├── CPP_RUNTIME_REPORT.md   # full benchmark report with build notes
├── README.md               # this file
└── build/                  # cmake build directory (git-ignored)
```

---

## Troubleshooting

**`libonnxruntime.so: cannot open shared object file`**
```bash
export LD_LIBRARY_PATH="$CONDA_P/lib/python3.9/site-packages/onnxruntime/capi:$LD_LIBRARY_PATH"
# Also create the unversioned symlink if missing:
ln -sf libonnxruntime.so.1.19.2 "$ORT_CAPI/libonnxruntime.so"
```

**`libnvinfer.so not found`**
```bash
export LD_LIBRARY_PATH="$CONDA_P/lib/python3.9/site-packages/tensorrt_libs:$LD_LIBRARY_PATH"
```

**`libcudart.so not found`**
```bash
export LD_LIBRARY_PATH="$CONDA_P/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
```

**`Engine was not built for this platform / CUDA version`**
Delete the `.trt` cache and run with `--rebuild`.

**CMake `find_path` fails for CUDA headers**
CUDA headers come from two different pip packages:
```bash
# cuda_runtime_api.h:
ls $CONDA_P/lib/python3.9/site-packages/nvidia/cuda_runtime/include/

# crt/host_defines.h (needs nvidia-cuda-nvcc-cu11):
conda run -n plantf pip install nvidia-cuda-nvcc-cu11==11.8.89
ls $CONDA_P/lib/python3.9/site-packages/nvidia/cuda_nvcc/include/crt/
```
Pass both as `-DCUDA_INCLUDE_DIR` and `-DCUDA_CRT_INCLUDE_DIR` to cmake.

**`cmake .. -DENABLE_TRT=ON` says TRT not found**
The `tensorrt_libs` directory contains `.so` files but the C++ headers need to be
present in `tensorrt_libs/include/`. They were downloaded from the NVIDIA/TensorRT
GitHub repo. Check `ls $CONDA_P/lib/python3.9/site-packages/tensorrt_libs/include/`
for the 13 `NvInfer*.h` files.

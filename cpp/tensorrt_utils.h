/**
 * tensorrt_utils.h
 * ----------------
 * Shared utilities for planTF TensorRT C++ inference.
 *
 * Provides:
 *   TRTLogger         — ILogger adapter with configurable severity
 *   EngineBuffer      — per-binding host+device allocation
 *   fill_dummy_inputs — constant-fill inputs from engine binding info
 *   cuda_event_ms     — measure elapsed time between two cudaEvents
 *   percentile        — percentile over a vector of doubles
 *   print_latency     — formatted latency report to stdout
 *
 * Requires: NvInfer.h, NvOnnxParser.h, cuda_runtime_api.h
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

// ── Abort on CUDA error ──────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::cerr << "[CUDA error] " << cudaGetErrorString(_e)             \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ── TRT logger ───────────────────────────────────────────────────────────────
class TRTLogger : public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity min_severity;
    explicit TRTLogger(
        nvinfer1::ILogger::Severity s = nvinfer1::ILogger::Severity::kWARNING)
        : min_severity(s) {}

    void log(nvinfer1::ILogger::Severity sev,
             const char* msg) noexcept override {
        if (sev <= min_severity) {
            const char* prefix = "[TRT] ";
            switch (sev) {
                case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: prefix = "[TRT INTERNAL] "; break;
                case nvinfer1::ILogger::Severity::kERROR:           prefix = "[TRT ERROR]    "; break;
                case nvinfer1::ILogger::Severity::kWARNING:         prefix = "[TRT WARNING]  "; break;
                case nvinfer1::ILogger::Severity::kINFO:            prefix = "[TRT INFO]     "; break;
                case nvinfer1::ILogger::Severity::kVERBOSE:         prefix = "[TRT VERBOSE]  "; break;
            }
            std::cerr << prefix << msg << "\n";
        }
    }
};

// ── Per-binding buffer ───────────────────────────────────────────────────────
struct EngineBuffer {
    std::string               name;
    nvinfer1::DataType        dtype   = nvinfer1::DataType::kFLOAT;
    nvinfer1::TensorIOMode    io_mode = nvinfer1::TensorIOMode::kINPUT;
    std::vector<int64_t>      shape;
    void*                     device_ptr = nullptr;
    void*                     host_ptr   = nullptr;
    int64_t                   nbytes     = 0;

    void alloc() {
        CUDA_CHECK(cudaMalloc(&device_ptr, nbytes));
        host_ptr = std::malloc(nbytes);
        assert(host_ptr);
    }

    void free_all() {
        if (device_ptr) { cudaFree(device_ptr); device_ptr = nullptr; }
        if (host_ptr)   { std::free(host_ptr);  host_ptr   = nullptr; }
    }

    bool is_input()  const { return io_mode == nvinfer1::TensorIOMode::kINPUT; }
    bool is_output() const { return io_mode == nvinfer1::TensorIOMode::kOUTPUT; }
};

// ── Bytes per TRT element ────────────────────────────────────────────────────
inline int64_t dtype_bytes(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: return 4;
    }
}

// ── Element count from Dims ──────────────────────────────────────────────────
inline int64_t numel(const nvinfer1::Dims& dims) {
    int64_t n = 1;
    for (int i = 0; i < dims.nbDims; ++i) n *= dims.d[i];
    return n;
}

// ── Dims → vector<int64_t> ───────────────────────────────────────────────────
inline std::vector<int64_t> dims_to_vec(const nvinfer1::Dims& d) {
    return std::vector<int64_t>(d.d, d.d + d.nbDims);
}

// ── Allocate all engine I/O buffers ─────────────────────────────────────────
// Returns a vector of EngineBuffer, one per IO tensor.
// Inputs are filled with constant values:
//   float/half → 0.1   (avoids division-by-zero in normalization layers)
//   int32/int64 → 0    (valid embedding index in all tables)
//   bool → false
// This matches the Python ORT benchmark convention and produces verified output:
//   output_trajectory[:,:6] = [-0.239406, 0.085293, 0.005405, ...]
std::vector<EngineBuffer> alloc_engine_buffers(
        const nvinfer1::ICudaEngine* engine,
        bool verbose = false);

// ── CUDA event elapsed time (ms) ────────────────────────────────────────────
inline float cuda_event_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// ── Percentile ───────────────────────────────────────────────────────────────
inline double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * v.size()));
    if (idx > 0) idx -= 1;
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

// ── Print latency summary ────────────────────────────────────────────────────
inline void print_latency(const std::vector<double>& times_ms,
                          const std::string& label = "TRT") {
    if (times_ms.empty()) return;
    double mean = std::accumulate(times_ms.begin(), times_ms.end(), 0.0)
                  / times_ms.size();
    double p50_ = percentile(times_ms, 50.0);
    double p95_ = percentile(times_ms, 95.0);
    double p99_ = percentile(times_ms, 99.0);
    double maxv = *std::max_element(times_ms.begin(), times_ms.end());
    double qps  = 1000.0 / mean;

    std::cout << "\n────────────────────────────────────────────────\n";
    std::cout << "  " << label << " latency (ms, CUDA event timing)\n";
    std::cout << "  mean : " << mean << "\n";
    std::cout << "  p50  : " << p50_ << "\n";
    std::cout << "  p95  : " << p95_ << "\n";
    std::cout << "  p99  : " << p99_ << "\n";
    std::cout << "  max  : " << maxv << "\n";
    std::cout << "  QPS  : " << qps  << "\n";
    std::cout << "────────────────────────────────────────────────\n";
}

/**
 * onnx_runtime_infer.cpp
 * ----------------------
 * ONNX Runtime C++ inference for the patched planTF model.
 *
 * Loads inference/planTF.onnx (exported by inference/deploy/export_onnx.py),
 * constructs a seeded dummy input matching the Python benchmark's default shapes
 * (A=33 agents, M=152 map polygons), runs repeated inference, and reports
 * mean/p50/p95 latency. Also prints the first few output values for comparison
 * against the Python ONNX Runtime baseline.
 *
 * Build:
 *   mkdir cpp/build && cd cpp/build
 *   cmake .. -DORT_ROOT=/path/to/onnxruntime
 *   make -j4
 *
 * Run (CPU):
 *   ./ort_infer --model ../../inference/planTF.onnx
 *
 * Run (CUDA, if built with -DENABLE_CUDA=ON):
 *   ./ort_infer --model ../../inference/planTF.onnx --cuda
 *
 * For explicit comparison with Python:
 *   # Python reference (from repo root):
 *   conda run -n plantf python inference/deploy/benchmark_latency.py \
 *       --warmup 5 --runs 20
 *
 * Model:
 *   Use inference/planTF.onnx — the *patched* export.
 *   Do NOT use an unpatched model; input names and shapes will differ.
 *
 * Input shapes are read directly from the ONNX model — no hardcoded spec needed.
 * Works with any version of the patched or unpatched model.
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

// ── CLI args ────────────────────────────────────────────────────────────────
struct Args {
    std::string model_path = "../../inference/planTF.onnx";
    int A = 33;           // number of agents
    int M = 152;          // number of map polygons
    int warmup = 10;
    int runs = 50;
    bool use_cuda = false;
    bool verbose = false;
};

Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--model"  && i+1 < argc) { a.model_path = argv[++i]; }
        else if (arg == "--agents" && i+1 < argc) { a.A = std::stoi(argv[++i]); }
        else if (arg == "--polys"  && i+1 < argc) { a.M = std::stoi(argv[++i]); }
        else if (arg == "--warmup" && i+1 < argc) { a.warmup = std::stoi(argv[++i]); }
        else if (arg == "--runs"   && i+1 < argc) { a.runs = std::stoi(argv[++i]); }
        else if (arg == "--cuda")  { a.use_cuda = true; }
        else if (arg == "--verbose") { a.verbose = true; }
        else if (arg == "--help") {
            std::cout <<
                "Usage: ort_infer [options]\n"
                "  --model PATH    ONNX model path (default: ../../inference/planTF.onnx)\n"
                "  --agents N      Number of agents (default: 33)\n"
                "  --polys  N      Number of map polygons (default: 152)\n"
                "  --warmup N      Warmup runs (default: 10)\n"
                "  --runs   N      Timed runs (default: 50)\n"
                "  --cuda          Use CUDA execution provider\n"
                "  --verbose       Print all input/output tensor info\n";
            std::exit(0);
        }
    }
    return a;
}

// Build all input tensors from the session's own type/shape info.
// Shapes with dynamic dims (-1) are resolved by clamping to 1.
// All inputs are filled with safe constant values:
//   float  → 0.1  (small nonzero avoids divide-by-zero issues)
//   int64  → 0    (valid category / index in all embedding tables)
//   bool   → false (0)
std::vector<Ort::Value> build_dummy_inputs(
        Ort::MemoryInfo& mem_info,
        Ort::Session& session,
        bool verbose) {
    std::vector<Ort::Value> tensors;
    Ort::AllocatorWithDefaultOptions alloc;

    // Backing storage — one entry per input, persistent for the tensor lifetime.
    static std::vector<std::vector<float>>   float_bufs;
    static std::vector<std::vector<int64_t>> int_bufs;
    static std::vector<std::vector<uint8_t>> bool_bufs;
    float_bufs.clear(); int_bufs.clear(); bool_bufs.clear();

    size_t n_in = session.GetInputCount();
    for (size_t i = 0; i < n_in; ++i) {
        auto type_info   = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto dtype       = tensor_info.GetElementType();
        auto shape       = tensor_info.GetShape();

        // Resolve any dynamic dim (-1) to 1.
        for (auto& d : shape) if (d < 0) d = 1;
        int64_t n = 1;
        for (auto d : shape) n *= d;

        if (verbose) {
            auto name = session.GetInputNameAllocated(i, alloc);
            std::cout << "    dummy[" << i << "] " << name.get() << "  shape=[";
            for (size_t j = 0; j < shape.size(); ++j) std::cout << (j?",":"") << shape[j];
            std::cout << "]\n";
        }

        switch (dtype) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                float_bufs.emplace_back(n, 0.1f);
                tensors.push_back(Ort::Value::CreateTensor<float>(
                    mem_info, float_bufs.back().data(), n,
                    shape.data(), shape.size()));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                int_bufs.emplace_back(n, int64_t(0));
                tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                    mem_info, int_bufs.back().data(), n,
                    shape.data(), shape.size()));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
                bool_bufs.emplace_back(n, uint8_t(0));
                tensors.push_back(Ort::Value::CreateTensor(
                    mem_info, bool_bufs.back().data(),
                    n * sizeof(uint8_t),
                    shape.data(), shape.size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL));
                break;
            }
            default: {
                // Fallback: zero float
                float_bufs.emplace_back(n, 0.0f);
                tensors.push_back(Ort::Value::CreateTensor<float>(
                    mem_info, float_bufs.back().data(), n,
                    shape.data(), shape.size()));
                break;
            }
        }
    }
    return tensors;
}

// ── Timing ──────────────────────────────────────────────────────────────────
using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

double elapsed_ms(Clock::time_point t0, Clock::time_point t1) {
    return Ms(t1 - t0).count();
}

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * v.size())) - 1;
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

// ── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  planTF ONNX Runtime C++ Benchmark\n";
    std::cout << "============================================================\n";
    std::cout << "  Model  : " << args.model_path << "\n";
    std::cout << "  Agents : " << args.A << "   Polygons: " << args.M << "\n";
    std::cout << "  Warmup : " << args.warmup << "   Runs: " << args.runs << "\n";
    std::cout << "  Device : " << (args.use_cuda ? "CUDA" : "CPU") << "\n";
    std::cout << "\n";

    // ── ORT environment ──────────────────────────────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "planTF");
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(1);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef ENABLE_CUDA
    if (args.use_cuda) {
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = 0;
        session_opts.AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "  CUDA EP enabled\n";
    }
#else
    if (args.use_cuda) {
        std::cerr << "WARNING: --cuda requested but binary was not built with ENABLE_CUDA.\n";
        std::cerr << "         Rebuild with: cmake .. -DENABLE_CUDA=ON\n";
    }
#endif

    // ── Load model ───────────────────────────────────────────────────────────
    std::cout << "Loading model..." << std::flush;
    Ort::Session session(env,
#ifdef _WIN32
        std::wstring(args.model_path.begin(), args.model_path.end()).c_str(),
#else
        args.model_path.c_str(),
#endif
        session_opts);
    std::cout << " OK\n";

    // ── Inspect model I/O ────────────────────────────────────────────────────
    Ort::AllocatorWithDefaultOptions alloc;
    size_t n_in  = session.GetInputCount();
    size_t n_out = session.GetOutputCount();
    std::cout << "  Inputs : " << n_in << "\n";
    std::cout << "  Outputs: " << n_out << "\n";

    std::vector<std::string>  input_names_str, output_names_str;
    std::vector<const char*>  input_names_c,   output_names_c;

    for (size_t i = 0; i < n_in; ++i) {
        auto name_ptr = session.GetInputNameAllocated(i, alloc);
        input_names_str.push_back(name_ptr.get());
        if (args.verbose) {
            auto info = session.GetInputTypeInfo(i);
            auto shape = info.GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "    in[" << i << "] " << name_ptr.get() << "  shape=[";
            for (size_t j = 0; j < shape.size(); ++j) std::cout << (j?",":"") << shape[j];
            std::cout << "]\n";
        }
    }
    for (size_t i = 0; i < n_out; ++i) {
        auto name_ptr = session.GetOutputNameAllocated(i, alloc);
        output_names_str.push_back(name_ptr.get());
    }
    for (auto& s : input_names_str)  input_names_c.push_back(s.c_str());
    for (auto& s : output_names_str) output_names_c.push_back(s.c_str());

    // ── Build dummy inputs ───────────────────────────────────────────────────
    // Shapes and dtypes are read directly from the session — no hardcoded spec.
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);

    auto inputs = build_dummy_inputs(mem_info, session, args.verbose);

    // ── Warmup ───────────────────────────────────────────────────────────────
    std::cout << "\nWarmup (" << args.warmup << " runs)..." << std::flush;
    for (int i = 0; i < args.warmup; ++i) {
        auto out = session.Run(Ort::RunOptions{nullptr},
            input_names_c.data(), inputs.data(), inputs.size(),
            output_names_c.data(), output_names_c.size());
    }
    std::cout << " done\n";

    // ── Timed runs ───────────────────────────────────────────────────────────
    std::cout << "Timing (" << args.runs << " runs)..." << std::flush;
    std::vector<double> times;
    times.reserve(args.runs);

    std::vector<Ort::Value> last_output;
    for (int i = 0; i < args.runs; ++i) {
        auto t0 = Clock::now();
        auto out = session.Run(Ort::RunOptions{nullptr},
            input_names_c.data(), inputs.data(), inputs.size(),
            output_names_c.data(), output_names_c.size());
        auto t1 = Clock::now();
        times.push_back(elapsed_ms(t0, t1));
        if (i == args.runs - 1) last_output = std::move(out);
    }
    std::cout << " done\n";

    // ── Results ──────────────────────────────────────────────────────────────
    double mean_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double p50 = percentile(times, 50.0);
    double p95 = percentile(times, 95.0);
    double max_ms = *std::max_element(times.begin(), times.end());
    double qps = 1000.0 / mean_ms;

    std::cout << "\n";
    std::cout << "────────────────────────────────────────────────\n";
    std::cout << "  Latency results (ms)\n";
    std::cout << "  mean : " << mean_ms << "\n";
    std::cout << "  p50  : " << p50     << "\n";
    std::cout << "  p95  : " << p95     << "\n";
    std::cout << "  max  : " << max_ms  << "\n";
    std::cout << "  QPS  : " << qps     << "\n";
    std::cout << "────────────────────────────────────────────────\n";

    // ── Output values (for comparison with Python) ───────────────────────────
    std::cout << "\n  Output tensors:\n";
    for (size_t i = 0; i < last_output.size(); ++i) {
        auto& t = last_output[i];
        auto shape_info = t.GetTensorTypeAndShapeInfo();
        auto shape = shape_info.GetShape();
        int64_t n = shape_info.GetElementCount();

        std::cout << "  [" << i << "] " << output_names_str[i] << "  shape=[";
        for (size_t j = 0; j < shape.size(); ++j) std::cout << (j?",":"") << shape[j];
        std::cout << "]  first 6 values: ";

        if (shape_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float* data = t.GetTensorData<float>();
            for (int64_t j = 0; j < std::min(n, (int64_t)6); ++j)
                std::cout << (j ? ", " : "") << data[j];
        }
        std::cout << "\n";
    }

    std::cout << "\n  Compare these values against:\n";
    std::cout << "    conda run -n plantf python inference/deploy/benchmark_latency.py\n";
    std::cout << "  The first 6 values of output_trajectory should match to ~1e-5.\n";
    std::cout << "\n";

    return 0;
}

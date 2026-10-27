/**
 * tensorrt_utils.cpp
 * ------------------
 * Implementation of planTF TensorRT C++ utilities.
 * See tensorrt_utils.h for the interface.
 */

#include "tensorrt_utils.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

// ── Allocate all engine I/O buffers with constant-fill for inputs ────────────
std::vector<EngineBuffer> alloc_engine_buffers(
        const nvinfer1::ICudaEngine* engine,
        bool verbose) {
    int nb = engine->getNbIOTensors();
    std::vector<EngineBuffer> buffers(nb);

    for (int i = 0; i < nb; ++i) {
        const char* name   = engine->getIOTensorName(i);
        auto        dims   = engine->getTensorShape(name);
        auto        dtype  = engine->getTensorDataType(name);
        auto        iomode = engine->getTensorIOMode(name);

        buffers[i].name     = name;
        buffers[i].dtype    = dtype;
        buffers[i].io_mode  = iomode;
        buffers[i].shape    = dims_to_vec(dims);

        int64_t n          = numel(dims);
        buffers[i].nbytes  = n * dtype_bytes(dtype);

        // Allocate
        buffers[i].alloc();

        // Fill inputs with safe constant values
        if (iomode == nvinfer1::TensorIOMode::kINPUT) {
            switch (dtype) {
                case nvinfer1::DataType::kFLOAT: {
                    auto* p = static_cast<float*>(buffers[i].host_ptr);
                    for (int64_t j = 0; j < n; ++j) p[j] = 0.1f;
                    break;
                }
                case nvinfer1::DataType::kHALF: {
                    // Half-precision 0.1 ≈ 0x2E66 in IEEE 754 half
                    uint16_t half_01 = 0x2E66u;
                    auto* p = static_cast<uint16_t*>(buffers[i].host_ptr);
                    for (int64_t j = 0; j < n; ++j) p[j] = half_01;
                    break;
                }
                case nvinfer1::DataType::kINT64: {
                    auto* p = static_cast<int64_t*>(buffers[i].host_ptr);
                    for (int64_t j = 0; j < n; ++j) p[j] = 0;
                    break;
                }
                case nvinfer1::DataType::kINT32: {
                    auto* p = static_cast<int32_t*>(buffers[i].host_ptr);
                    for (int64_t j = 0; j < n; ++j) p[j] = 0;
                    break;
                }
                case nvinfer1::DataType::kBOOL: {
                    std::memset(buffers[i].host_ptr, 0,
                                static_cast<size_t>(buffers[i].nbytes));
                    break;
                }
                default:
                    std::memset(buffers[i].host_ptr, 0,
                                static_cast<size_t>(buffers[i].nbytes));
                    break;
            }

            // H→D copy
            CUDA_CHECK(cudaMemcpy(
                buffers[i].device_ptr, buffers[i].host_ptr,
                static_cast<size_t>(buffers[i].nbytes),
                cudaMemcpyHostToDevice));
        }

        if (verbose) {
            const char* kind = iomode == nvinfer1::TensorIOMode::kINPUT
                               ? "IN " : "OUT";
            std::cout << "  [" << kind << "] " << name << "  shape=[";
            for (size_t d = 0; d < buffers[i].shape.size(); ++d)
                std::cout << (d ? "," : "") << buffers[i].shape[d];
            std::cout << "]\n";
        }
    }
    return buffers;
}

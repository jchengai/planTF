cmake_minimum_required(VERSION 3.16)
project(planTF_cpp_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Build options ────────────────────────────────────────────────────────────
option(ENABLE_CUDA   "Enable ONNX Runtime CUDA execution provider" OFF)
option(ENABLE_TRT    "Enable TensorRT C++ inference target"        OFF)
option(ENABLE_ASAN   "Enable AddressSanitizer (debug only)"        OFF)

# ── ONNX Runtime ─────────────────────────────────────────────────────────────
# Set ORT_ROOT to your ONNX Runtime installation, e.g.:
#   cmake .. -DORT_ROOT=/opt/onnxruntime-linux-x64-1.19.2
# Or install system-wide and rely on find_package.
if(NOT DEFINED ORT_ROOT)
    # Try to find in conda env if CONDA_PREFIX is set
    if(DEFINED ENV{CONDA_PREFIX})
        set(ORT_ROOT "$ENV{CONDA_PREFIX}")
    endif()
endif()

find_path(ORT_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        "${ORT_ROOT}"
        "${ORT_ROOT}/include"
        "${ORT_ROOT}/include/onnxruntime"
        # pip-installed wheel (onnxruntime C API headers live in capi/)
        "$ENV{CONDA_PREFIX}/lib/python3.9/site-packages/onnxruntime/capi"
        /usr/local/include/onnxruntime
        /opt/onnxruntime/include
    NO_DEFAULT_PATH
)

find_library(ORT_LIBRARY
    NAMES onnxruntime
    HINTS
        "${ORT_ROOT}"
        "${ORT_ROOT}/lib"
        # pip-installed wheel: libonnxruntime.so lives in capi/
        "$ENV{CONDA_PREFIX}/lib/python3.9/site-packages/onnxruntime/capi"
        /usr/local/lib
        /opt/onnxruntime/lib
    NO_DEFAULT_PATH
)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime not found.\n"
        "  Set -DORT_ROOT=/path/to/onnxruntime\n"
        "  or download from https://github.com/microsoft/onnxruntime/releases\n"
        "  Expected: include/onnxruntime_cxx_api.h and lib/libonnxruntime.so"
    )
endif()

message(STATUS "ORT include : ${ORT_INCLUDE_DIR}")
message(STATUS "ORT library : ${ORT_LIBRARY}")

# ── CUDA (optional) ──────────────────────────────────────────────────────────
# We do NOT use find_package(CUDA) because nvcc may not be installed.
# Instead, locate headers and libcudart directly (works with pip conda wheel).
if(ENABLE_CUDA)
    if(NOT DEFINED CUDA_INC_DIR)
        set(CUDA_INC_DIR "$ENV{CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_runtime/include")
    endif()
    if(NOT DEFINED CUDA_CRT_INC_DIR)
        # cuda_runtime_api.h includes <crt/host_defines.h>; this subdirectory is
        # part of the cuda_nvcc wheel (not cuda_runtime). Provide it separately.
        set(CUDA_CRT_INC_DIR "$ENV{CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_nvcc/include")
    endif()
    if(NOT DEFINED CUDA_LIB_DIR)
        set(CUDA_LIB_DIR "$ENV{CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_runtime/lib")
    endif()
    find_path(CUDA_INCLUDE_DIR NAMES cuda_runtime_api.h
        HINTS "${CUDA_INC_DIR}" /usr/local/cuda/include NO_DEFAULT_PATH)
    find_path(CUDA_CRT_INCLUDE_DIR NAMES crt/host_defines.h
        HINTS "${CUDA_CRT_INC_DIR}" /usr/local/cuda/include NO_DEFAULT_PATH)
    find_library(CUDA_RT_LIBRARY NAMES cudart
        HINTS "${CUDA_LIB_DIR}" /usr/local/cuda/lib64 NO_DEFAULT_PATH)
    if(NOT CUDA_INCLUDE_DIR OR NOT CUDA_RT_LIBRARY)
        message(FATAL_ERROR
            "CUDA runtime not found.\n"
            "  Set -DCUDA_INC_DIR and -DCUDA_LIB_DIR, or install nvidia-cuda-runtime via pip.\n"
        )
    endif()
    message(STATUS "CUDA include : ${CUDA_INCLUDE_DIR}")
    if(CUDA_CRT_INCLUDE_DIR)
        message(STATUS "CUDA crt inc : ${CUDA_CRT_INCLUDE_DIR}")
    endif()
    message(STATUS "CUDA RT lib  : ${CUDA_RT_LIBRARY}")
    include_directories(${CUDA_INCLUDE_DIR})
    if(CUDA_CRT_INCLUDE_DIR)
        include_directories(${CUDA_CRT_INCLUDE_DIR})
    endif()
    add_definitions(-DENABLE_CUDA=1)
endif()

# ── TensorRT (optional) ──────────────────────────────────────────────────────
if(ENABLE_TRT)
    if(NOT ENABLE_CUDA)
        message(FATAL_ERROR "ENABLE_TRT requires ENABLE_CUDA=ON")
    endif()
    # Default TRT root = conda pip-installed tensorrt_libs wheel
    if(NOT DEFINED TRT_ROOT)
        set(TRT_ROOT "$ENV{CONDA_PREFIX}/lib/python3.9/site-packages/tensorrt_libs")
    endif()

    find_path(TRT_INCLUDE_DIR
        NAMES NvInfer.h
        HINTS
            "${TRT_ROOT}/include"
            "${TRT_ROOT}"
            /usr/local/include
            /opt/TensorRT/include
        NO_DEFAULT_PATH
    )
    find_library(TRT_LIBRARY
        NAMES nvinfer
        HINTS "${TRT_ROOT}" /usr/local/lib /opt/TensorRT/lib
        NO_DEFAULT_PATH
    )
    find_library(TRT_ONNX_LIB
        NAMES nvonnxparser
        HINTS "${TRT_ROOT}" /usr/local/lib /opt/TensorRT/lib
        NO_DEFAULT_PATH
    )

    if(NOT TRT_INCLUDE_DIR OR NOT TRT_LIBRARY)
        message(FATAL_ERROR
            "TensorRT not found.\n"
            "  Set -DTRT_ROOT=/path/to/tensorrt_libs\n"
            "  or install tensorrt-cu11 via pip (which ships the shared libs)\n"
            "  Headers should be at TRT_ROOT/include/NvInfer.h\n"
            "  (downloaded from https://github.com/NVIDIA/TensorRT)\n"
        )
    endif()
    message(STATUS "TRT include  : ${TRT_INCLUDE_DIR}")
    message(STATUS "TRT library  : ${TRT_LIBRARY}")
    message(STATUS "TRT ONNX lib : ${TRT_ONNX_LIB}")
    add_definitions(-DENABLE_TRT=1)
endif()

# ── Sanitizer ────────────────────────────────────────────────────────────────
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -g)
    add_link_options(-fsanitize=address)
endif()

# ── Shared include path ──────────────────────────────────────────────────────
include_directories(${ORT_INCLUDE_DIR})

# ── Target: ort_infer ────────────────────────────────────────────────────────
# ONNX Runtime CPU (+ optional CUDA) inference benchmark
add_executable(ort_infer onnx_runtime_infer.cpp)
target_link_libraries(ort_infer PRIVATE ${ORT_LIBRARY})
if(ENABLE_CUDA)
    target_link_libraries(ort_infer PRIVATE ${CUDA_RT_LIBRARY})
endif()
target_compile_options(ort_infer PRIVATE -O2 -Wall -Wextra)

# ── Target: trt_infer (optional) ─────────────────────────────────────────────
if(ENABLE_TRT)
    add_executable(trt_infer tensorrt_infer.cpp tensorrt_utils.cpp)
    target_include_directories(trt_infer PRIVATE
        ${TRT_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIR}
        ${ORT_INCLUDE_DIR}
    )
    target_link_libraries(trt_infer PRIVATE
        ${TRT_LIBRARY}
        ${TRT_ONNX_LIB}
        ${CUDA_RT_LIBRARY}
    )
    target_compile_options(trt_infer PRIVATE -O2 -Wall -Wextra)
    # Runtime note: export LD_LIBRARY_PATH=${TRT_ROOT}:${CUDA_LIB_DIR}:$LD_LIBRARY_PATH
endif()

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS ort_infer DESTINATION bin)
if(ENABLE_TRT)
    install(TARGETS trt_infer DESTINATION bin)
endif()

# ── Build summary ────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "planTF C++ runtime build configuration:")
message(STATUS "  CUDA provider   : ${ENABLE_CUDA}")
message(STATUS "  TensorRT target : ${ENABLE_TRT}")
message(STATUS "  ASan            : ${ENABLE_ASAN}")
message(STATUS "  ORT             : ${ORT_LIBRARY}")
message(STATUS "")

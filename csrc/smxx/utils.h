#pragma once

// Utility macros and helpers for SM-generic kernels.
// Based on FlashMLA (https://github.com/deepseek-ai/FlashMLA)
// Original: github.com/IISuperluminaLII/FlashMLA_Windows_Linux_sm120 (SM120 fork of github.com/deepseek-ai/FlashMLA): csrc/utils.h
// License: MIT, Copyright (c) 2025 DeepSeek — see THIRD_PARTY_NOTICES.md

#define CHECK_CUDA(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                              \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())

#define FLASH_ASSERT(cond)                                                                                \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                 \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define FLASH_DEVICE_ASSERT(cond)                                                                         \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            printf("DEVICE ASSERT FAILED (%s:%d): %s\n", __FILE__, __LINE__, #cond);                     \
        }                                                                                                 \
    } while(0)

template<typename T>
__inline__ __host__ __device__ T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

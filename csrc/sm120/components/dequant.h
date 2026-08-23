#pragma once

//==============================================================================
// SM120 FP8 Dequantization Components
//
// Wraps the SM90 FP8 dequantization intrinsics (which use SM80+ compatible
// instructions) and adds gather-dequant helpers for sparse attention.
//
// The core cvt_fp8x8_bf16x8() uses float32 intermediates and bf16 scaling,
// which works on any SM80+ GPU including SM120.
//==============================================================================

#include <cuda_fp8.h>
#include <cuda_bf16.h>

namespace sm120::sparse {

//==============================================================================
// FP8 data containers
//==============================================================================
struct fp8x8 {
    __nv_fp8x4_e4m3 lo;
    __nv_fp8x4_e4m3 hi;
};

struct fp8x16 {
    fp8x8 lo;
    fp8x8 hi;
};

// BF16 x8 container for dequantized output
struct bf16x8 {
    __nv_bfloat162 a01;
    __nv_bfloat162 a23;
    __nv_bfloat162 a45;
    __nv_bfloat162 a67;
};

using fp8 = __nv_fp8_e4m3;

//==============================================================================
// Core FP8 -> BF16 conversion with scaling
// Converts 8 FP8 e4m3 values to 8 BF16 values, multiplied by scale
//
// This uses SM80+ compatible operations:
//   FP8 -> FP32 (hardware cast) -> multiply by scale -> FP32 -> BF16
//==============================================================================
__device__ __forceinline__
bf16x8 cvt_fp8x8_bf16x8(const fp8x8 &inputs, const __nv_bfloat162 &scale_bf162) {
    #define SM120_DEQUANT_FP8x4(OUTPUT_BF16_LO, OUTPUT_BF16_HI, FP8x4) \
    { \
        float4 fp32x4 = (float4)(FP8x4); \
        OUTPUT_BF16_LO = __hmul2(__float22bfloat162_rn({fp32x4.x, fp32x4.y}), scale_bf162); \
        OUTPUT_BF16_HI = __hmul2(__float22bfloat162_rn({fp32x4.z, fp32x4.w}), scale_bf162); \
    }

    bf16x8 result;
    SM120_DEQUANT_FP8x4(result.a01, result.a23, inputs.lo);
    SM120_DEQUANT_FP8x4(result.a45, result.a67, inputs.hi);

    #undef SM120_DEQUANT_FP8x4
    return result;
}

//==============================================================================
// Cache-hinted global memory loads using inline PTX
// These use .nc (non-coherent / texture cache) loads with L1/L2 hints
//==============================================================================

enum class L1CacheHint {
    NO_ALLOCATE,
    EVICT_FIRST,
    EVICT_NORMAL,
    EVICT_LAST
};

enum class L2PrefetchHint {
    B64,
    B128,
    B256
};

template<
    typename T,
    L1CacheHint l1_cache_hint,
    L2PrefetchHint l2_prefetch_hint
>
__device__ __forceinline__
T load_128b_from_gmem(const void* addr) {
    static_assert(sizeof(T) == 128/8);
    int4 ret;

    #define SM120_LOAD_EXEC(L1_HINT_STR, L2_HINT_STR) { \
        asm volatile("ld.global.nc.L1::" L1_HINT_STR ".L2::" L2_HINT_STR ".v4.s32 {%0, %1, %2, %3}, [%4];" \
            : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) \
            : "l"(addr)); \
    }

    #define SM120_DISPATCH_L2(L1_HINT_STR) { \
        if constexpr(l2_prefetch_hint == L2PrefetchHint::B64) \
            SM120_LOAD_EXEC(L1_HINT_STR, "64B") \
        else if constexpr(l2_prefetch_hint == L2PrefetchHint::B128) \
            SM120_LOAD_EXEC(L1_HINT_STR, "128B") \
        else if constexpr(l2_prefetch_hint == L2PrefetchHint::B256) \
            SM120_LOAD_EXEC(L1_HINT_STR, "256B") \
    }

    if constexpr(l1_cache_hint == L1CacheHint::NO_ALLOCATE)
        SM120_DISPATCH_L2("no_allocate")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_FIRST)
        SM120_DISPATCH_L2("evict_first")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_NORMAL)
        SM120_DISPATCH_L2("evict_normal")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_LAST)
        SM120_DISPATCH_L2("evict_last")

    #undef SM120_LOAD_EXEC
    #undef SM120_DISPATCH_L2
    return *reinterpret_cast<T*>(&ret);
}

template<
    typename T,
    L1CacheHint l1_cache_hint,
    L2PrefetchHint l2_prefetch_hint
>
__device__ __forceinline__
T load_64b_from_gmem(const void* addr) {
    static_assert(sizeof(T) == 64/8);
    int2 ret;

    #define SM120_LOAD_EXEC_64(L1_HINT_STR, L2_HINT_STR) { \
        asm volatile("ld.global.nc.L1::" L1_HINT_STR ".L2::" L2_HINT_STR ".v2.s32 {%0, %1}, [%2];" \
            : "=r"(ret.x), "=r"(ret.y) \
            : "l"(addr)); \
    }

    #define SM120_DISPATCH_L2_64(L1_HINT_STR) { \
        if constexpr(l2_prefetch_hint == L2PrefetchHint::B64) \
            SM120_LOAD_EXEC_64(L1_HINT_STR, "64B") \
        else if constexpr(l2_prefetch_hint == L2PrefetchHint::B128) \
            SM120_LOAD_EXEC_64(L1_HINT_STR, "128B") \
        else if constexpr(l2_prefetch_hint == L2PrefetchHint::B256) \
            SM120_LOAD_EXEC_64(L1_HINT_STR, "256B") \
    }

    if constexpr(l1_cache_hint == L1CacheHint::NO_ALLOCATE)
        SM120_DISPATCH_L2_64("no_allocate")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_FIRST)
        SM120_DISPATCH_L2_64("evict_first")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_NORMAL)
        SM120_DISPATCH_L2_64("evict_normal")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_LAST)
        SM120_DISPATCH_L2_64("evict_last")

    #undef SM120_LOAD_EXEC_64
    #undef SM120_DISPATCH_L2_64
    return *reinterpret_cast<T*>(&ret);
}

//==============================================================================
// Gather-dequant helper: load FP8 token from global, dequant to BF16 in smem
//
// Loads 16 FP8 values (128 bits) from global memory at the given address,
// dequantizes with the provided scale, and stores 16 BF16 values to shared
// memory. Used by the producer warps in sparse decode/prefill kernels.
//==============================================================================
__device__ __forceinline__
void gather_dequant_16_to_smem(
    const fp8* __restrict__ gK_src,    // Source: global memory FP8 data
    __nv_bfloat16* __restrict__ sK_dst, // Destination: shared memory BF16
    __nv_bfloat16 scale                 // Dequantization scale
) {
    fp8x16 data = load_128b_from_gmem<fp8x16, L1CacheHint::EVICT_LAST, L2PrefetchHint::B256>(gK_src);
    __nv_bfloat162 scale2 = __bfloat162bfloat162(scale);

    bf16x8 lo = cvt_fp8x8_bf16x8(data.lo, scale2);
    bf16x8 hi = cvt_fp8x8_bf16x8(data.hi, scale2);

    // Store 16 BF16 values (2 x 128-bit stores)
    *reinterpret_cast<int4*>(sK_dst)     = *reinterpret_cast<int4*>(&lo);
    *reinterpret_cast<int4*>(sK_dst + 8) = *reinterpret_cast<int4*>(&hi);
}

}  // namespace sm120::sparse

// src/flint/flint_dec.cuh - fused single-token decode kernels of strata-flint: the per-layer chain after attention in
// three launches instead of six (wo + residual + norm; routed gate/up + SwiGLU + q8_1; routed down + the weighted
// combine + the shared expert + residual + the next norm).  The Q4_K / Q8_0 dot products are llama.cpp's
// (vecdotq.cuh: vec_dot_q4_K_q8_1 / vec_dot_q8_0_q8_1, VDR 2), as Strata transcribes them in iq_kernels.cu.
#pragma once

#include "flint_kernels.cuh"
#include "strata/kernels/dp4a.hpp"

namespace fld {

using flk::QB;
using q36k::wsum;

struct BQ4K {   // block_q4_K: 256 values
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(BQ4K) == 144, "block_q4_K");
struct BQ80 {   // block_q8_0: 32 values
    half d;
    int8_t qs[32];
};
static_assert(sizeof(BQ80) == 34, "block_q8_0");

__device__ __forceinline__ int get_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__device__ __forceinline__ int get_int_b4(const void* x, int i32) { return ((const int*) x)[i32]; }

__device__ __forceinline__ void k_scale_min(const uint8_t* scales8, int bq8_offset, uint16_t aux[2]) {
    const uint16_t* scales = (const uint16_t*) scales8;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}

/// call `iqs` (0, 2, .. 30) of Q4_K super-block `kbx` against its 8 q8_1 blocks at bq8_1 (llama.cpp's vec_dot_q4_K_q8_1)
__device__ __forceinline__ float dot_q4k(const void* __restrict__ vbq, const QB* __restrict__ bq8_1, int kbx, int iqs) {
    const BQ4K* bq = (const BQ4K*) vbq + kbx;
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* q4 = (const int*) (bq->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int v0 = q4[0], v1 = q4[4];
    uint16_t aux[2];
    k_scale_min(bq->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const QB* b8 = bq8_1 + bq8_offset + i;
        const float d8 = __low2float(b8->ds);
        const int* q8 = (const int*) b8->qs + ((iqs / 2) % 4);
        const int u0 = q8[0], u1 = q8[4];
        const int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = STRATA_DP4A(v1i, u1, STRATA_DP4A(v0i, u0, 0));
        const int dot2 = STRATA_DP4A(0x01010101, u1, STRATA_DP4A(0x01010101, u0, 0));
        sumf_d += d8 * (dot1 * sc[i]);
        sumf_m += d8 * (dot2 * m[i]);
    }
    const float2 dm = __half22float2(bq->dm);
    return dm.x * sumf_d - dm.y * sumf_m;
}
/// call `iqs` (0, 2, 4, 6) of Q8_0 block `kbx` against its q8_1 block (llama.cpp's vec_dot_q8_0_q8_1)
__device__ __forceinline__ float dot_q80(const void* __restrict__ vbq, const QB* __restrict__ bq8_1, int kbx, int iqs) {
    const BQ80* b = (const BQ80*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) sumi = STRATA_DP4A(get_int_b2(b->qs, iqs + i), get_int_b4(bq8_1->qs, iqs + i), sumi);
    return __half2float(b->d) * __low2float(bq8_1->ds) * (float) sumi;
}

/// One row of n_in values (type TY: 12 Q4_K, 8 Q8_0) against one q8_1 activation, the whole warp, the calls lane-strided.
template <int TY>
__device__ __forceinline__ float row_dot(const uint8_t* __restrict__ row, const QB* __restrict__ x, int n_in, int lane) {
    float s = 0.f;
    if (TY == 12) {
        const int n = n_in / 256 * 16;
        for (int k = lane; k < n; k += 32) s += dot_q4k(row, x + (k / 16) * 8, k / 16, 2 * (k % 16));
    } else {
        const int n = n_in / 32 * 4;
        for (int k = lane; k < n; k += 32) s += dot_q80(row, x + k / 4, k / 4, 2 * (k % 4));
    }
    return wsum(s);
}
template <int TY> __host__ __device__ constexpr size_t row_bytes(int n) { return TY == 12 ? (size_t) n / 256 * 144 : (size_t) n / 32 * 34; }

/// The norm epilogue of the last block (`ticket` counted by every block, reset by the last): x is final; h = rms(x) * w and
/// its q8_1 image hq (n = 2048, 512 or 256 threads).
__device__ __forceinline__ void norm_epilogue(float* __restrict__ x, const float* __restrict__ w, float* __restrict__ h, QB* __restrict__ hq,
                                              int n, float eps, float* red) {
    const int tid = threadIdx.x, lane = tid & 31, nw = blockDim.x >> 5;
    float s = 0.f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = __ldcg(x + i);
        s += v * v;
    }
    s = wsum(s);
    if (lane == 0) red[tid >> 5] = s;
    __syncthreads();
    if (tid < 32) {
        const float a = wsum(lane < nw ? red[lane] : 0.f);
        if (lane == 0) red[0] = a;
    }
    __syncthreads();
    const float r = rsqrtf(red[0] / (float) n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        const float hv = __ldcg(x + i) * r * w[i];
        h[i] = hv;
        if (hq) q36k::q8_store(hq, i, hv, lane);
    }
}
__device__ __forceinline__ bool last_block(unsigned int* ticket, int* s_last) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) *s_last = atomicAdd(ticket, 1u) == gridDim.x * gridDim.y - 1;
    __syncthreads();
    if (!*s_last) return false;
    __threadfence();
    return true;
}

/// y = W a (one row per warp, 8 rows per block; W of type TY, n_in values per row, a's q8_1 image), x += y, and in the
/// last block h = rms(x) * w with its q8_1 image (the output projection and the post-attention norm).  grid n_out / 8,
/// block 256.  n_out == the embedding (2048).
template <int TY>
__global__ void __launch_bounds__(256) k_gemv_res_norm(const uint8_t* __restrict__ W, const QB* __restrict__ a, int n_in,
                                                       float* __restrict__ x, const float* __restrict__ w, float* __restrict__ h,
                                                       QB* __restrict__ hq, int n, float eps, unsigned int* __restrict__ ticket) {
    __shared__ float red[32];
    __shared__ int s_last;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    const float v = row_dot<TY>(W + (size_t) r * row_bytes<TY>(n_in), a, n_in, lane);
    if (lane == 0) x[r] += v;
    if (!last_block(ticket, &s_last)) return;
    norm_epilogue(x, w, h, hq, n, eps, red);
    if (threadIdx.x == 0) *ticket = 0u;
}

/// The routed experts' gate and up rows and SwiGLU for one token: block (32-row group b, pick k) computes rows 32 b ..
/// 32 b + 31 of expert ids[k]'s gate and up (warp w: rows w and w + 16), then their SwiGLU as one q8_1 block of the
/// pick's hidden (hq: K x FF / 32 blocks).  grid (FF / 32, K), block 512.
template <int TG>
__global__ void __launch_bounds__(512) k_exp_gu(const int32_t* __restrict__ ids, const uint8_t* __restrict__ blobs, size_t stride,
                                                size_t up_off, const QB* __restrict__ xq, int n_embd, int FF, QB* __restrict__ hq) {
    __shared__ float g[32], u[32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int k = blockIdx.y, b = blockIdx.x;
    const uint8_t* blob = blobs + (size_t) ids[k] * stride;
    const size_t rb = row_bytes<TG>(n_embd);
    const int ra = b * 32 + warp, rc = ra + 16;
    const float ga = row_dot<TG>(blob + (size_t) ra * rb, xq, n_embd, lane);
    const float gc = row_dot<TG>(blob + (size_t) rc * rb, xq, n_embd, lane);
    const float ua = row_dot<TG>(blob + up_off + (size_t) ra * rb, xq, n_embd, lane);
    const float uc = row_dot<TG>(blob + up_off + (size_t) rc * rb, xq, n_embd, lane);
    if (lane == 0) {
        g[warp] = ga; g[warp + 16] = gc;
        u[warp] = ua; u[warp + 16] = uc;
    }
    __syncthreads();
    if (warp == 0) {
        const float gi = g[lane];
        const float hv = gi / (1.0f + __expf(-gi)) * u[lane];
        q36k::q8_store(hq + (size_t) k * (FF / 32) + b, lane, hv, lane);
    }
}

/// The routed experts' down rows for one token, their weighted sum (pick order), the shared expert's output and the
/// residual: x[r] += sum_k w_k down_k[r] . h_k + shared[r] (warp per output row, 16 rows per block); the last block then
/// h = rms(x) * wn and its q8_1 image (the next layer's input).  grid N / 16, block 512.
template <int TD>
__global__ void __launch_bounds__(512) k_exp_down(const int32_t* __restrict__ ids, const float* __restrict__ wts, int K_,
                                                  const uint8_t* __restrict__ blobs, size_t stride, size_t down_off,
                                                  const QB* __restrict__ hq, int FF, const float* __restrict__ shared,
                                                  float* __restrict__ x, const float* __restrict__ wn, float* __restrict__ h,
                                                  QB* __restrict__ xq, int n, float eps, unsigned int* __restrict__ ticket) {
    __shared__ float red[32];
    __shared__ int s_last;
    __shared__ const uint8_t* rows0[16];
    __shared__ float sw[16];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (threadIdx.x < K_) {
        rows0[threadIdx.x] = blobs + (size_t) ids[threadIdx.x] * stride + down_off;
        sw[threadIdx.x] = wts[threadIdx.x];
    }
    __syncthreads();
    const int r = blockIdx.x * 16 + warp;
    const size_t rb = row_bytes<TD>(FF);
    float acc = 0.f;
    for (int k = 0; k < K_; ++k) {
        const float d = row_dot<TD>(rows0[k] + (size_t) r * rb, hq + (size_t) k * (FF / 32), FF, lane);
        acc = k == 0 ? sw[0] * d : fmaf(sw[k], d, acc);
    }
    if (lane == 0) x[r] += acc + shared[r];
    if (!last_block(ticket, &s_last)) return;
    norm_epilogue(x, wn, h, xq, n, eps, red);
    if (threadIdx.x == 0) *ticket = 0u;
}

/// k_exp_gu over the grouped entries (one token: K groups of one entry each, ascending expert id): block (32-row group
/// b, group g) reads expert grp_ptr[g]'s blob and writes the hidden's q8_1 block at the group's position, where
/// native_expert_down_q8 reads it.  grid (FF / 32, K), block 512.
template <int TG>
__global__ void __launch_bounds__(512) k_exp_gu_grp(const unsigned long long* __restrict__ grp_ptr, size_t up_off,
                                                    const QB* __restrict__ xq, int n_embd, int FF, QB* __restrict__ hq) {
    __shared__ float g[32], u[32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int k = blockIdx.y, b = blockIdx.x;
    const uint8_t* blob = (const uint8_t*) grp_ptr[k];
    const size_t rb = row_bytes<TG>(n_embd);
    const int ra = b * 32 + warp, rc = ra + 16;
    const float ga = row_dot<TG>(blob + (size_t) ra * rb, xq, n_embd, lane);
    const float gc = row_dot<TG>(blob + (size_t) rc * rb, xq, n_embd, lane);
    const float ua = row_dot<TG>(blob + up_off + (size_t) ra * rb, xq, n_embd, lane);
    const float uc = row_dot<TG>(blob + up_off + (size_t) rc * rb, xq, n_embd, lane);
    if (lane == 0) {
        g[warp] = ga; g[warp + 16] = gc;
        u[warp] = ua; u[warp + 16] = uc;
    }
    __syncthreads();
    if (warp == 0) {
        const float gi = g[lane];
        const float hv = gi / (1.0f + __expf(-gi)) * u[lane];
        q36k::q8_store(hq + (size_t) k * (FF / 32) + b, lane, hv, lane);
    }
}

}  // namespace fld

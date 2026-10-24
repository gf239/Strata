// src/flint/flint_fa.cuh - the prompt's attention for strata-flint as one flash-attention kernel (FP16 tensor cores,
// mma.sync m16n8k16, FP32 accumulation and online softmax), in place of the scores / softmax / values passes over a
// materialised score matrix.  Flint's specifics: 8 query heads per KV head share every K / V tile (a block's 64 rows are
// 8 queries x 8 heads), the causal and 513-cell window masks per row, and the head's learned sink logit folded into the
// softmax denominator at the end.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ffa {

constexpr int kD = 128;       // head dim
constexpr int kG = 8;         // query heads per KV head
constexpr int kBr = 64;       // rows per block: 8 queries x 8 heads, 16 per warp
constexpr int kBc = 64;       // keys per tile
constexpr int kStride = kD + 8;   // shared rows 272 bytes apart: ldmatrix's 8 rows hit distinct banks

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

/// att[t][h][d] (f32) for the chunk's T queries at positions p0 .. p0 + T - 1 (q16: [T][H][128] FP16, normed and roped),
/// keys from K / V [kvh][kstride][128] FP16 whose row j is cell lo + j (the full layers: the cache itself, lo 0; the
/// window layers: the gathered window).  Row r of KV head g's rows is query r / 8, head 8 g + r % 8.  grid
/// (ceil(T / 8), HKV), block 128.
__global__ void __launch_bounds__(128) k_fa_prompt(const __half* __restrict__ q16, const __half* __restrict__ K,
                                                   const __half* __restrict__ V, int kstride, int lo, int T, int H, int p0,
                                                   int win, float scale, const float* __restrict__ sinks,
                                                   float* __restrict__ att) {
    __shared__ __align__(16) __half ks[kBc][kStride];
    __shared__ __align__(16) __half vs[kBc][kStride];
    const int g = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int t0 = blockIdx.x * (kBr / kG);   // the block's first query
    // this warp's rows: queries t0 + 2 warp (rows 0..7: heads 0..7) and t0 + 2 warp + 1 (rows 8..15)
    const int tw = t0 + 2 * warp;
    const int qa = tw, qb = tw + 1;           // a thread's two rows: (qa, head gid) and (qb, head gid)
    const int posa = p0 + qa, posb = p0 + qb;
    const __half* Kg = K + (size_t) g * kstride * kD;
    const __half* Vg = V + (size_t) g * kstride * kD;
    // Q fragments (A, 16 x 128 as 8 k-steps) straight from global: rows (q, head) of this warp
    uint32_t qf[8][4];
    {
        const int tq = min(qa, T - 1), tq2 = min(qb, T - 1);
        const __half* qra = q16 + ((size_t) tq * H + g * kG + gid) * kD;
        const __half* qrb = q16 + ((size_t) tq2 * H + g * kG + gid) * kD;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            qf[k][0] = *reinterpret_cast<const uint32_t*>(qra + 16 * k + 2 * tig);
            qf[k][1] = *reinterpret_cast<const uint32_t*>(qrb + 16 * k + 2 * tig);
            qf[k][2] = *reinterpret_cast<const uint32_t*>(qra + 16 * k + 8 + 2 * tig);
            qf[k][3] = *reinterpret_cast<const uint32_t*>(qrb + 16 * k + 8 + 2 * tig);
        }
    }
    constexpr float kLog2e = 1.4426950408889634f;
    const float sl2 = scale * kLog2e;
    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
    // the block's cells: from the first query's window start (or 0) to the last query's own cell
    const int tlast = min(t0 + kBr / kG - 1, T - 1);
    const int cbeg = win > 0 ? max(lo, p0 + t0 - (win - 1)) : lo;
    const int cend = p0 + tlast;   // inclusive
    const int loa = win > 0 ? posa - (win - 1) : 0, lob = win > 0 ? posb - (win - 1) : 0;
    for (int c0 = cbeg; c0 <= cend; c0 += kBc) {
        __syncthreads();
        // K / V tile: 64 rows x 16 uint4 each, 128 threads -> 8 each
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int i = threadIdx.x + r * 128;
            const int j = i >> 4, d8 = i & 15;
            const int c = c0 + j;
            uint4 k4 = make_uint4(0u, 0u, 0u, 0u), v4 = k4;
            if (c <= cend) {
                k4 = __ldg(reinterpret_cast<const uint4*>(Kg + (size_t) (c - lo) * kD) + d8);
                v4 = __ldg(reinterpret_cast<const uint4*>(Vg + (size_t) (c - lo) * kD) + d8);
            }
            *reinterpret_cast<uint4*>(&ks[j][d8 * 8]) = k4;
            *reinterpret_cast<uint4*>(&vs[j][d8 * 8]) = v4;
        }
        __syncthreads();
        // S = Q K^T: 16 rows x 64 keys (8 n-tiles of 8)
        float s[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
#pragma unroll
            for (int n2 = 0; n2 < 4; ++n2) {   // two n-tiles per ldmatrix.x4
                uint32_t b[4];
                const int key = n2 * 16 + (lane & 7) + 8 * (lane >> 4);
                const int d = k * 16 + 8 * ((lane >> 3) & 1);
                ldsm_x4(b, &ks[key][d]);
                mma16816(s[2 * n2], qf[k], b[0], b[1]);
                mma16816(s[2 * n2 + 1], qf[k], b[2], b[3]);
            }
        }
        // masks and the online softmax (log2 domain); a thread holds row a's keys 2 tig, 2 tig + 1 of each n-tile in
        // s[n][0..1] and row b's in s[n][2..3]
        float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int c = c0 + n * 8 + 2 * tig + e;
                const bool va = c <= posa && c >= loa && qa < T;
                const bool vb = c <= posb && c >= lob && qb < T;
                s[n][e] = va ? s[n][e] * sl2 : -INFINITY;
                s[n][2 + e] = vb ? s[n][2 + e] * sl2 : -INFINITY;
                mxa = fmaxf(mxa, s[n][e]);
                mxb = fmaxf(mxb, s[n][2 + e]);
            }
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float mna = fmaxf(ma, mxa), mnb = fmaxf(mb, mxb);
        // a row with no cell yet (all -inf) keeps -inf; exp2(-inf - -inf) is guarded
        const float ca = mna == -INFINITY ? 1.f : exp2f(ma - mna), cb = mnb == -INFINITY ? 1.f : exp2f(mb - mnb);
        float sa = 0.f, sb = 0.f;
        uint32_t pf[4][4];   // P as A fragments: 4 k-steps of 16 keys
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const float p0a = mna == -INFINITY ? 0.f : exp2f(s[n][0] - mna), p1a = mna == -INFINITY ? 0.f : exp2f(s[n][1] - mna);
            const float p0b = mnb == -INFINITY ? 0.f : exp2f(s[n][2] - mnb), p1b = mnb == -INFINITY ? 0.f : exp2f(s[n][3] - mnb);
            sa += p0a + p1a;
            sb += p0b + p1b;
            const int kk = n >> 1;
            if ((n & 1) == 0) {
                pf[kk][0] = pack_h2(p0a, p1a);
                pf[kk][1] = pack_h2(p0b, p1b);
            } else {
                pf[kk][2] = pack_h2(p0a, p1a);
                pf[kk][3] = pack_h2(p0b, p1b);
            }
        }
        sa += __shfl_xor_sync(0xffffffffu, sa, 1);
        sa += __shfl_xor_sync(0xffffffffu, sa, 2);
        sb += __shfl_xor_sync(0xffffffffu, sb, 1);
        sb += __shfl_xor_sync(0xffffffffu, sb, 2);
        la = la * ca + sa;
        lb = lb * cb + sb;
        ma = mna;
        mb = mnb;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            o[j][0] *= ca; o[j][1] *= ca;
            o[j][2] *= cb; o[j][3] *= cb;
        }
        // O += P V: V as B (k = keys, n = d) through ldmatrix.trans, two d-tiles per load
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int j2 = 0; j2 < 8; ++j2) {
                uint32_t b[4];
                const int key = kk * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);
                const int d = j2 * 16 + 8 * (lane >> 4);
                ldsm_x4_t(b, &vs[key][d]);
                mma16816(o[2 * j2], pf[kk], b[0], b[1]);
                mma16816(o[2 * j2 + 1], pf[kk], b[2], b[3]);
            }
        }
    }
    // the sink: one more logit in each row's denominator, no value
    const float sk = sinks[g * kG + gid] * kLog2e;
    const float mfa = fmaxf(ma, sk), mfb = fmaxf(mb, sk);
    const float fa = ma == -INFINITY ? 0.f : exp2f(ma - mfa), fb = mb == -INFINITY ? 0.f : exp2f(mb - mfb);
    const float ia = 1.0f / (la * fa + exp2f(sk - mfa)), ib = 1.0f / (lb * fb + exp2f(sk - mfb));
    const int h = g * kG + gid;
    if (qa < T) {
        float* dst = att + ((size_t) qa * H + h) * kD;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            *reinterpret_cast<float2*>(dst + j * 8 + 2 * tig) = make_float2(o[j][0] * fa * ia, o[j][1] * fa * ia);
    }
    if (qb < T) {
        float* dst = att + ((size_t) qb * H + h) * kD;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            *reinterpret_cast<float2*>(dst + j * 8 + 2 * tig) = make_float2(o[j][2] * fb * ib, o[j][3] * fb * ib);
    }
}

}  // namespace ffa

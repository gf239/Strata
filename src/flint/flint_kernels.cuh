// src/flint/flint_kernels.cuh - the Flint-specific kernels of strata-flint (src/flint/flint_engine.cu): attention with
// per-head QK RMSNorm, learned sinks, a sliding window and RoPE only on the window layers; the sigmoid top-k router;
// the MoE end with the ungated shared expert.  The q8_1 helpers are the Qwen3.6 engine's (q36_kernels.cuh).
#pragma once

// q36_kernels.cuh defines its kernels in a header: included here with internal linkage, so `strata` (which links the
// Qwen3.6 path's copy too) gets two private copies instead of two definitions of one symbol.  Its own includes first,
// outside the anonymous namespace.
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
namespace {
#include "../qwen35/q36_kernels.cuh"
}  // namespace

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace flk {

using q36k::QB;
using q36k::wmax;
using q36k::wsum;

constexpr int kHD = 128;    // head dim (the kernels are written for it)
constexpr int kGrp = 8;     // query heads per KV head
constexpr int kTile = 32;   // cells per decode-attention tile (one per lane)
constexpr int kKS = kHD + 2;   // K rows in shared memory padded by one word: lane j reading row j hits bank (j + d) % 32

/// The attention layer's per-token preparation for T rows at positions *pos0 + t: q, k and v of row t at qs + t * ldq,
/// ks + t * ldk, vs + t * ldv; q and k RMS-normed per head with their weights, NEOX RoPE over the whole head on the
/// rope layers, q into `q` (f32 [T][H][HD]), k / v into the layer's fp16 cache at their cell (pos % cap for a ring).
/// grid (H + HKV, T), block 128.
__global__ void __launch_bounds__(kHD) k_prep(const float* __restrict__ qs, int ldq, const float* __restrict__ ks, int ldk,
                                              const float* __restrict__ vs, int ldv, const float* __restrict__ qw,
                                              const float* __restrict__ kw, float* __restrict__ q, __half* __restrict__ kc,
                                              __half* __restrict__ vc, int H, int cap, int ring, int rope,
                                              const int32_t* __restrict__ pos0, float base, float eps) {
    __shared__ float buf[kHD];
    __shared__ float red[kHD / 32];
    const int head = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int pos = *pos0 + t;
    const bool isq = head < H;
    const int kh = head - H;
    const float* src = isq ? qs + (size_t) t * ldq + (size_t) head * kHD : ks + (size_t) t * ldk + (size_t) kh * kHD;
    const float v0 = src[i];
    float s = wsum(v0 * v0);
    if ((i & 31) == 0) red[i >> 5] = s;
    __syncthreads();
    const float ss = red[0] + red[1] + red[2] + red[3];
    const float r = rsqrtf(ss / (float) kHD + eps);
    float v = v0 * r * (isq ? qw[i] : kw[i]);
    if (rope) {
        buf[i] = v;
        __syncthreads();
        constexpr int half = kHD / 2;
        const int p = i < half ? i : i - half;
        const float theta = (float) pos * powf(base, -2.0f * (float) p / (float) kHD);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    if (isq) {
        q[((size_t) t * H + head) * kHD + i] = v;
        return;
    }
    const int cell = ring ? pos % cap : pos;
    kc[((size_t) kh * cap + cell) * kHD + i] = __float2half(v);
    vc[((size_t) kh * cap + cell) * kHD + i] = __float2half(vs[(size_t) t * ldv + (size_t) kh * kHD + i]);
}

/// Decode attention partials.  Block (split s, KV head g, row t) takes its share of the cells row t sees - [pos - (win
/// - 1), pos] on a window layer (win > 0), [0, pos] otherwise - in tiles of 32 cells (K and V in shared memory), the 8
/// query heads of g one warp each with an online softmax.  Per (row, head, split): the unnormalised accumulator, its
/// max and its sum (a split without cells: max -inf, sum 0).  grid (NS, HKV, T), block 256.
__global__ void __launch_bounds__(256) k_dattn(const float* __restrict__ q, const __half* __restrict__ kc,
                                               const __half* __restrict__ vc, int H, int cap, int ring, int win,
                                               const int32_t* __restrict__ pos0, float scale, int NS,
                                               float* __restrict__ pacc, float* __restrict__ pm, float* __restrict__ pl) {
    __shared__ __align__(16) __half ks[kTile][kKS];
    __shared__ __align__(16) __half vs[kTile][kHD];
    __shared__ float qs[kGrp][kHD];
    const int s = blockIdx.x, g = blockIdx.y, t = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = g * kGrp + warp;
    const int pos = *pos0 + t;
    const int lo = win > 0 ? max(0, pos - (win - 1)) : 0;
    const int n = pos - lo + 1;
    int chunk = (n + NS - 1) / NS;
    chunk = (chunk + kTile - 1) / kTile * kTile;
    const int c0 = lo + s * chunk, c1 = min(pos + 1, c0 + chunk);
    for (int i = threadIdx.x; i < kGrp * kHD; i += blockDim.x)
        qs[i / kHD][i % kHD] = q[((size_t) t * H + g * kGrp) * kHD + i];
    float m = -INFINITY, l = 0.f;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const __half* kb = kc + (size_t) g * cap * kHD;
    const __half* vb = vc + (size_t) g * cap * kHD;
    for (int b0 = c0; b0 < c1; b0 += kTile) {
        __syncthreads();   // the previous tile consumed (the first time: qs written)
        for (int i = threadIdx.x; i < kTile * (kHD / 2); i += blockDim.x) {
            const int j = i / (kHD / 2), d2 = i % (kHD / 2);
            const int c = b0 + j;
            half2 k2 = __floats2half2_rn(0.f, 0.f), v2 = k2;
            if (c < c1) {
                const size_t cell = (size_t) (ring ? c % cap : c);
                k2 = reinterpret_cast<const half2*>(kb + cell * kHD)[d2];
                v2 = reinterpret_cast<const half2*>(vb + cell * kHD)[d2];
            }
            reinterpret_cast<half2*>(&ks[j][0])[d2] = k2;
            reinterpret_cast<half2*>(&vs[j][0])[d2] = v2;
        }
        __syncthreads();
        const int c = b0 + lane;
        float sc = -INFINITY;
        if (c < c1) {
            const half2* kr = reinterpret_cast<const half2*>(&ks[lane][0]);
            const float* qh = qs[warp];
            float acc = 0.f;
#pragma unroll 16
            for (int d2 = 0; d2 < kHD / 2; ++d2) {
                const float2 kf = __half22float2(kr[d2]);
                acc = fmaf(qh[2 * d2], kf.x, acc);
                acc = fmaf(qh[2 * d2 + 1], kf.y, acc);
            }
            sc = acc * scale;
        }
        const float mn = fmaxf(m, wmax(sc));
        const float corr = expf(m - mn);
        const float p = c < c1 ? expf(sc - mn) : 0.f;
        l = l * corr + wsum(p);
        a0 *= corr; a1 *= corr; a2 *= corr; a3 *= corr;
        m = mn;
        const int nj = min(kTile, c1 - b0);
        for (int j = 0; j < nj; ++j) {
            const float pj = __shfl_sync(0xffffffffu, p, j);
            const half2* vr = reinterpret_cast<const half2*>(&vs[j][lane * 4]);
            const float2 v01 = __half22float2(vr[0]), v23 = __half22float2(vr[1]);
            a0 = fmaf(pj, v01.x, a0);
            a1 = fmaf(pj, v01.y, a1);
            a2 = fmaf(pj, v23.x, a2);
            a3 = fmaf(pj, v23.y, a3);
        }
    }
    const size_t o = ((size_t) t * H + h) * NS + s;
    float4* dst = reinterpret_cast<float4*>(pacc + o * kHD) + lane;
    *dst = make_float4(a0, a1, a2, a3);
    if (lane == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}

/// k_dattn's splits merged, the head's sink logit in the softmax denominator: att[t][h][d] (row t * H + h); with `aq`
/// the q8_1 image of att (the output projection's input) as well.  grid (T * H), block 128.
__global__ void __launch_bounds__(kHD) k_dattn_merge(const float* __restrict__ pacc, const float* __restrict__ pm,
                                                     const float* __restrict__ pl, const float* __restrict__ sinks, int H,
                                                     int NS, float* __restrict__ att, QB* __restrict__ aq) {
    const int row = blockIdx.x, h = row % H, d = threadIdx.x;
    const size_t o = (size_t) row * NS;
    const float sk = sinks[h];
    float M = sk;
    for (int s = 0; s < NS; ++s) M = fmaxf(M, pm[o + s]);
    float num = 0.f, den = expf(sk - M);
    for (int s = 0; s < NS; ++s) {
        const float w = expf(pm[o + s] - M);
        num = fmaf(w, pacc[(o + s) * kHD + d], num);
        den = fmaf(w, pl[o + s], den);
    }
    const float v = num / den;
    att[(size_t) row * kHD + d] = v;
    if (aq) q36k::q8_store(aq + (size_t) row * (kHD / 32), d, v, d & 31);
}

/// The prompt's window-layer K / V: cells [lo, lo + P) of the ring (cap cells) into compact [kvh][P][HD] halves.
/// grid1(HKV * P * HD / 2) over half2.
__global__ void k_kv_gather(const __half* __restrict__ kc, const __half* __restrict__ vc, __half* __restrict__ k16,
                            __half* __restrict__ v16, int cap, int lo, int P, int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int H2 = kHD / 2;
    if (i >= (int64_t) HKV * P * H2) return;
    const int64_t per = (int64_t) P * H2;
    const int kvh = (int) (i / per);
    const int64_t rest = i % per;
    const int j = (int) (rest / H2), d2 = (int) (rest % H2);
    const size_t src = ((size_t) kvh * cap + (size_t) ((lo + j) % cap)) * H2 + d2;
    reinterpret_cast<half2*>(k16)[i] = reinterpret_cast<const half2*>(kc)[src];
    reinterpret_cast<half2*>(v16)[i] = reinterpret_cast<const half2*>(vc)[src];
}

/// The prompt's softmax with the sink: row (h, t) of S (ld floats, P cells from absolute cell lo) -> FP16 probabilities
/// over the cells query p0 + t sees (c <= p0 + t, and c > p0 + t - win on a window layer), 0 elsewhere; the sink logit
/// takes its share of the denominator.  grid (T, H), block 256.
__global__ void k_softmax_sink(const float* __restrict__ S, __half* __restrict__ S16, int ld, int T, int P, int lo, int p0,
                               int win, float scale, const float* __restrict__ sinks) {
    __shared__ float sh[256];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int qp = p0 + t;
    const int jmax = min(P, qp - lo + 1);                       // cells [0, jmax) are <= qp
    const int jmin = win > 0 ? max(0, qp - (win - 1) - lo) : 0;  // and >= qp - (win - 1)
    const float sk = sinks[h];
    float mx = sk;
    for (int j = jmin + (int) threadIdx.x; j < jmax; j += blockDim.x) mx = fmaxf(mx, row[j] * scale);
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    mx = sh[0];
    __syncthreads();
    float sum = 0.f;
    for (int j = jmin + (int) threadIdx.x; j < jmax; j += blockDim.x) sum += expf(row[j] * scale - mx);
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.0f / (sh[0] + expf(sk - mx));
    for (int j = threadIdx.x; j < P; j += blockDim.x)
        out[j] = __float2half(j >= jmin && j < jmax ? expf(row[j] * scale - mx) * inv : 0.f);
}

/// One token's sigmoid top-K over NE = 256 router logits (one warp): ids in pick order, the picked sigmoids renormalised
/// to sum 1 (llama.cpp's flint: the sum clamped at 6.1e-5).  Ties go to the lower id.
template <int K>
__device__ __forceinline__ void route_sig_warp(const float* __restrict__ lg, int lane, int32_t* __restrict__ ids,
                                               float* __restrict__ weights, int* __restrict__ sids) {
    constexpr int NE = 256, V = NE / 32;
    float values[V];
#pragma unroll
    for (int i = 0; i < V; ++i) values[i] = 1.0f / (1.0f + expf(-lg[lane + i * 32]));
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < K; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < V; ++i)
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            if (sids) sids[rank] = expert;
        }
        if (lane == 0) selected_sum += best;
        if (rank == lane) selected = best;
    }
    selected_sum = fmaxf(__shfl_sync(0xffffffffu, selected_sum, 0), 6.103515625e-5f);
    if (lane < K) weights[lane] = selected / selected_sum;
}

/// The router's top-K for T <= 8 tokens (one warp each) and, in the same launch, the grouping of the picks per expert
/// for native_expert_grouped (q36k::k_route_group's grouping).  block (32, 8).
template <int K>
__global__ void __launch_bounds__(256) k_route_group(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                                     float* __restrict__ weights, int n_tok, int ld,
                                                     const uint8_t* blobs, unsigned long long blob,
                                                     unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                                                     int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int sids[64], sid[64];
    const int tk = threadIdx.y, lane = threadIdx.x;
    if (tk < n_tok) route_sig_warp<K>(logits + (size_t) tk * ld, lane, ids + (size_t) tk * K, weights + (size_t) tk * K,
                                      sids + tk * K);
    __syncthreads();
    const int n_ent = n_tok * K;
    const int e = tk * 32 + lane;
    const int id = e < n_ent ? sids[e] : 0;
    const bool valid = e < n_ent;
    int rank = 0;
    if (valid) {
        for (int e2 = 0; e2 < n_ent; ++e2) {
            const int id2 = sids[e2];
            if (id2 < id || (id2 == id && e2 < e)) ++rank;
        }
        ent_dst[rank] = e;
        ent_tok[rank] = e / K;
        sid[rank] = id;
    }
    __syncthreads();
    if (valid && (rank == 0 || sid[rank - 1] != sid[rank])) {
        int g = 0;
        for (int qq = 1; qq < rank; ++qq) g += sid[qq - 1] != sid[qq] ? 1 : 0;
        g += rank > 0 ? 1 : 0;
        grp_ptr[g] = (unsigned long long) (blobs + (size_t) id * blob);
        grp_start[g] = rank;
    }
    if (e == 0) {
        int G = n_ent > 0 ? 1 : 0;
        for (int qq = 1; qq < n_ent; ++qq) G += sid[qq - 1] != sid[qq] ? 1 : 0;
        grp_start[G] = n_ent;
        *n_groups = G;
    }
}

/// The router's top-K for a prompt chunk: one warp per token.  grid (ceil(T / 8)), block (32, 8).
template <int K>
__global__ void __launch_bounds__(256) k_route(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                               float* __restrict__ weights, int n_tok, int ld) {
    const int tk = blockIdx.x * 8 + threadIdx.y;
    if (tk >= n_tok) return;
    route_sig_warp<K>(logits + (size_t) tk * ld, threadIdx.x, ids + (size_t) tk * K, weights + (size_t) tk * K, nullptr);
}

/// The MoE block's end for one row (block per row): x += sum_k w_k part_k (pick order) + shared; then, with `wn`,
/// h = rms(x) * wn (the next layer's norm) and with `hq` its q8_1 image.  T <= 8 rows of n <= 2048 (n % 256 == 0):
/// 1024 threads.
__global__ void __launch_bounds__(1024) k_moe_out_small(float* x, const float* parts, const int32_t* prow, const float* wts,
                                                        int K_, const float* shared, const float* wn, float* h, int n,
                                                        float eps, QB* hq) {
    __shared__ float vv[2048];
    __shared__ float sh[256];
    __shared__ const float* pk[16];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    if ((int) threadIdx.x < K_) pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    for (int i = threadIdx.x; i < n; i += 1024) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k) acc = fmaf(wr[k], pk[k][i], acc);
        acc += shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        vv[i] = v;
    }
    if (wn == nullptr) return;
    __syncthreads();
    if (threadIdx.x < 256) {
        float s = 0.f;
        for (int i = threadIdx.x; i < n; i += 256) s += vv[i] * vv[i];
        sh[threadIdx.x] = s;
    }
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float hv = vv[i] * r * wn[i];
        h[(size_t) t * n + i] = hv;
        if (hq) q36k::q8_store(hq + (size_t) t * (n >> 5), i, hv, threadIdx.x & 31);
    }
}

/// The same for a prompt chunk (any T, block of 256 per row, no q8_1 image).
__global__ void __launch_bounds__(256) k_moe_out(float* x, const float* parts, const int32_t* prow, const float* wts, int K_,
                                                 const float* shared, const float* wn, float* h, int n, float eps) {
    __shared__ float sh[256];
    __shared__ const float* pk[16];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    if ((int) threadIdx.x < K_) pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k) acc = fmaf(wr[k], pk[k][i], acc);
        acc += shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        s += v * v;
    }
    if (wn == nullptr) return;
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) h[(size_t) t * n + i] = xr[i] * r * wn[i];
}

}  // namespace flk

// ------------------------------------------------------------------------------------------------ fused decode kernels
namespace flk {

/// The q8_1 block of 32 values held 4 per lane by 8 consecutive lanes (lanes 8b .. 8b + 7 hold block b's values
/// 4 (lane % 8) .. + 3): the same scale and sum as native_quantize_q8_1 (d = amax / 127, kept finite).
__device__ __forceinline__ void q8_store4(QB* __restrict__ blk, float4 v, int lane) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    float sum = (v.x + v.y) + (v.z + v.w);
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = strata::kernels::q8_1_finite(amax / 127.0f);
    char4 q;
    q.x = strata::kernels::q8_1_quant(v.x, d, amax);
    q.y = strata::kernels::q8_1_quant(v.y, d, amax);
    q.z = strata::kernels::q8_1_quant(v.z, d, amax);
    q.w = strata::kernels::q8_1_quant(v.w, d, amax);
    reinterpret_cast<char4*>(blk->qs)[lane & 7] = q;
    if ((lane & 7) == 0) blk->ds = strata::kernels::q8_1_ds(d, sum);
}

/// One head of 128 values held 4 per lane (lane l: dims 4l .. 4l + 3), RMS-normed with `w` and, with rope, NEOX-rotated
/// at position pos (pair p = d % 64: dims p and p + 64 sit in lanes l and l ^ 16).
__device__ __forceinline__ float4 head_norm_rope(float4 v, const float* __restrict__ w, int lane, bool rope, int pos,
                                                 float base, float eps) {
    const float ss = wsum((v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w));
    const float r = rsqrtf(ss / (float) kHD + eps);
    const float4 wv = reinterpret_cast<const float4*>(w)[lane];
    v.x = v.x * r * wv.x; v.y = v.y * r * wv.y; v.z = v.z * r * wv.z; v.w = v.w * r * wv.w;
    if (!rope) return v;
    const float4 o = make_float4(__shfl_xor_sync(0xffffffffu, v.x, 16), __shfl_xor_sync(0xffffffffu, v.y, 16),
                                 __shfl_xor_sync(0xffffffffu, v.z, 16), __shfl_xor_sync(0xffffffffu, v.w, 16));
    const bool lo = lane < 16;
    const int p0 = 4 * (lane & 15);
    float a[4] = {v.x, v.y, v.z, v.w}, b[4] = {o.x, o.y, o.z, o.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float theta = (float) pos * powf(base, -2.0f * (float) (p0 + k) / (float) kHD);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        a[k] = lo ? a[k] * cs - b[k] * sn : b[k] * sn + a[k] * cs;
    }
    return make_float4(a[0], a[1], a[2], a[3]);
}

/// Decode attention for T <= 8 rows in ONE launch: k_prep + k_dattn + k_dattn_merge.  Block (split s, KV head g, row t):
/// its 8 query heads normed / roped from the qkv row itself; the block whose share holds the row's own cell also norms /
/// ropes k and writes k / v into the cache before its tiles read them; then the split's online softmax; the last block of
/// (t, g) to finish (a ticket per (t, g), reset by it) merges the splits that had cells with the sink logit and writes att
/// and its q8_1 image.  grid (NS, HKV, T), block 256.
__global__ void __launch_bounds__(256) k_attn_dec(const float* __restrict__ qs_, int ldq, const float* __restrict__ ks_, int ldk,
                                                  const float* __restrict__ vs_, int ldv, const float* __restrict__ qw,
                                                  const float* __restrict__ kw, __half* __restrict__ kc, __half* __restrict__ vc,
                                                  int H, int cap, int ring, int win, int rope, const int32_t* __restrict__ pos0,
                                                  float base, float eps, float scale, int NS, float* __restrict__ pacc,
                                                  float* __restrict__ pm, float* __restrict__ pl, const float* __restrict__ sinks,
                                                  float* __restrict__ att, QB* __restrict__ aq, unsigned int* __restrict__ ticket) {
    __shared__ __align__(16) __half ks[kTile][kKS];
    __shared__ __align__(16) __half vs[kTile][kHD];
    __shared__ __align__(16) float qs[kGrp][kHD];
    __shared__ int s_last;
    const int s = blockIdx.x, g = blockIdx.y, t = blockIdx.z, HKV = gridDim.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = g * kGrp + warp;
    const int pos = *pos0 + t;
    const int lo = win > 0 ? max(0, pos - (win - 1)) : 0;
    const int n = pos - lo + 1;
    int chunk = (n + NS - 1) / NS;
    chunk = (chunk + kTile - 1) / kTile * kTile;
    const int c0 = lo + s * chunk, c1 = min(pos + 1, c0 + chunk);
    {
        const float4 v = reinterpret_cast<const float4*>(qs_ + (size_t) t * ldq + (size_t) h * kHD)[lane];
        reinterpret_cast<float4*>(qs[warp])[lane] = head_norm_rope(v, qw, lane, rope != 0, pos, base, eps);
    }
    if (c0 <= pos && pos < c1) {   // the share that holds the row's own cell writes its k / v first
        const size_t cell = (size_t) (ring ? pos % cap : pos);
        if (warp == 0) {
            const float4 v = reinterpret_cast<const float4*>(ks_ + (size_t) t * ldk + (size_t) g * kHD)[lane];
            const float4 k = head_norm_rope(v, kw, lane, rope != 0, pos, base, eps);
            half2* dst = reinterpret_cast<half2*>(kc + ((size_t) g * cap + cell) * kHD) + 2 * lane;
            dst[0] = __floats2half2_rn(k.x, k.y);
            dst[1] = __floats2half2_rn(k.z, k.w);
        } else if (warp == 1) {
            const float4 v = reinterpret_cast<const float4*>(vs_ + (size_t) t * ldv + (size_t) g * kHD)[lane];
            half2* dst = reinterpret_cast<half2*>(vc + ((size_t) g * cap + cell) * kHD) + 2 * lane;
            dst[0] = __floats2half2_rn(v.x, v.y);
            dst[1] = __floats2half2_rn(v.z, v.w);
        }
    }
    float m = -INFINITY, l = 0.f;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const __half* kb = kc + (size_t) g * cap * kHD;
    const __half* vb = vc + (size_t) g * cap * kHD;
    for (int b0 = c0; b0 < c1; b0 += kTile) {
        __syncthreads();   // the previous tile consumed (the first time: qs and the own cell written)
        for (int i = threadIdx.x; i < kTile * (kHD / 2); i += blockDim.x) {
            const int j = i / (kHD / 2), d2 = i % (kHD / 2);
            const int c = b0 + j;
            half2 k2 = __floats2half2_rn(0.f, 0.f), v2 = k2;
            if (c < c1) {
                const size_t cell = (size_t) (ring ? c % cap : c);
                k2 = reinterpret_cast<const half2*>(kb + cell * kHD)[d2];
                v2 = reinterpret_cast<const half2*>(vb + cell * kHD)[d2];
            }
            reinterpret_cast<half2*>(&ks[j][0])[d2] = k2;
            reinterpret_cast<half2*>(&vs[j][0])[d2] = v2;
        }
        __syncthreads();
        const int c = b0 + lane;
        float sc = -INFINITY;
        if (c < c1) {
            const half2* kr = reinterpret_cast<const half2*>(&ks[lane][0]);
            const float* qh = qs[warp];
            float acc = 0.f;
#pragma unroll 16
            for (int d2 = 0; d2 < kHD / 2; ++d2) {
                const float2 kf = __half22float2(kr[d2]);
                acc = fmaf(qh[2 * d2], kf.x, acc);
                acc = fmaf(qh[2 * d2 + 1], kf.y, acc);
            }
            sc = acc * scale;
        }
        const float mn = fmaxf(m, wmax(sc));
        const float corr = expf(m - mn);
        const float p = c < c1 ? expf(sc - mn) : 0.f;
        l = l * corr + wsum(p);
        a0 *= corr; a1 *= corr; a2 *= corr; a3 *= corr;
        m = mn;
        const int nj = min(kTile, c1 - b0);
        for (int j = 0; j < nj; ++j) {
            const float pj = __shfl_sync(0xffffffffu, p, j);
            const half2* vr = reinterpret_cast<const half2*>(&vs[j][lane * 4]);
            const float2 v01 = __half22float2(vr[0]), v23 = __half22float2(vr[1]);
            a0 = fmaf(pj, v01.x, a0);
            a1 = fmaf(pj, v01.y, a1);
            a2 = fmaf(pj, v23.x, a2);
            a3 = fmaf(pj, v23.y, a3);
        }
    }
    const size_t o = ((size_t) t * H + h) * NS + s;
    reinterpret_cast<float4*>(pacc + o * kHD)[lane] = make_float4(a0, a1, a2, a3);
    if (lane == 0) {
        pm[o] = m;
        pl[o] = l;
    }
    // the last block of (t, g) merges
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned prev = atomicAdd(ticket + t * HKV + g, 1u);
        s_last = prev == (unsigned) (NS - 1);
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const int ns_used = min(NS, (n + chunk - 1) / chunk);
    const size_t ob = ((size_t) t * H + h) * NS;
    const float sk = sinks[h];
    float M = sk;
    for (int si = 0; si < ns_used; ++si) M = fmaxf(M, __ldcg(pm + ob + si));
    float4 num = make_float4(0.f, 0.f, 0.f, 0.f);
    float den = expf(sk - M);
    for (int si = 0; si < ns_used; ++si) {
        const float w = expf(__ldcg(pm + ob + si) - M);
        const float4 pa = __ldcg(reinterpret_cast<const float4*>(pacc + (ob + si) * kHD) + lane);
        num.x = fmaf(w, pa.x, num.x); num.y = fmaf(w, pa.y, num.y);
        num.z = fmaf(w, pa.z, num.z); num.w = fmaf(w, pa.w, num.w);
        den = fmaf(w, __ldcg(pl + ob + si), den);
    }
    const float inv = 1.0f / den;
    const float4 v = make_float4(num.x * inv, num.y * inv, num.z * inv, num.w * inv);
    reinterpret_cast<float4*>(att + ((size_t) t * H + h) * kHD)[lane] = v;
    q8_store4(aq + ((size_t) t * H + h) * (kHD / 32) + lane / 8, v, lane);
    if (threadIdx.x == 0) ticket[t * HKV + g] = 0u;
}

/// x += y; h = rms(x) * w and h's q8_1 image, for rows of n = 1024 * NPT (block of 1024 per row, the values in registers).
template <int NPT>
__global__ void __launch_bounds__(1024) k_add_rms_q8(float* __restrict__ x, const float* __restrict__ y,
                                                     const float* __restrict__ w, float* __restrict__ h, QB* __restrict__ hq,
                                                     float eps) {
    constexpr int n = 1024 * NPT;
    __shared__ float red[32];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float* xr = x + (size_t) t * n;
    const float* yr = y + (size_t) t * n;
    float v[NPT];
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < NPT; ++k) {
        const int i = tid + k * 1024;
        v[k] = xr[i] + yr[i];
        xr[i] = v[k];
        s += v[k] * v[k];
    }
    s = wsum(s);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    if (warp == 0) {
        const float a = wsum(red[lane]);
        if (lane == 0) red[0] = a;
    }
    __syncthreads();
    const float r = rsqrtf(red[0] / (float) n + eps);
#pragma unroll
    for (int k = 0; k < NPT; ++k) {
        const int i = tid + k * 1024;
        const float hv = v[k] * r * w[i];
        h[(size_t) t * n + i] = hv;
        q36k::q8_store(hq + (size_t) t * (n >> 5), i, hv, lane);
    }
}

/// The MoE end for decode rows: x += sum_k w_k part_k + shared; h = rms(x) * wn and, with hq, its q8_1 image (rows of
/// n = 1024 * NPT, block of 1024 per row, the values in registers).
template <int NPT>
__global__ void __launch_bounds__(1024) k_moe_out_q8(float* __restrict__ x, const float* __restrict__ parts, const float* __restrict__ wts,
                                                     int K_, const float* __restrict__ shared, const float* __restrict__ wn,
                                                     float* __restrict__ h, float eps, QB* __restrict__ hq) {
    constexpr int n = 1024 * NPT;
    __shared__ float red[32];
    __shared__ float sw[16];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid < K_) sw[tid] = wts[(size_t) t * K_ + tid];
    __syncthreads();
    float* xr = x + (size_t) t * n;
    const float* pr = parts + (size_t) t * K_ * n;
    float v[NPT];
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < NPT; ++k) {
        const int i = tid + k * 1024;
        float acc = sw[0] * pr[i];
        for (int e = 1; e < K_; ++e) acc = fmaf(sw[e], pr[(size_t) e * n + i], acc);
        acc += shared[(size_t) t * n + i];
        v[k] = xr[i] + acc;
        xr[i] = v[k];
        s += v[k] * v[k];
    }
    s = wsum(s);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    if (warp == 0) {
        const float a = wsum(red[lane]);
        if (lane == 0) red[0] = a;
    }
    __syncthreads();
    const float r = rsqrtf(red[0] / (float) n + eps);
#pragma unroll
    for (int k = 0; k < NPT; ++k) {
        const int i = tid + k * 1024;
        const float hv = v[k] * r * wn[i];
        h[(size_t) t * n + i] = hv;
        if (hq) q36k::q8_store(hq + (size_t) t * (n >> 5), i, hv, lane);
    }
}

/// The router for T <= 8 decode rows in ONE launch: logits[t][e] = W[e] . h[t] (f32, one warp per expert row, 8 per
/// block), then the last block to finish (a ticket, reset by it) runs k_route_group's top-K and grouping.
/// grid (E / 8, T), block (32, 8).
template <int K>
__global__ void __launch_bounds__(256) k_router(const float* __restrict__ W, const float* __restrict__ hx, float* __restrict__ logits,
                                                int n_in, int E, int n_tok, int32_t* __restrict__ ids, float* __restrict__ weights,
                                                const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr,
                                                int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok,
                                                unsigned int* __restrict__ ticket) {
    __shared__ int s_last;
    __shared__ float slg[8][256];
    __shared__ int sids[64], sid[64];
    const int warp = threadIdx.y, lane = threadIdx.x, tid = warp * 32 + lane;
    {
        const int o = blockIdx.x * 8 + warp, t = blockIdx.y;
        const float4* wr = reinterpret_cast<const float4*>(W + (size_t) o * n_in);
        const float4* xr = reinterpret_cast<const float4*>(hx + (size_t) t * n_in);
        float acc = 0.f;
        for (int i = lane; i < n_in / 4; i += 32) {
            const float4 a = __ldg(wr + i), b = xr[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        acc = wsum(acc);
        if (lane == 0) logits[(size_t) t * E + o] = acc;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(ticket, 1u) == gridDim.x * gridDim.y - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    for (int i = tid; i < n_tok * E; i += 256) slg[i / E][i % E] = __ldcg(logits + i);
    __syncthreads();
    if (warp < n_tok) route_sig_warp<K>(slg[warp], lane, ids + (size_t) warp * K, weights + (size_t) warp * K, sids + warp * K);
    __syncthreads();
    const int n_ent = n_tok * K;
    const int e = tid;
    const int id = e < n_ent ? sids[e] : 0;
    const bool valid = e < n_ent;
    int rank = 0;
    if (valid) {
        for (int e2 = 0; e2 < n_ent; ++e2) {
            const int id2 = sids[e2];
            if (id2 < id || (id2 == id && e2 < e)) ++rank;
        }
        ent_dst[rank] = e;
        ent_tok[rank] = e / K;
        sid[rank] = id;
    }
    __syncthreads();
    if (valid && (rank == 0 || sid[rank - 1] != sid[rank])) {
        int gq = 0;
        for (int qq = 1; qq < rank; ++qq) gq += sid[qq - 1] != sid[qq] ? 1 : 0;
        gq += rank > 0 ? 1 : 0;
        grp_ptr[gq] = (unsigned long long) (blobs + (size_t) id * blob);
        grp_start[gq] = rank;
    }
    if (e == 0) {
        int G = n_ent > 0 ? 1 : 0;
        for (int qq = 1; qq < n_ent; ++qq) G += sid[qq - 1] != sid[qq] ? 1 : 0;
        grp_start[G] = n_ent;
        *n_groups = G;
        *ticket = 0u;
    }
}

}  // namespace flk

namespace flk {

/// The router for ONE decode row: logits[e] = W[e] . h (f32, one warp per expert row, 4 rows per block, every load of a
/// lane issued before the first product), then the last block (a ticket, reset by it) takes the top K with one warp and
/// writes the K groups (one expert each, ascending id) for native_expert_grouped.  n_in = 128 * NI4.
/// grid (E / 4), block (32, 4).
template <int K, int NI4>
__global__ void __launch_bounds__(128) k_router1(const float* __restrict__ W, const float* __restrict__ hx, float* __restrict__ logits,
                                                 int E, int32_t* __restrict__ ids, float* __restrict__ weights, const uint8_t* blobs,
                                                 unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                                                 int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok,
                                                 unsigned int* __restrict__ ticket) {
    __shared__ int s_last;
    __shared__ int sids[32];
    const int warp = threadIdx.y, lane = threadIdx.x;
    {
        const int o = blockIdx.x * 4 + warp;
        const float4* wr = reinterpret_cast<const float4*>(W + (size_t) o * (NI4 * 128));
        const float4* xr = reinterpret_cast<const float4*>(hx);
        float4 a[NI4], b[NI4];
#pragma unroll
        for (int k = 0; k < NI4; ++k) a[k] = __ldg(wr + lane + 32 * k);
#pragma unroll
        for (int k = 0; k < NI4; ++k) b[k] = xr[lane + 32 * k];
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < NI4; ++k) acc += a[k].x * b[k].x + a[k].y * b[k].y + a[k].z * b[k].z + a[k].w * b[k].w;
        acc = wsum(acc);
        if (lane == 0) logits[o] = acc;
    }
    __threadfence();
    __syncthreads();
    if (warp == 0 && lane == 0) s_last = atomicAdd(ticket, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_last || warp != 0) return;
    __threadfence();
    constexpr int NE = 256, V = NE / 32;
    float values[V];
#pragma unroll
    for (int i = 0; i < V; ++i) values[i] = 1.0f / (1.0f + expf(-__ldcg(logits + lane + i * 32)));
    float selected = 0.0f, selected_sum = 0.0f;
    int my_id = 0;
    for (int rank = 0; rank < K; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < V; ++i)
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) values[expert / 32] = -INFINITY;
        selected_sum += best;
        if (rank == lane) { selected = best; my_id = expert; }
    }
    selected_sum = fmaxf(selected_sum, 6.103515625e-5f);
    if (lane < K) {
        ids[lane] = my_id;
        weights[lane] = selected / selected_sum;
        sids[lane] = my_id;
    }
    __syncwarp();
    if (lane < K) {
        int rank = 0;
        for (int j = 0; j < K; ++j) rank += sids[j] < my_id ? 1 : 0;
        ent_dst[rank] = lane;
        ent_tok[rank] = 0;
        grp_ptr[rank] = (unsigned long long) (blobs + (size_t) my_id * blob);
        grp_start[rank] = rank;
    }
    if (lane == 0) {
        grp_start[K] = K;
        *n_groups = K;
        *ticket = 0u;
    }
}

}  // namespace flk

namespace flk {

constexpr int kKS2 = kHD + 8;   // K rows in shared memory 272 bytes apart: a quarter-warp's 16-byte row reads hit 8 x 4 banks

/// One head (4 values per lane) RMS-normed and, with `rope` (a [pos][64] table of cos / sin pairs), NEOX-rotated.
__device__ __forceinline__ float4 head_norm_rope_t(float4 v, const float* __restrict__ w, int lane, const float2* __restrict__ rope,
                                                   float eps) {
    const float ss = wsum((v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w));
    const float r = rsqrtf(ss / (float) kHD + eps);
    const float4 wv = reinterpret_cast<const float4*>(w)[lane];
    v.x = v.x * r * wv.x; v.y = v.y * r * wv.y; v.z = v.z * r * wv.z; v.w = v.w * r * wv.w;
    if (!rope) return v;
    const float4 o = make_float4(__shfl_xor_sync(0xffffffffu, v.x, 16), __shfl_xor_sync(0xffffffffu, v.y, 16),
                                 __shfl_xor_sync(0xffffffffu, v.z, 16), __shfl_xor_sync(0xffffffffu, v.w, 16));
    const bool lo = lane < 16;
    const float4* cs4 = reinterpret_cast<const float4*>(rope + 4 * (lane & 15));   // pairs p0 .. p0 + 3: (cos, sin) each
    const float4 c01 = cs4[0], c23 = cs4[1];
    const float c[4] = {c01.x, c01.z, c23.x, c23.z}, s[4] = {c01.y, c01.w, c23.y, c23.w};
    const float a[4] = {v.x, v.y, v.z, v.w}, b[4] = {o.x, o.y, o.z, o.w};
    float r4[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) r4[k] = lo ? a[k] * c[k] - b[k] * s[k] : b[k] * s[k] + a[k] * c[k];
    return make_float4(r4[0], r4[1], r4[2], r4[3]);
}

/// k_attn_dec with the rotary angles from a table (rope_tab: [ctx][64] (cos, sin), null on a no-rope layer), 16-byte K / V
/// tile loads and exp2 softmax.  The same launch shape and outputs.
__global__ void __launch_bounds__(256) k_attn_dec2(const float* __restrict__ qs_, int ldq, const float* __restrict__ ks_, int ldk,
                                                   const float* __restrict__ vs_, int ldv, const float* __restrict__ qw,
                                                   const float* __restrict__ kw, __half* __restrict__ kc, __half* __restrict__ vc,
                                                   int H, int cap, int ring, int win, const float2* __restrict__ rope_tab,
                                                   const int32_t* __restrict__ pos0, float eps, float scale, int NS,
                                                   float* __restrict__ pacc, float* __restrict__ pm, float* __restrict__ pl,
                                                   const float* __restrict__ sinks, float* __restrict__ att, QB* __restrict__ aq,
                                                   unsigned int* __restrict__ ticket) {
    __shared__ __align__(16) __half ks[kTile][kKS2];
    __shared__ __align__(16) __half vs[kTile][kHD];
    __shared__ __align__(16) float qs[kGrp][kHD];
    __shared__ int s_last;
    const int s = blockIdx.x, g = blockIdx.y, t = blockIdx.z, HKV = gridDim.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = g * kGrp + warp;
    const int pos = *pos0 + t;
    const int lo = win > 0 ? max(0, pos - (win - 1)) : 0;
    const int n = pos - lo + 1;
    int chunk = (n + NS - 1) / NS;
    chunk = (chunk + kTile - 1) / kTile * kTile;
    const int c0 = lo + s * chunk, c1 = min(pos + 1, c0 + chunk);
    const float2* rope = rope_tab ? rope_tab + (size_t) pos * (kHD / 2) : nullptr;
    constexpr float kLog2e = 1.4426950408889634f;
    const float sl2 = scale * kLog2e;   // scores in log2 units
    {
        const float4 v = reinterpret_cast<const float4*>(qs_ + (size_t) t * ldq + (size_t) h * kHD)[lane];
        float4 qv = head_norm_rope_t(v, qw, lane, rope, eps);
        qv.x *= sl2; qv.y *= sl2; qv.z *= sl2; qv.w *= sl2;
        reinterpret_cast<float4*>(qs[warp])[lane] = qv;
    }
    if (c0 <= pos && pos < c1) {
        const size_t cell = (size_t) (ring ? pos % cap : pos);
        if (warp == 0) {
            const float4 v = reinterpret_cast<const float4*>(ks_ + (size_t) t * ldk + (size_t) g * kHD)[lane];
            const float4 k = head_norm_rope_t(v, kw, lane, rope, eps);
            half2* dst = reinterpret_cast<half2*>(kc + ((size_t) g * cap + cell) * kHD) + 2 * lane;
            dst[0] = __floats2half2_rn(k.x, k.y);
            dst[1] = __floats2half2_rn(k.z, k.w);
        } else if (warp == 1) {
            const float4 v = reinterpret_cast<const float4*>(vs_ + (size_t) t * ldv + (size_t) g * kHD)[lane];
            half2* dst = reinterpret_cast<half2*>(vc + ((size_t) g * cap + cell) * kHD) + 2 * lane;
            dst[0] = __floats2half2_rn(v.x, v.y);
            dst[1] = __floats2half2_rn(v.z, v.w);
        }
    }
    float m = -INFINITY, l = 0.f;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const __half* kb = kc + (size_t) g * cap * kHD;
    const __half* vb = vc + (size_t) g * cap * kHD;
    for (int b0 = c0; b0 < c1; b0 += kTile) {
        __syncthreads();
        {   // 32 rows x 16 uint4 for K and for V: 512 each, two of each per thread
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int i = threadIdx.x + r * 256;
                const int j = i >> 4, d8 = i & 15;
                const int c = b0 + j;
                uint4 k4 = make_uint4(0u, 0u, 0u, 0u), v4 = k4;
                if (c < c1) {
                    const size_t cell = (size_t) (ring ? c % cap : c);
                    k4 = __ldg(reinterpret_cast<const uint4*>(kb + cell * kHD) + d8);
                    v4 = __ldg(reinterpret_cast<const uint4*>(vb + cell * kHD) + d8);
                }
                reinterpret_cast<uint4*>(&ks[j][0])[d8] = k4;
                reinterpret_cast<uint4*>(&vs[j][0])[d8] = v4;
            }
        }
        __syncthreads();
        const int c = b0 + lane;
        float sc = -INFINITY;
        if (c < c1) {
            const uint4* kr = reinterpret_cast<const uint4*>(&ks[lane][0]);
            const float4* qh = reinterpret_cast<const float4*>(qs[warp]);
            float acc = 0.f;
#pragma unroll
            for (int d8 = 0; d8 < kHD / 8; ++d8) {
                const uint4 k4 = kr[d8];
                const float4 qa = qh[2 * d8], qb = qh[2 * d8 + 1];
                const float2 k0 = __half22float2(*reinterpret_cast<const half2*>(&k4.x));
                const float2 k1 = __half22float2(*reinterpret_cast<const half2*>(&k4.y));
                const float2 k2 = __half22float2(*reinterpret_cast<const half2*>(&k4.z));
                const float2 k3 = __half22float2(*reinterpret_cast<const half2*>(&k4.w));
                acc = fmaf(qa.x, k0.x, acc); acc = fmaf(qa.y, k0.y, acc);
                acc = fmaf(qa.z, k1.x, acc); acc = fmaf(qa.w, k1.y, acc);
                acc = fmaf(qb.x, k2.x, acc); acc = fmaf(qb.y, k2.y, acc);
                acc = fmaf(qb.z, k3.x, acc); acc = fmaf(qb.w, k3.y, acc);
            }
            sc = acc;
        }
        const float mn = fmaxf(m, wmax(sc));
        const float corr = exp2f(m - mn);
        const float p = c < c1 ? exp2f(sc - mn) : 0.f;
        l = l * corr + wsum(p);
        a0 *= corr; a1 *= corr; a2 *= corr; a3 *= corr;
        m = mn;
        const int nj = min(kTile, c1 - b0);
        for (int j = 0; j < nj; ++j) {
            const float pj = __shfl_sync(0xffffffffu, p, j);
            const uint2 v2 = reinterpret_cast<const uint2*>(&vs[j][0])[lane];
            const float2 v01 = __half22float2(*reinterpret_cast<const half2*>(&v2.x));
            const float2 v23 = __half22float2(*reinterpret_cast<const half2*>(&v2.y));
            a0 = fmaf(pj, v01.x, a0);
            a1 = fmaf(pj, v01.y, a1);
            a2 = fmaf(pj, v23.x, a2);
            a3 = fmaf(pj, v23.y, a3);
        }
    }
    const size_t o = ((size_t) t * H + h) * NS + s;
    reinterpret_cast<float4*>(pacc + o * kHD)[lane] = make_float4(a0, a1, a2, a3);
    if (lane == 0) {
        pm[o] = m;
        pl[o] = l;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned prev = atomicAdd(ticket + t * HKV + g, 1u);
        s_last = prev == (unsigned) (NS - 1);
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const int ns_used = min(NS, (n + chunk - 1) / chunk);
    const size_t ob = ((size_t) t * H + h) * NS;
    const float sk = sinks[h] * kLog2e;
    float M = sk;
    for (int si = 0; si < ns_used; ++si) M = fmaxf(M, __ldcg(pm + ob + si));
    float4 num = make_float4(0.f, 0.f, 0.f, 0.f);
    float den = exp2f(sk - M);
    for (int si = 0; si < ns_used; ++si) {
        const float w = exp2f(__ldcg(pm + ob + si) - M);
        const float4 pa = __ldcg(reinterpret_cast<const float4*>(pacc + (ob + si) * kHD) + lane);
        num.x = fmaf(w, pa.x, num.x); num.y = fmaf(w, pa.y, num.y);
        num.z = fmaf(w, pa.z, num.z); num.w = fmaf(w, pa.w, num.w);
        den = fmaf(w, __ldcg(pl + ob + si), den);
    }
    const float inv = 1.0f / den;
    const float4 v = make_float4(num.x * inv, num.y * inv, num.z * inv, num.w * inv);
    reinterpret_cast<float4*>(att + ((size_t) t * H + h) * kHD)[lane] = v;
    q8_store4(aq + ((size_t) t * H + h) * (kHD / 32) + lane / 8, v, lane);
    if (threadIdx.x == 0) ticket[t * HKV + g] = 0u;
}

}  // namespace flk

namespace flk {

/// k_router1 with one expert row per block (128 threads, 4 float4 of the row per thread): 256 blocks in flight instead
/// of 64, then the same last-block top K and groups.  grid E, block 128.  n_in = 2048.
/// (BF: the router's weights in BF16, 8 per 16-byte load: the half of the f32 router's bytes.)
template <int K, bool BF = false>
__global__ void __launch_bounds__(128) k_router256(const void* __restrict__ Wv, const float* __restrict__ hx, float* __restrict__ logits,
                                                   int32_t* __restrict__ ids, float* __restrict__ weights, const uint8_t* blobs,
                                                   unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                                                   int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok,
                                                   unsigned int* __restrict__ ticket) {
    __shared__ float red[4];
    __shared__ int s_last;
    __shared__ int sids[32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    {
        float acc = 0.f;
        const float4* xr = reinterpret_cast<const float4*>(hx);
        if constexpr (BF) {   // thread: values 8 tid .. 8 tid + 7 and 1024 + 8 tid ..
            const uint4* wr = reinterpret_cast<const uint4*>((const uint16_t*) Wv + (size_t) blockIdx.x * 2048);
            uint4 a[2];
            float4 b[4];
#pragma unroll
            for (int k = 0; k < 2; ++k) a[k] = __ldg(wr + tid + 128 * k);
#pragma unroll
            for (int k = 0; k < 2; ++k) { b[2 * k] = xr[2 * (tid + 128 * k)]; b[2 * k + 1] = xr[2 * (tid + 128 * k) + 1]; }
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const uint32_t w4[4] = {a[k].x, a[k].y, a[k].z, a[k].w};
                const float xs[8] = {b[2 * k].x, b[2 * k].y, b[2 * k].z, b[2 * k].w,
                                     b[2 * k + 1].x, b[2 * k + 1].y, b[2 * k + 1].z, b[2 * k + 1].w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc = fmaf(__uint_as_float(w4[j] << 16), xs[2 * j], acc);
                    acc = fmaf(__uint_as_float(w4[j] & 0xffff0000u), xs[2 * j + 1], acc);
                }
            }
        } else {
            const float4* wr = reinterpret_cast<const float4*>((const float*) Wv + (size_t) blockIdx.x * 2048);
            float4 a[4], b[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) a[k] = __ldg(wr + tid + 128 * k);
#pragma unroll
            for (int k = 0; k < 4; ++k) b[k] = xr[tid + 128 * k];
#pragma unroll
            for (int k = 0; k < 4; ++k) acc += a[k].x * b[k].x + a[k].y * b[k].y + a[k].z * b[k].z + a[k].w * b[k].w;
        }
        acc = wsum(acc);
        if (lane == 0) red[warp] = acc;
        __syncthreads();
        if (tid == 0) logits[blockIdx.x] = (red[0] + red[1]) + (red[2] + red[3]);
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(ticket, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_last || warp != 0) return;
    __threadfence();
    constexpr int NE = 256, V = NE / 32;
    float values[V];
#pragma unroll
    for (int i = 0; i < V; ++i) values[i] = 1.0f / (1.0f + expf(-__ldcg(logits + lane + i * 32)));
    float selected = 0.0f, selected_sum = 0.0f;
    int my_id = 0;
    for (int rank = 0; rank < K; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < V; ++i)
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) values[expert / 32] = -INFINITY;
        selected_sum += best;
        if (rank == lane) { selected = best; my_id = expert; }
    }
    selected_sum = fmaxf(selected_sum, 6.103515625e-5f);
    if (lane < K) {
        ids[lane] = my_id;
        weights[lane] = selected / selected_sum;
        sids[lane] = my_id;
    }
    __syncwarp();
    if (lane < K) {
        int rank = 0;
        for (int j = 0; j < K; ++j) rank += sids[j] < my_id ? 1 : 0;
        ent_dst[rank] = lane;
        ent_tok[rank] = 0;
        grp_ptr[rank] = (unsigned long long) (blobs + (size_t) my_id * blob);
        grp_start[rank] = rank;
    }
    if (lane == 0) {
        grp_start[K] = K;
        *n_groups = K;
        *ticket = 0u;
    }
}

}  // namespace flk

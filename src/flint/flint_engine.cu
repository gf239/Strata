// src/flint/flint_engine.cu - strata-flint: the engine path for Recursive Flint (GGUF architecture flint, 40B-A2.5B),
// a driver of its own beside Flash-Next's and Qwen3.6's (src/qwen35/q36_engine.cu, whose structure it follows), built
// from Strata's kernels:
//
//   * weights straight from the GGUF (llama.cpp's flint layout): dense Q8_0 / K-quant matrices in their blocks (native
//     MMVQ; attn_q / attn_k / attn_v as one [q | k | v] matrix when they share a type), the 1-D tensors and the F32
//     router as f32, the routed experts repacked at load into native expert blobs [gate | up | down] per (layer, expert)
//     for `native_expert_grouped` (decode) and llama.cpp's MMQ (prompt); the token embedding in pinned host memory
//     (one row per token crosses PCIe), --embd-gpu keeps it in VRAM;
//   * the pre-norm residual layer: x += attn(rms(x) * attn_norm); x += ffn(rms(x) * ffn_norm);
//   * attention: 32 query heads over 4 KV heads of 128, per-head q / k RMSNorm, a learned sink logit per head in every
//     softmax denominator, window layers (the current cell and the 512 before it, NEOX RoPE, a ring cache) and full
//     layers (no positional encoding); K / V in fp16; decode as split-K over a fixed grid (graph-safe), the prompt as
//     cuBLAS FP16 GEMMs around a masked softmax;
//   * MoE: sigmoid router over 256 experts, the top 6 by score, their scores renormalised to sum 1; the always-on shared
//     expert (SwiGLU, no gate) added; the leading dense layers a SwiGLU MLP;
//   * the prompt read in chunks of --chunk tokens, decode one token at a time from a captured graph (positions on the
//     device), --no-graph eagerly.
//
// Modes: --serve (serve/server.py's engine protocol: READY / GEN / PP / T / DONE / STOP), --tokens IDS [--n N]
// [--force IDS] [--dump-full-logits F] (tools/qwen35/kl.py's reference interface), --bench-pp N / --bench-tg N.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/flint/engine.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "flint_dec.cuh"
#include "flint_fa.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

namespace K = strata::kernels;
namespace MMQ = strata::prefill::mmq;
using flk::QB;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-flint: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kThreads = 256;

// ------------------------------------------------------------------------------------------------ small kernels
__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// y[r] = rms(x[r]) * w, one block per row of n.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
/// x[r] += y[r]; h[r] = rms(x[r]) * w (one block per row of n).
__global__ void k_add_rms(float* x, const float* y, const float* w, float* h, int n, float eps) {
    __shared__ float sh[kThreads];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float* hr = h + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) hr[i] = xr[i] * r * w[i];
}
__global__ void k_swiglu(const float* g, const float* u, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}
/// y[t][o] = W[o] . x[t] for f32 W [n_out][n_in], n_in a multiple of 128: one warp per row, float4 loads, 8 rows per
/// block (grid (ceil(n_out / 8), T)).
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }
unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

/// out[t][o] = W[o] . x[t] in FP32 for a prompt chunk's router (q36_engine.cu's k_sgemm_nt): a (16 TM) x 64 tile per
/// block, K in steps of 16 through shared memory, split-K partials (blockIdx.z) summed by k_sum_splits.
template <int TM, bool BF = false>
__global__ void __launch_bounds__(256) k_sgemm_nt(const float* __restrict__ W, const float* __restrict__ x,
                                                  float* __restrict__ out, int K_, int n_out, int T, int ldo, int kz) {
    constexpr int BM = 16 * TM, BN = 64, BK = 16;
    const int kb = blockIdx.z * kz, ke = min(K_, kb + kz);
    out += (size_t) blockIdx.z * T * ldo;
    __shared__ float As[BK][BM + 4], Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tm = (tid / 16) * TM, tn = (tid % 16) * 4;
    const int lr = tid / 4, lk = (tid % 4) * 4;
    float acc[TM][4] = {};
    for (int k0 = kb; k0 < ke; k0 += BK) {
        if (lr < BM) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + lr < T && k0 + lk < ke) a = *reinterpret_cast<const float4*>(x + (size_t) (m0 + lr) * K_ + k0 + lk);
            As[lk][lr] = a.x; As[lk + 1][lr] = a.y; As[lk + 2][lr] = a.z; As[lk + 3][lr] = a.w;
        }
        float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n0 + lr < n_out && k0 + lk < ke) {
            if constexpr (BF) {   // W holds BF16 bits: four of them
                const uint2 w2 = *reinterpret_cast<const uint2*>((const uint16_t*) W + (size_t) (n0 + lr) * K_ + k0 + lk);
                b = make_float4(__uint_as_float(w2.x << 16), __uint_as_float(w2.x & 0xffff0000u), __uint_as_float(w2.y << 16),
                                __uint_as_float(w2.y & 0xffff0000u));
            } else {
                b = *reinterpret_cast<const float4*>(W + (size_t) (n0 + lr) * K_ + k0 + lk);
            }
        }
        Bs[lk][lr] = b.x; Bs[lk + 1][lr] = b.y; Bs[lk + 2][lr] = b.z; Bs[lk + 3][lr] = b.w;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ar[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = As[k][tm + i];
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tn]);
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (m0 + tm + i < T && n0 + tn + j < n_out) out[(size_t) (m0 + tm + i) * ldo + n0 + tn + j] = acc[i][j];
}
__global__ void k_sum_splits(const float* part, float* out, int64_t n, int splits) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.f;
    for (int z = 0; z < splits; ++z) a += part[(size_t) z * n + i];
    out[i] = a;
}
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

/// The prompt's routed entries sorted by expert for MMQ (q36_engine.cu): bounds[x] .. bounds[x + 1] are expert x's
/// rows; sorted row r holds entry dst[r] (= t * K + k) of token src[r]; inv[e] is entry e's sorted row.
__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) atomicAdd(&cnt[ids[e]], 1);
}
__global__ void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur, int32_t* max_rows) {
    if (threadIdx.x != 0) return;
    int pos = 0, mx = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
        mx = max(mx, cnt[x]);
    }
    bounds[n_expert] = pos;
    *max_rows = mx;
}
__global__ void k_route_scatter(const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst, int32_t* src,
                                int32_t* inv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    const int r = atomicAdd(&cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K_;
    inv[e] = r;
}
/// The routed entries grouped per expert for native_expert_grouped when MMQ cannot take the prompt (q36's k_group).
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs, unsigned long long blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                        int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}

/// The greedy pick of each logits row: argmax, the lowest id on ties.  Stage 1: gridDim.x blocks write the best of
/// their stride into (pv, pi); stage 2 (one block) reads those.
constexpr int kArgBlocks = 128;
__global__ void k_argmax(const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2) {
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    x += (size_t) blockIdx.y * (stage2 ? 0 : n);
    pv += (size_t) blockIdx.y * kArgBlocks;
    pi += (size_t) blockIdx.y * kArgBlocks;
    out += blockIdx.y;
    float v = -INFINITY;
    int id = 0x7fffffff;
    if (stage2) {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (pv[i] > v || (pv[i] == v && pi[i] < id)) { v = pv[i]; id = pi[i]; }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            if (x[i] > v) { v = x[i]; id = i; }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (stage2) *out = bi[0];
        else { pv[blockIdx.x] = bv[0]; pi[blockIdx.x] = bi[0]; }
    }
}

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    bool swa = false, rope = false, moe = false;
    const float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *sinks = nullptr;
    bool fused_qkv = false;
    Mat qkv;            // [q | k | v] rows (fused_qkv)
    Mat wq, wk, wv;     // otherwise
    Mat wo;
    Mat gate, up, down;   // the dense MLP
    const float* router = nullptr;   // [n_expert][N] f32 (or BF16 bits when router_bf16)
    bool router_bf16 = false;
    Mat sh_gate, sh_up, sh_down;
    K::NativeExpertLayout L;
    uint8_t* blobs = nullptr;   // n_expert blobs, `stride` bytes apart, then kMmqTail zero bytes
    size_t stride = 0;
};

struct Model {
    std::shared_ptr<strata::GgufModel> file;
    int n_layers = 0, N = 0, H = 0, HKV = 0, HD = 0, n_expert = 0, top_k = 0, FF = 0, FS = 0, FFD = 0, n_vocab = 0;
    int win = 0, n_dense = 0, ctx_train = 262144;
    float eps = 1e-6f, rope_base = 10000.f, w_scale = 1.f;
    bool w_norm = true;
    std::vector<Layer> layers;
    const float* out_norm = nullptr;
    Mat head;
    const uint8_t* embd = nullptr;   // device pointer (VRAM, or pinned host memory mapped)
    int embd_type = -1;
    size_t embd_row = 0;
    size_t vram = 0;
};

constexpr size_t kMmqTail = 4096;   // zeroed bytes after the last expert (MMQ reads a little past a matrix)
size_t gcd_sz(size_t a, size_t b) { return b ? gcd_sz(b, a % b) : a; }
size_t lcm_sz(size_t a, size_t b) { return a / gcd_sz(a, b) * b; }
size_t block_bytes(int type) {
    int be = 0, bb = 0;
    return strata::block_geometry((uint32_t) type, be, bb) ? (size_t) bb : 1;
}
size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}
float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    uint32_t u;
    if (exponent == 0) {
        if (mantissa == 0) {
            u = sign;
        } else {
            uint32_t mm = mantissa, shift = 0;
            while ((mm & 0x400) == 0) {
                mm <<= 1;
                ++shift;
            }
            u = sign | ((uint32_t) (127 - 14 - shift) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (exponent == 0x1f) {
        u = sign | 0x7f800000u | (mantissa << 13);
    } else {
        u = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void* dmalloc(size_t bytes, size_t& total, const char* what) {
    void* d = nullptr;
    const cudaError_t e = cudaMalloc(&d, bytes);
    if (e != cudaSuccess) {
        size_t fr = 0, tot = 0;
        (void) cudaGetLastError();
        cudaMemGetInfo(&fr, &tot);
        std::fprintf(stderr, "strata-flint: out of VRAM placing %s (%.1f MiB; %.2f GiB placed, %.2f GiB free): use a smaller "
                     "quant, a shorter --max-context or a smaller --chunk\n", what, bytes / 1048576.0, total / 1073741824.0,
                     fr / 1073741824.0);
        std::exit(1);
    }
    total += bytes;
    return d;
}

/// A tensor as f32 on the device (F32 as it is, BF16 / F16 widened).
float* upload_f32(const strata::TensorInfo& t, const uint8_t* src, size_t& total) {
    std::vector<float> h((size_t) t.elements());
    if (t.type == 0) std::memcpy(h.data(), src, h.size() * 4);
    else if (t.type == 30)
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 1)
        for (size_t i = 0; i < h.size(); ++i) h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else {
        std::fprintf(stderr, "strata-flint: %s is %s, expected F32, BF16 or F16\n", t.name.c_str(), t.type_name());
        std::exit(1);
    }
    float* d = (float*) dmalloc(h.size() * 4, total, t.name.c_str());
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

bool load(const std::string& path, Model& m, std::string& err, bool embd_gpu) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    const strata::MetaValue* arch = gm.meta().get("general.architecture");
    if (!arch || arch->s != "flint") {
        err = "strata-flint runs Recursive Flint (GGUF architecture flint) files only";
        return false;
    }
    auto key = [&](const std::string& k) -> const strata::MetaValue* { return gm.meta().get("flint." + k); };
    auto num = [&](const std::string& k, double& out) -> bool {
        const strata::MetaValue* v = key(k);
        if (!v || !v->is_num()) {
            err = "missing flint." + k;
            return false;
        }
        out = v->num();
        return true;
    };
    double d_layers, d_n, d_h, d_hkv, d_hd, d_ne, d_k, d_ff, d_fs, d_ffd, d_win, d_dense, d_eps, d_base;
    if (!num("block_count", d_layers) || !num("embedding_length", d_n) || !num("attention.head_count", d_h) ||
        !num("attention.head_count_kv", d_hkv) || !num("attention.key_length", d_hd) || !num("expert_count", d_ne) ||
        !num("expert_used_count", d_k) || !num("expert_feed_forward_length", d_ff) ||
        !num("expert_shared_feed_forward_length", d_fs) || !num("feed_forward_length", d_ffd) ||
        !num("attention.sliding_window", d_win) || !num("leading_dense_block_count", d_dense) ||
        !num("attention.layer_norm_rms_epsilon", d_eps) || !num("rope.freq_base", d_base))
        return false;
    m.n_layers = (int) d_layers; m.N = (int) d_n; m.H = (int) d_h; m.HKV = (int) d_hkv; m.HD = (int) d_hd;
    m.n_expert = (int) d_ne; m.top_k = (int) d_k; m.FF = (int) d_ff; m.FS = (int) d_fs; m.FFD = (int) d_ffd;
    m.win = (int) d_win; m.n_dense = (int) d_dense; m.eps = (float) d_eps; m.rope_base = (float) d_base;
    if (const strata::MetaValue* v = key("context_length")) m.ctx_train = (int) v->num();
    double gating = 2, wnorm = 1, wscale = 1;
    if (const strata::MetaValue* v = key("expert_gating_func")) gating = v->num();
    if (const strata::MetaValue* v = key("expert_weights_norm")) wnorm = v->num();
    if (const strata::MetaValue* v = key("expert_weights_scale")) wscale = v->num();
    m.w_norm = wnorm != 0;
    m.w_scale = wscale == 0 ? 1.f : (float) wscale;
    if (gating != 2 || !m.w_norm || m.w_scale != 1.f) {
        err = "a router this driver does not take (it runs sigmoid gating, renormalised weights, scale 1)";
        return false;
    }
    if (m.HD != flk::kHD || m.H != m.HKV * flk::kGrp || m.n_expert != 256 || (m.top_k != 6 && m.top_k != 8) ||
        m.N % 256 != 0 || m.N > 2048 || m.FF % 32 != 0) {
        err = "a geometry this driver's kernels do not take (head 128, 8 query heads per KV head, 256 experts, top-6 / 8, "
              "embedding <= 2048)";
        return false;
    }
    const strata::MetaValue* swa_pat = key("attention.sliding_window_pattern");
    const strata::MetaValue* rope_pat = key("attention.rope_pattern");
    if (!swa_pat || !rope_pat || swa_pat->items.size() != (size_t) m.n_layers || rope_pat->items.size() != (size_t) m.n_layers) {
        err = "missing flint.attention.sliding_window_pattern / rope_pattern";
        return false;
    }
    size_t& bytes = m.vram;
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-flint: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        return upload_f32(*t, gm.shard(s).tensor_data(*t), bytes);
    };
    auto check_mat = [&](const std::string& name, const strata::TensorInfo* t) {
        if (!K::native_mmvq_supported((int) t->type)) {
            std::fprintf(stderr, "strata-flint: %s is %s, which the native MMVQ does not take\n", name.c_str(), t->type_name());
            std::exit(1);
        }
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        check_mat(name, t);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = (uint8_t*) dmalloc(nb, bytes, name.c_str());
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        w.d = d;
        return w;
    };
    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    m.layers.resize((size_t) m.n_layers);
    uint8_t* stage = nullptr;   // pinned staging for one layer's expert blobs
    size_t stage_bytes = 0;
    for (int l = 0; l < m.n_layers; ++l) {
        Layer& y = m.layers[(size_t) l];
        y.swa = swa_pat->items[(size_t) l].u != 0;
        y.rope = rope_pat->items[(size_t) l].u != 0;
        y.moe = l >= m.n_dense;
        y.attn_norm = f32(L(l, "attn_norm.weight"));
        y.ffn_norm = f32(L(l, "ffn_norm.weight"));
        y.q_norm = f32(L(l, "attn_q_norm.weight"));
        y.k_norm = f32(L(l, "attn_k_norm.weight"));
        y.sinks = f32(L(l, "attn_sinks.weight"));
        {
            size_t sq = 0, sk = 0, sv = 0;
            const strata::TensorInfo* tq = find(L(l, "attn_q.weight"), sq);
            const strata::TensorInfo* tk = find(L(l, "attn_k.weight"), sk);
            const strata::TensorInfo* tv = find(L(l, "attn_v.weight"), sv);
            if (tq->type == tk->type && tq->type == tv->type && tq->shape[0] == tk->shape[0] && tq->shape[0] == tv->shape[0]) {
                check_mat(L(l, "attn_q.weight"), tq);
                const uint64_t bq = strata::tensor_payload_bytes(*tq), bk = strata::tensor_payload_bytes(*tk),
                               bv = strata::tensor_payload_bytes(*tv);
                uint8_t* d = (uint8_t*) dmalloc(bq + bk + bv, bytes, "attn_qkv");
                CK(cudaMemcpy(d, gm.shard(sq).tensor_data(*tq), bq, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d + bq, gm.shard(sk).tensor_data(*tk), bk, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d + bq + bk, gm.shard(sv).tensor_data(*tv), bv, cudaMemcpyHostToDevice));
                y.fused_qkv = true;
                y.qkv.d = d;
                y.qkv.type = (int) tq->type;
                y.qkv.n_in = (int) tq->shape[0];
                y.qkv.n_out = (int) (tq->shape[1] + tk->shape[1] + tv->shape[1]);
            } else {
                y.wq = mat(L(l, "attn_q.weight"));
                y.wk = mat(L(l, "attn_k.weight"));
                y.wv = mat(L(l, "attn_v.weight"));
            }
        }
        y.wo = mat(L(l, "attn_output.weight"));
        if (!y.moe) {
            y.gate = mat(L(l, "ffn_gate.weight"));
            y.up = mat(L(l, "ffn_up.weight"));
            y.down = mat(L(l, "ffn_down.weight"));
            continue;
        }
        if (gm.find(L(l, "exp_probs_b.bias")) || gm.find(L(l, "ffn_exp_probs_b.bias"))) {
            err = L(l, "ffn_exp_probs_b.bias: a router bias this driver does not take yet");
            return false;
        }
        {   // the router as the file has it: F32 (llama.cpp's default), or BF16 kept as BF16 (the half of the bytes)
            size_t sr = 0;
            const strata::TensorInfo* tr = find(L(l, "ffn_gate_inp.weight"), sr);
            if (tr->type == 30) {
                const uint64_t nb = strata::tensor_payload_bytes(*tr);
                uint8_t* d = (uint8_t*) dmalloc(nb, bytes, "router");
                CK(cudaMemcpy(d, gm.shard(sr).tensor_data(*tr), nb, cudaMemcpyHostToDevice));
                y.router = (const float*) d;
                y.router_bf16 = true;
            } else {
                y.router = upload_f32(*tr, gm.shard(sr).tensor_data(*tr), bytes);
            }
        }
        y.sh_gate = mat(L(l, "ffn_gate_shexp.weight"));
        y.sh_up = mat(L(l, "ffn_up_shexp.weight"));
        y.sh_down = mat(L(l, "ffn_down_shexp.weight"));
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo* tg = find(L(l, "ffn_gate_exps.weight"), sg);
        const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
        const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
        if (tg->type != tu->type) {
            err = L(l, "ffn_gate_exps / ffn_up_exps differ in type");
            return false;
        }
        if (!K::native_expert_supported((int) tg->type, (int) td->type, m.N, m.FF)) {
            err = L(l, "experts are ") + tg->type_name() + "/" + td->type_name() + ", which the native expert kernels do not take";
            return false;
        }
        y.L = K::native_expert_layout((int) tg->type, (int) td->type, m.N, m.FF);
        const size_t gsz = (size_t) m.FF * y.L.gu_row, dsz = (size_t) m.N * y.L.d_row;
        const size_t al = lcm_sz(lcm_sz(block_bytes(y.L.gu_type), block_bytes(y.L.d_type)), 16);
        y.stride = (y.L.bytes + al - 1) / al * al;
        const size_t total = (size_t) m.n_expert * y.stride + kMmqTail;
        if (total > stage_bytes) {
            if (stage) CK(cudaFreeHost(stage));
            CK(cudaHostAlloc((void**) &stage, total, cudaHostAllocDefault));
            stage_bytes = total;
        }
        std::memset(stage, 0, total);
        const uint8_t* pg = gm.shard(sg).tensor_data(*tg);
        const uint8_t* pu = gm.shard(su).tensor_data(*tu);
        const uint8_t* pd = gm.shard(sd).tensor_data(*td);
        for (int e = 0; e < m.n_expert; ++e) {
            uint8_t* dst = stage + (size_t) e * y.stride;
            std::memcpy(dst, pg + (size_t) e * gsz, gsz);
            std::memcpy(dst + y.L.up_off, pu + (size_t) e * gsz, gsz);
            std::memcpy(dst + y.L.down_off, pd + (size_t) e * dsz, dsz);
        }
        y.blobs = (uint8_t*) dmalloc(total, bytes, "routed experts");
        CK(cudaMemcpy(y.blobs, stage, total, cudaMemcpyHostToDevice));
    }
    if (stage) CK(cudaFreeHost(stage));
    m.out_norm = f32("output_norm.weight");
    m.head = mat("output.weight");
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("token_embd.weight", s);
        if (!K::embed_type_supported((int) t->type)) {
            err = std::string("token_embd.weight is ") + t->type_name() + ", which the embedding kernel does not take";
            return false;
        }
        m.embd_type = (int) t->type;
        m.embd_row = row_bytes(m.embd_type, (int64_t) t->shape[0]);
        m.n_vocab = (int) t->shape[1];
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        if (embd_gpu) {
            uint8_t* d = (uint8_t*) dmalloc(nb, bytes, "token_embd");
            CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
            m.embd = d;
        } else {   // pinned, mapped: a token reads its one row over PCIe
            uint8_t* hp = nullptr;
            CK(cudaHostAlloc((void**) &hp, nb, cudaHostAllocMapped));
            std::memcpy(hp, gm.shard(s).tensor_data(*t), nb);
            void* dp = nullptr;
            CK(cudaHostGetDevicePointer(&dp, hp, 0));
            m.embd = (const uint8_t*) dp;
        }
    }
    if (m.head.n_out != m.n_vocab) {
        err = "output.weight and token_embd.weight differ in vocabulary";
        return false;
    }
    int n_swa = 0;
    for (const Layer& y : m.layers) n_swa += y.swa;
    std::fprintf(stderr, "strata-flint: %s: %d layers (%d window, %d dense), %d experts top-%d, %.2f GiB on the GPU in %.1f s "
                 "(token embedding in %s)\n", path.c_str(), m.n_layers, n_swa, m.n_dense, m.n_expert, m.top_k,
                 (double) bytes / (1 << 30), ms_since(t0) / 1000.0, embd_gpu ? "VRAM" : "pinned host memory");
    return true;
}

// ------------------------------------------------------------------------------------------------ the engine
float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    int ctx = 32768, tmax = 512, R = 2048;   // context, prompt chunk, window ring cells
    int64_t pos = 0;
    float *x, *h, *y, *qkv, *q, *att, *logits_r, *eparts, *sg, *su, *shh, *sy, *ewts, *logits, *hmain, *dg, *du, *dh;
    void *q8, *xq, *escr;
    int32_t *ids, *grp_start, *n_groups, *ent_dst, *ent_tok, *tok, *pos0, *pick;
    int32_t* hpin = nullptr;
    unsigned long long* grp_ptr;
    float *pacc, *pm, *pl, *amv;
    int32_t* ami;
    // decode attention splits: window layers 17 (513 cells -> one 32-cell tile per split), full layers 64; a split's
    // latency is its tiles, so more splits are faster (RTX 4090, tg128 after 512: 32/16 265.1, 64/17 271.9 tok/s)
    int ns_full = std::getenv("FLINT_NS_FULL") ? std::atoi(std::getenv("FLINT_NS_FULL")) : 64;
    int ns_swa = std::getenv("FLINT_NS_SWA") ? std::atoi(std::getenv("FLINT_NS_SWA")) : 17;
    std::vector<__half*> kc, vc;
    std::vector<int> cap;
    int ld_qkv = 0;
    // the prompt
    strata::prefill::Gemm gemm;
    cublasHandle_t blas = nullptr;
    size_t att_floats = (size_t) 16 << 20;
    float* att_s = nullptr;
    __half *att_p16 = nullptr, *k16 = nullptr, *v16 = nullptr, *q16 = nullptr, *x16 = nullptr;
    static constexpr int kSplitK = 16;
    float* sk_part = nullptr;
    bool use_mmq = false;
    std::unique_ptr<MMQ::Context> mctx;
    int32_t *rcnt = nullptr, *rcur = nullptr, *rbounds = nullptr, *rmax = nullptr, *ident = nullptr, *rinv = nullptr;
    void *mq_x = nullptr, *mq_hq = nullptr;
    float *mq_gu = nullptr, *mq_h = nullptr;
    // decode
    bool use_graph = true;
    cudaGraphExec_t g1 = nullptr;
    bool q8rows = std::getenv("FLINT_Q8ROWS") != nullptr && std::string(std::getenv("FLINT_Q8ROWS")) != "0";
    // the fused decode kernels (k_attn_dec, k_router, k_add_rms_q8, k_moe_out_q8) and the shared expert beside the
    // routed ones (a second stream: a fork / join in the decode graph); FLINT_FUSED=0 / FLINT_FORK=0 turn them off
    bool fused = std::getenv("FLINT_FUSED") == nullptr || std::string(std::getenv("FLINT_FUSED")) != "0";
    bool fork = std::getenv("FLINT_FORK") == nullptr || std::string(std::getenv("FLINT_FORK")) != "0";
    cudaStream_t st2 = nullptr;
    cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
    unsigned int *aticket = nullptr, *rticket = nullptr;
    float2* rope_tab = nullptr;   // [ctx][64] (cos, sin) of the window layers' rotary angles, computed in double at init
    bool attn2 = std::getenv("FLINT_ATTN1") == nullptr;   // k_attn_dec2 (FLINT_ATTN1: the first fused kernel)
    bool fa = std::getenv("FLINT_FA") == nullptr || std::string(std::getenv("FLINT_FA")) != "0";   // the prompt's flash attention
    void* q8s = nullptr;   // the shared expert's hidden q8_1 image (its own: the branch runs beside the routed experts)
    // FLINT_FUSED2=1 (opt-in): one token's chain after attention in three launches (fld::k_gemv_res_norm, k_exp_gu,
    // k_exp_down) where the types allow (wo Q8_0 / Q4_K; experts Q4_K gate/up, Q4_K / Q8_0 down).  Measured slower on an
    // RTX 4090 with Flint's Q4_K_M (258.8 vs 271.3 tok/s, tg128): its warp-per-row dots lose more than the launches save.
    bool fused2 = std::getenv("FLINT_FUSED2") != nullptr && std::string(std::getenv("FLINT_FUSED2")) != "0";
    // FLINT_GU=0 / FLINT_R256=0: the routed gate/up + SwiGLU + q8_1 in one launch (fld::k_exp_gu, 9.99 us vs 9.48 + 1.44
    // for native_gu + swiglu_q8_1) and the router over 256 blocks (k_router256) turned off
    bool exp_gu = std::getenv("FLINT_GU") == nullptr || std::string(std::getenv("FLINT_GU")) != "0";
    bool r256 = std::getenv("FLINT_R256") == nullptr || std::string(std::getenv("FLINT_R256")) != "0";
    unsigned int *wticket = nullptr, *dticket = nullptr;
    void* hq_exp = nullptr;   // the picks' hidden q8_1 images (K x FF / 32 blocks)
    bool shgu = std::getenv("FLINT_SHGU") == nullptr || std::string(std::getenv("FLINT_SHGU")) != "0";
    bool xq_ready = false, xq_post = false;

    void init(const Model& model, int ctx_, int tmax_) {
        m = &model;
        ctx = ctx_;
        tmax = std::max(tmax_, 8);
        R = std::max(2048, (m->win + tmax + 255) / 256 * 256);
        const size_t T = (size_t) tmax, N = (size_t) m->N, Kp = (size_t) m->top_k, FF = (size_t) m->FF;
        const size_t QD = (size_t) m->H * m->HD, KD = (size_t) m->HKV * m->HD;
        ld_qkv = (int) (QD + 2 * KD);
        const int max_in = std::max({(int) QD, m->FFD, m->FS, m->N});
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&st2, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming));
        {
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            std::fprintf(stderr, "strata-flint: %.2f GiB of VRAM free before the engine's buffers (%.2f GiB total)\n",
                         fr / 1073741824.0, tot / 1073741824.0);
        }
        x = dalloc(T * N); h = dalloc(T * N); y = dalloc(T * N); hmain = dalloc(T * N);
        qkv = dalloc(T * ld_qkv); q = dalloc(T * QD); att = dalloc(T * QD);
        logits_r = dalloc(T * m->n_expert);
        eparts = dalloc(T * Kp * N);
        sg = dalloc(T * m->FS); su = dalloc(T * m->FS); shh = dalloc(T * m->FS); sy = dalloc(T * N);
        // the dense layers' gate / up / hidden: their own buffers only where the MoE prompt buffers cannot hold them
        // (the first layers' MLP and the MoE never run at once: dg, du in eparts, dh in mq_gu - see below)
        const bool dense_alias = (size_t) Kp * N >= 2 * (size_t) m->FFD && 2 * (size_t) Kp * FF >= (size_t) m->FFD;
        ewts = dalloc(T * Kp);
        logits = dalloc((size_t) 8 * m->n_vocab);
        CK(cudaMalloc(&q8, K::native_q8_1_bytes(max_in, 1) * T));
        CK(cudaMalloc(&xq, K::native_q8_1_bytes((int) N, 1) * T));
        CK(cudaMalloc(&q8s, K::native_q8_1_bytes(m->FS, 1) * 8));
        aticket = dalloc_t<unsigned int>((size_t) 8 * m->HKV);
        rticket = dalloc_t<unsigned int>(1);
        {
            std::vector<float2> tab((size_t) ctx * (m->HD / 2));
            for (int p = 0; p < ctx; ++p)
                for (int i = 0; i < m->HD / 2; ++i) {
                    const double th = (double) p * std::pow((double) m->rope_base, -2.0 * i / (double) m->HD);
                    tab[(size_t) p * (m->HD / 2) + i] = make_float2((float) std::cos(th), (float) std::sin(th));
                }
            CK(cudaMalloc(&rope_tab, tab.size() * sizeof(float2)));
            CK(cudaMemcpy(rope_tab, tab.data(), tab.size() * sizeof(float2), cudaMemcpyHostToDevice));
        }
        wticket = dalloc_t<unsigned int>(1);
        dticket = dalloc_t<unsigned int>(1);
        CK(cudaMalloc(&hq_exp, K::native_q8_1_bytes(m->FF, 1) * (size_t) m->top_k));
        fused = fused && m->N == 2048;
        x16 = dalloc_t<__half>(T * (size_t) max_in);
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-flint: cuBLAS handle failed\n");
            std::exit(1);
        }
        att_s = dalloc(att_floats);
        att_p16 = dalloc_t<__half>(att_floats);
        k16 = dalloc_t<__half>(KD * (size_t) R);
        v16 = dalloc_t<__half>(KD * (size_t) R);
        q16 = dalloc_t<__half>(T * QD);
        sk_part = dalloc((size_t) kSplitK * T * m->n_expert);
        use_mmq = MMQ::built() && std::getenv("FLINT_NO_MMQ") == nullptr;
        for (const Layer& L : m->layers)
            if (L.moe)
                use_mmq = use_mmq && MMQ::supported(L.L.gu_type) && MMQ::supported(L.L.d_type) &&
                          MMQ::fits(L.L.gu_type, 2 * m->FF) && MMQ::fits(L.L.d_type, m->N);
        if (use_mmq) {
            const int64_t ne = (int64_t) T * Kp;
            mctx = std::make_unique<MMQ::Context>();
            rcnt = dalloc_t<int32_t>((size_t) m->n_expert);
            rcur = dalloc_t<int32_t>((size_t) m->n_expert);
            rbounds = dalloc_t<int32_t>((size_t) m->n_expert + 1);
            rmax = dalloc_t<int32_t>(1);
            ident = dalloc_t<int32_t>((size_t) ne);
            rinv = dalloc_t<int32_t>((size_t) ne);
            CK(cudaDeviceSynchronize());
            MMQ::iota(ident, ne, st);
            CK(cudaMalloc(&mq_x, std::max(MMQ::q8_bytes(ne, m->N), MMQ::q8_bytes((int64_t) T, max_in))));
            dbounds = dalloc_t<int32_t>(2);
            CK(cudaMalloc(&mq_hq, MMQ::q8_bytes(ne, m->FF)));
            mq_gu = dalloc((size_t) ne * 2 * FF);
            mq_h = dalloc((size_t) ne * FF);
        }
        if (use_mmq && dense_alias) {
            dg = eparts;
            du = eparts + T * (size_t) m->FFD;
            dh = mq_gu;
        } else {
            dg = dalloc(T * m->FFD); du = dalloc(T * m->FFD); dh = dalloc(T * m->FFD);
        }
        std::fprintf(stderr, "strata-flint: prompt experts on %s\n", use_mmq ? "MMQ (int8)" : "the native grouped kernels");
        {
            std::string gerr;
            int64_t big = 0;
            for (const Layer& L : m->layers)
                for (const Mat* w : {&L.qkv, &L.wq, &L.wk, &L.wv, &L.wo, &L.gate, &L.up, &L.down, &L.sh_gate, &L.sh_up, &L.sh_down})
                    big = std::max<int64_t>(big, (int64_t) w->n_in * w->n_out);
            if (!gemm.init(st, big, gerr)) {
                std::fprintf(stderr, "strata-flint: prompt GEMM: %s\n", gerr.c_str());
                std::exit(1);
            }
        }
        // the grouped expert kernels' scratch: decode rows only when MMQ reads the prompt's experts
        CK(cudaMalloc(&escr, K::native_expert_scratch_bytes((int64_t) ((use_mmq ? 8 : T) * Kp), (int64_t) FF)));
        ids = dalloc_t<int32_t>(T * Kp);
        grp_ptr = dalloc_t<unsigned long long>(std::min<size_t>(T * Kp, (size_t) m->n_expert) + 1);
        grp_start = dalloc_t<int32_t>(std::min<size_t>(T * Kp, (size_t) m->n_expert) + 1);
        n_groups = dalloc_t<int32_t>(1);
        ent_dst = dalloc_t<int32_t>(T * Kp);
        ent_tok = dalloc_t<int32_t>(T * Kp);
        tok = dalloc_t<int32_t>(T);
        pos0 = dalloc_t<int32_t>(1);
        pick = dalloc_t<int32_t>(8);
        amv = dalloc((size_t) kArgBlocks * 8);
        ami = dalloc_t<int32_t>((size_t) kArgBlocks * 8);
        CK(cudaHostAlloc((void**) &hpin, 64 * sizeof(int32_t), cudaHostAllocDefault));
        const int nsm = std::max(ns_full, ns_swa);
        pacc = dalloc((size_t) 8 * m->H * nsm * m->HD);
        pm = dalloc((size_t) 8 * m->H * nsm);
        pl = dalloc((size_t) 8 * m->H * nsm);
        size_t kvb = 0;
        {
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            std::fprintf(stderr, "strata-flint: %.2f GiB of VRAM free before the K / V caches\n", fr / 1073741824.0);
        }
        for (const Layer& L : m->layers) {
            const int c = L.swa ? R : ctx;
            cap.push_back(c);
            kc.push_back(dalloc_t<__half>(KD * (size_t) c));
            vc.push_back(dalloc_t<__half>(KD * (size_t) c));
            kvb += 2 * KD * (size_t) c * 2;
        }
        CK(cudaDeviceSynchronize());
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-flint: context %d cells (K / V %.2f GiB fp16, window ring %d cells), prompt chunk %d, "
                     "%.2f GiB of VRAM free\n", ctx, kvb / 1073741824.0, R, tmax, fr / 1073741824.0);
    }

    void reset() { pos = 0; }

    // ---- products
    static constexpr int kGemmT = 8;
    void quant(const float* xin, int n, int T, void* qo) { K::native_quantize_q8_1(xin, qo, n * T, 1, st); }
    void prep(const float* xin, int n, int T, void* qo) {
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) n * T), kThreads, 0, st>>>(xin, x16, (int64_t) n * T);
        else quant(xin, n, T, qo);
    }
    /// y [T][n_out] = W x from x's q8_1 image (T <= 8), or from x16 (a prompt chunk: FP16 GEMM).
    void apply(const Mat& w, const void* qi, float* out, int T) {
        if (T > kGemmT) gemm.native((const uint16_t*) x16, w.type, w.d, out, T, w.n_out, w.n_in);
        else mmq(w, qi, out, T);
    }
    /// y [T][n_out] = W x [T][n_in] (f32 rows): T <= 8 on the native MMVQ; a prompt chunk on llama.cpp's MMQ (int8
    /// activations with a scale per 32 values, as llama.cpp runs these products: Flint's later layers carry activations
    /// in the thousands, past what the FP16 GEMM's inputs hold), or the FP16 GEMM where MMQ does not take the type.
    void mm(const Mat& w, const float* xin, float* out, int T) {
        if (T > kGemmT && dense_mmq(w)) {
            MMQ::quantize(xin, nullptr, mq_x, w.type, w.n_in, w.n_in, T, st);
            MMQ::Product p;
            p.w = w.d; p.type = w.type; p.w_rows = w.n_out; p.w_cols = w.n_in; p.expert_bytes = 0;
            p.n = 1; p.xq = mq_x; p.bounds = dbounds; p.ids = ident; p.total_rows = T; p.max_rows = T;
            p.dst = out; p.ld_dst = w.n_out;
            mctx->run(p, st);
            return;
        }
        prep(xin, w.n_in, T, q8);
        apply(w, q8, out, T);
    }
    bool dense_mmq(const Mat& w) const {
        return use_mmq && !no_dense_mmq && MMQ::supported(w.type) && MMQ::fits(w.type, w.n_out) && w.n_in % 256 == 0;
    }
    bool no_dense_mmq = std::getenv("FLINT_DENSE_FP16") != nullptr;
    int32_t* dbounds = nullptr;   // {0, T} on the device: one matrix as MMQ's single expert
    int dbounds_T = -1;
    bool q8_small(const Mat& w, const void* qi, float* out, int T, cudaStream_t ss = nullptr) {
        if (!ss) ss = st;
        if (!q8rows || w.type != 8 || T < 1 || T > 8 || w.n_in % 32 != 0) return false;
        const int nb = w.n_in / 32;
        if (nb > 64) return false;
        const int NJ = nb <= 32 ? 1 : 2;
#define FL_Q8(NCV)                                                                                         \
    case NCV:                                                                                              \
        if (NJ == 1) q36k::launch_q8_rows<NCV, 1, false>(w.d, nullptr, qi, out, nb, w.n_out, ss);          \
        else q36k::launch_q8_rows<NCV, 2, false>(w.d, nullptr, qi, out, nb, w.n_out, ss);                  \
        break;
        switch (T) { FL_Q8(1) FL_Q8(2) FL_Q8(3) FL_Q8(4) FL_Q8(5) FL_Q8(6) FL_Q8(7) FL_Q8(8) }
#undef FL_Q8
        return true;
    }
    void mmq(const Mat& w, const void* qi, float* out, int T, cudaStream_t ss = nullptr) {
        if (!ss) ss = st;
        if (q8_small(w, qi, out, T, ss)) return;
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) qi + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in, w.n_out, nc, ss);
        }
    }
    /// logits [T][E] = router . h, the router in F32 or BF16.
    void router_mm(const Layer& L, const float* xin, float* out, int T) {
        if (!L.router_bf16) { f32mm(L.router, xin, out, m->N, m->n_expert, T); return; }
        const int n_in = m->N, n_out = m->n_expert;
        const int gx = (n_out + 63) / 64, gy = (T + 63) / 64;
        const int want = std::max(1, std::min(kSplitK, 128 / std::max(1, gx * gy)));
        const int kz = (n_in / want + 15) / 16 * 16;
        const int splits = (n_in + kz - 1) / kz;
        float* dst = splits > 1 ? sk_part : out;
        k_sgemm_nt<4, true><<<dim3(gx, gy, splits), 256, 0, st>>>(L.router, xin, dst, n_in, n_out, T, n_out, kz);
        if (splits > 1) k_sum_splits<<<grid1((int64_t) T * n_out), kThreads, 0, st>>>(sk_part, out, (int64_t) T * n_out, splits);
    }
    void f32mm(const float* W, const float* xin, float* out, int n_in, int n_out, int T) {
        if (T <= kGemmT) {
            k_gemv_f32<<<dim3(rows8(n_out), T), kThreads, 0, st>>>(W, xin, out, n_in, n_out);
            return;
        }
        const int tm = n_out > 64 ? 4 : 1;
        const int gx = (n_out + 63) / 64, gy = (T + 16 * tm - 1) / (16 * tm);
        const int want = std::max(1, std::min(kSplitK, 128 / std::max(1, gx * gy)));
        const int kz = (n_in / want + 15) / 16 * 16;
        const int splits = (n_in + kz - 1) / kz;
        float* dst = splits > 1 ? sk_part : out;
        if (tm == 4) k_sgemm_nt<4><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
        else k_sgemm_nt<1><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
        if (splits > 1) k_sum_splits<<<grid1((int64_t) T * n_out), kThreads, 0, st>>>(sk_part, out, (int64_t) T * n_out, splits);
    }
    void prep_in(int T) {
        if (xq_ready && T <= kGemmT) {
            xq_ready = false;
            return;
        }
        xq_ready = false;
        prep(h, m->N, T, xq);
    }
    /// x += y; h = rms(x) * w; for T <= 8 also h's q8_1 image in xq (xq_post).
    void add_rms(const float* w, float* hout, int T) {
        xq_post = false;
        if (T <= kGemmT && hout == h) {
            if (fused) flk::k_add_rms_q8<2><<<T, 1024, 0, st>>>(x, y, w, h, (QB*) xq, m->eps);
            else q36k::k_add_rms_q<<<T, 256, 0, st>>>(x, y, w, h, (QB*) xq, m->N, m->eps);
            xq_post = true;
            return;
        }
        k_add_rms<<<T, kThreads, 0, st>>>(x, y, w, hout, m->N, m->eps);
    }
    /// The shared expert's gate + up + SwiGLU + q8_1 image (into q8) for T <= 4 rows in one launch, else false.
    bool shared_gu(const Layer& L, int T, cudaStream_t ss, void* qout) {
        const int FS = m->FS;
        const Mat &a = L.sh_gate, &b = L.sh_up;
        if (!shgu || T < 1 || T > 4 || a.type != 8 || b.type != 8 || a.n_in != b.n_in || a.n_in % 32 != 0 ||
            a.n_in / 32 > 64 || a.n_in / 32 <= 32 || FS % 32 != 0)
            return false;
        const int nb = a.n_in / 32;
        const dim3 grid((unsigned) (FS / 32));
#define FL_SH(NCV)                                                                                                    \
    case NCV:                                                                                                         \
        q36k::k_sh_gu<NCV, 2, false><<<grid, 1024, 0, ss>>>(a.d, nullptr, b.d, nullptr, (const QB*) xq, (QB*) qout, nb, FS); \
        break;
        switch (T) { FL_SH(1) FL_SH(2) FL_SH(3) FL_SH(4) }
#undef FL_SH
        return true;
    }

    // ---- attention
    /// true: the residual add and the post-attention norm are done too (h and its q8_1 image in xq).
    bool attention(const Layer& L, size_t li, int T, int64_t p0) {
        const int H = m->H, HKV = m->HKV, HD = m->HD;
        const size_t QD = (size_t) H * HD, KD = (size_t) HKV * HD;
        if (T <= kGemmT) prep_in(T);
        // (a prompt chunk: each product quantizes h itself, mm)
        auto proj = [&](const Mat& w, float* out) {
            if (T <= kGemmT) apply(w, xq, out, T);
            else mm(w, h, out, T);
        };
        const float *qs, *ks, *vs;
        int ldq, ldk, ldv;
        if (L.fused_qkv) {
            proj(L.qkv, qkv);
            qs = qkv; ks = qkv + QD; vs = qkv + QD + KD;
            ldq = ldk = ldv = ld_qkv;
        } else {
            float* bq = qkv;
            float* bk = qkv + (size_t) tmax * QD;
            float* bv = bk + (size_t) tmax * KD;
            proj(L.wq, bq);
            proj(L.wk, bk);
            proj(L.wv, bv);
            qs = bq; ks = bk; vs = bv;
            ldq = (int) QD; ldk = ldv = (int) KD;
        }
        const float scale = 1.0f / sqrtf((float) HD);
        if (T == 1 && fused) {   // (more rows would read each other's new cells before they are written)
            const int NS = L.swa ? ns_swa : ns_full;
            if (attn2)
                flk::k_attn_dec2<<<dim3(NS, HKV, 1), 256, 0, st>>>(qs, ldq, ks, ldk, vs, ldv, L.q_norm, L.k_norm, kc[li], vc[li], H,
                                                                  cap[li], L.swa ? 1 : 0, L.swa ? m->win : 0,
                                                                  L.rope ? rope_tab : nullptr, pos0, m->eps, scale, NS, pacc, pm, pl,
                                                                  L.sinks, att, (QB*) q8, aticket);
            else
                flk::k_attn_dec<<<dim3(NS, HKV, 1), 256, 0, st>>>(qs, ldq, ks, ldk, vs, ldv, L.q_norm, L.k_norm, kc[li], vc[li], H,
                                                                 cap[li], L.swa ? 1 : 0, L.swa ? m->win : 0, L.rope ? 1 : 0, pos0,
                                                                 m->rope_base, m->eps, scale, NS, pacc, pm, pl, L.sinks, att, (QB*) q8,
                                                                 aticket);
            if (fused2 && (L.wo.type == 8 || L.wo.type == 12) && L.wo.n_out == m->N && m->N % 8 == 0) {
                if (L.wo.type == 8)
                    fld::k_gemv_res_norm<8><<<m->N / 8, 256, 0, st>>>(L.wo.d, (const QB*) q8, L.wo.n_in, x, L.ffn_norm, h, (QB*) xq,
                                                                      m->N, m->eps, wticket);
                else
                    fld::k_gemv_res_norm<12><<<m->N / 8, 256, 0, st>>>(L.wo.d, (const QB*) q8, L.wo.n_in, x, L.ffn_norm, h, (QB*) xq,
                                                                       m->N, m->eps, wticket);
                xq_post = true;
                return true;
            }
            mmq(L.wo, q8, y, T);
            return false;
        }
        flk::k_prep<<<dim3(H + HKV, T), HD, 0, st>>>(qs, ldq, ks, ldk, vs, ldv, L.q_norm, L.k_norm, q, kc[li], vc[li], H, cap[li],
                                                    L.swa ? 1 : 0, L.rope ? 1 : 0, pos0, m->rope_base, m->eps);
        if (T <= kGemmT) {
            const int NS = L.swa ? ns_swa : ns_full;
            flk::k_dattn<<<dim3(NS, HKV, T), 256, 0, st>>>(q, kc[li], vc[li], H, cap[li], L.swa ? 1 : 0, L.swa ? m->win : 0,
                                                          pos0, scale, NS, pacc, pm, pl);
            flk::k_dattn_merge<<<T * H, HD, 0, st>>>(pacc, pm, pl, L.sinks, H, NS, att, (QB*) q8);
            mmq(L.wo, q8, y, T);
            return false;
        }
        // a prompt chunk: scores = Q K^T and out = P V as FP16 GEMMs (FP32 accumulation) per KV head over its 8 query
        // heads, the masked softmax with the sink in between; the queries in slices whose scores fit
        int lo = 0, P = (int) p0 + T, kstride = cap[li];
        const __half *Kb = kc[li], *Vb = vc[li];
        if (L.swa) {
            lo = (int) std::max<int64_t>(0, p0 - (m->win - 1));
            P = (int) (p0 + T - lo);
            flk::k_kv_gather<<<grid1((int64_t) HKV * P * HD / 2), kThreads, 0, st>>>(kc[li], vc[li], k16, v16, cap[li], lo, P, HKV);
            Kb = k16; Vb = v16; kstride = P;
        }
        k_to_f16<<<grid1((int64_t) T * QD), kThreads, 0, st>>>(q, q16, (int64_t) T * QD);
        if (fa) {   // one flash-attention launch: the keys from the cache (full layers) or the gathered window
            ffa::k_fa_prompt<<<dim3((T + 7) / 8, HKV), 128, 0, st>>>(q16, Kb, Vb, kstride, lo, T, H, (int) p0,
                                                                    L.swa ? m->win : 0, scale, L.sinks, att);
            mm(L.wo, att, y, T);
            return false;
        }
        const int ldp = (P + 7) / 8 * 8;
        const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (att_floats / ((size_t) H * ldp))));
        const int G = H / HKV;
        const float one = 1.0f, zero = 0.0f;
        for (int t0 = 0; t0 < T; t0 += tq) {
            const int nt = std::min(tq, T - t0);
            for (int kvh = 0; kvh < HKV; ++kvh) {
                const __half* Kh = Kb + (size_t) kvh * kstride * HD;
                const __half* Qh = q16 + ((size_t) t0 * H + kvh * G) * HD;
                float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, HD, &one, Kh, CUDA_R_16F, HD, 0, Qh,
                                               CUDA_R_16F, (int) QD, HD, &zero, Sh, CUDA_R_32F, ldp, (long long) nt * ldp, G,
                                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                    std::fprintf(stderr, "strata-flint: attention scores GEMM failed\n");
                    std::exit(1);
                }
            }
            flk::k_softmax_sink<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldp, nt, P, lo, (int) p0 + t0,
                                                                  L.swa ? m->win : 0, scale, L.sinks);
            for (int kvh = 0; kvh < HKV; ++kvh) {
                const __half* Vh = Vb + (size_t) kvh * kstride * HD;
                const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldp;
                float* Oh = att + ((size_t) t0 * H + kvh * G) * HD;
                if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, nt, P, &one, Vh, CUDA_R_16F, HD, 0, Sh,
                                               CUDA_R_16F, ldp, (long long) nt * ldp, &zero, Oh, CUDA_R_32F, (int) QD, HD, G,
                                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                    std::fprintf(stderr, "strata-flint: attention values GEMM failed\n");
                    std::exit(1);
                }
            }
        }
        mm(L.wo, att, y, T);
        return false;
    }

    // ---- the FFN blocks
    void dense(const Layer& L, int T, const float* wn, float* hout) {
        const int FFD = m->FFD;
        if (T <= kGemmT) {
            if (!xq_post) quant(h, m->N, T, xq);
            mmq(L.gate, xq, dg, T);
            mmq(L.up, xq, du, T);
            K::native_swiglu_quantize_q8_1(dg, du, q8, FFD, T, st);
            mmq(L.down, q8, y, T);
        } else {
            mm(L.gate, h, dg, T);
            mm(L.up, h, du, T);
            k_swiglu<<<grid1((int64_t) T * FFD), kThreads, 0, st>>>(dg, du, dh, (int64_t) T * FFD);
            mm(L.down, dh, y, T);
        }
        xq_ready = false;
        if (T <= kGemmT && hout == h) {
            if (fused) flk::k_add_rms_q8<2><<<T, 1024, 0, st>>>(x, y, wn, h, (QB*) xq, m->eps);
            else q36k::k_add_rms_q<<<T, 256, 0, st>>>(x, y, wn, h, (QB*) xq, m->N, m->eps);
            xq_ready = true;
        } else {
            k_add_rms<<<T, kThreads, 0, st>>>(x, y, wn, hout, m->N, m->eps);
        }
    }

    void moe(const Layer& L, int T, const float* wn, float* hout) {
        const int N = m->N, Kp = m->top_k, FF = m->FF, FS = m->FS, E = m->n_expert;
        const int64_t n_ent = (int64_t) T * Kp;
        const int32_t* prow = nullptr;
        if (T <= kGemmT) {
            if (!xq_post) quant(h, N, T, xq);
            // the shared expert: on st2 beside the router and the routed experts (fork / join), or in line
            cudaStream_t ss = fork ? st2 : st;
            if (fork) {
                CK(cudaEventRecord(ev_fork, st));
                CK(cudaStreamWaitEvent(st2, ev_fork, 0));
            }
            if (!shared_gu(L, T, ss, q8s)) {
                mmq(L.sh_gate, xq, sg, T, ss);
                mmq(L.sh_up, xq, su, T, ss);
                K::native_swiglu_quantize_q8_1(sg, su, q8s, FS, T, ss);
            }
            mmq(L.sh_down, q8s, sy, T, ss);
            if (fork) CK(cudaEventRecord(ev_join, st2));
            const bool f2 = fused2 && fused && T == 1 && N == 2048 && E == 256 && L.L.gu_type == 12 &&
                            (L.L.d_type == 12 || L.L.d_type == 8) && FF % 256 == 0 && N % 16 == 0;
            if (f2) {
                if (Kp == 6)
                    flk::k_router1<6, 16><<<E / 4, dim3(32, 4), 0, st>>>(L.router, h, logits_r, E, ids, ewts, L.blobs, L.stride, grp_ptr,
                                                                        grp_start, n_groups, ent_dst, ent_tok, rticket);
                else
                    flk::k_router1<8, 16><<<E / 4, dim3(32, 4), 0, st>>>(L.router, h, logits_r, E, ids, ewts, L.blobs, L.stride, grp_ptr,
                                                                        grp_start, n_groups, ent_dst, ent_tok, rticket);
                fld::k_exp_gu<12><<<dim3(FF / 32, Kp), 512, 0, st>>>(ids, L.blobs, L.stride, L.L.up_off, (const QB*) xq, N, FF,
                                                                     (QB*) hq_exp);
                if (fork) CK(cudaStreamWaitEvent(st, ev_join, 0));
                const bool mkq = hout == h;
                if (L.L.d_type == 12)
                    fld::k_exp_down<12><<<N / 16, 512, 0, st>>>(ids, ewts, Kp, L.blobs, L.stride, L.L.down_off, (const QB*) hq_exp, FF,
                                                                sy, x, wn, hout, mkq ? (QB*) xq : nullptr, N, m->eps, dticket);
                else
                    fld::k_exp_down<8><<<N / 16, 512, 0, st>>>(ids, ewts, Kp, L.blobs, L.stride, L.L.down_off, (const QB*) hq_exp, FF,
                                                               sy, x, wn, hout, mkq ? (QB*) xq : nullptr, N, m->eps, dticket);
                xq_ready = mkq;
                return;
            }
            if (fused && T == 1 && N == 2048 && E == 256 && (r256 || L.router_bf16)) {
#define FL_R256(KK, BF) flk::k_router256<KK, BF><<<E, 128, 0, st>>>(L.router, h, logits_r, ids, ewts, L.blobs, L.stride, grp_ptr, \
                                                                    grp_start, n_groups, ent_dst, ent_tok, rticket)
                if (L.router_bf16) { if (Kp == 6) FL_R256(6, true); else FL_R256(8, true); }
                else { if (Kp == 6) FL_R256(6, false); else FL_R256(8, false); }
#undef FL_R256
            } else if (fused && T == 1 && N == 2048 && E == 256) {
                if (Kp == 6)
                    flk::k_router1<6, 16><<<E / 4, dim3(32, 4), 0, st>>>(L.router, h, logits_r, E, ids, ewts, L.blobs, L.stride, grp_ptr,
                                                                        grp_start, n_groups, ent_dst, ent_tok, rticket);
                else
                    flk::k_router1<8, 16><<<E / 4, dim3(32, 4), 0, st>>>(L.router, h, logits_r, E, ids, ewts, L.blobs, L.stride, grp_ptr,
                                                                        grp_start, n_groups, ent_dst, ent_tok, rticket);
            } else if (fused) {
                if (Kp == 6)
                    flk::k_router<6><<<dim3(E / 8, T), dim3(32, 8), 0, st>>>(L.router, h, logits_r, N, E, T, ids, ewts, L.blobs,
                                                                            L.stride, grp_ptr, grp_start, n_groups, ent_dst,
                                                                            ent_tok, rticket);
                else
                    flk::k_router<8><<<dim3(E / 8, T), dim3(32, 8), 0, st>>>(L.router, h, logits_r, N, E, T, ids, ewts, L.blobs,
                                                                            L.stride, grp_ptr, grp_start, n_groups, ent_dst,
                                                                            ent_tok, rticket);
            } else {
                router_mm(L, h, logits_r, T);
                if (Kp == 6)
                    flk::k_route_group<6><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, E, L.blobs, L.stride, grp_ptr, grp_start,
                                                                     n_groups, ent_dst, ent_tok);
                else
                    flk::k_route_group<8><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, E, L.blobs, L.stride, grp_ptr, grp_start,
                                                                     n_groups, ent_dst, ent_tok);
            }
            if (exp_gu && fused && T == 1 && L.L.gu_type == 12 && FF % 32 == 0 && N % 256 == 0) {
                // gate/up + SwiGLU + the hidden's q8_1 image in one launch, then Strata's down kernel over those images
                fld::k_exp_gu_grp<12><<<dim3(FF / 32, Kp), 512, 0, st>>>(grp_ptr, L.L.up_off, (const QB*) xq, N, FF, (QB*) hq_exp);
                K::native_expert_down_q8(L.L, grp_ptr, grp_start, n_groups, ent_dst, std::min<int64_t>(n_ent, E), hq_exp,
                                         eparts, st);
            } else {
                K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, std::min<int64_t>(n_ent, E), n_ent,
                                         xq, escr, eparts, st);
            }
            if (fork) CK(cudaStreamWaitEvent(st, ev_join, 0));
            if (dbg) {
                dump_n("  experts", 0, eparts, (int) n_ent, N);
                dump_n("  shared", 0, sy, T, N);
            }
            const bool mkq = hout == h;
            if (fused) flk::k_moe_out_q8<2><<<T, 1024, 0, st>>>(x, eparts, ewts, Kp, sy, wn, hout, m->eps, mkq ? (QB*) xq : nullptr);
            else flk::k_moe_out_small<<<T, 1024, 0, st>>>(x, eparts, nullptr, ewts, Kp, sy, wn, hout, N, m->eps, mkq ? (QB*) xq : nullptr);
            xq_ready = mkq;
            return;
        }
        router_mm(L, h, logits_r, T);
        if (Kp == 6) flk::k_route<6><<<(T + 7) / 8, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, E);
        else flk::k_route<8><<<(T + 7) / 8, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, E);
        if (use_mmq) {
            // the experts on llama.cpp's MMQ (int8, the weights read once per expert): the entries sorted by expert on the
            // device, gate+up as one [2 FF] product per expert, SwiGLU, down into rows the combine reads through rinv
            CK(cudaMemsetAsync(rcnt, 0, (size_t) E * 4, st));
            k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, rcnt);
            k_route_scan<<<1, 32, 0, st>>>(rcnt, E, rbounds, rcur, rmax);
            k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, Kp, rcur, ent_dst, ent_tok, rinv);
            CK(cudaMemcpyAsync(hpin + 10, rmax, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            const int64_t maxr = hpin[10];
            MMQ::quantize(h, ent_tok, mq_x, L.L.gu_type, N, N, n_ent, st);
            MMQ::Product gu;
            gu.w = L.blobs; gu.type = L.L.gu_type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.expert_bytes = L.stride;
            gu.n = E; gu.xq = mq_x; gu.bounds = rbounds; gu.ids = ident; gu.total_rows = n_ent; gu.max_rows = maxr;
            gu.dst = mq_gu; gu.ld_dst = 2 * FF;
            mctx->run(gu, st);
            MMQ::swiglu(mq_gu, mq_h, n_ent, FF, false, st);
            MMQ::quantize(mq_h, nullptr, mq_hq, L.L.d_type, FF, FF, n_ent, st);
            MMQ::Product dn;
            dn.w = L.blobs + L.L.down_off; dn.type = L.L.d_type; dn.w_rows = N; dn.w_cols = FF;
            dn.expert_bytes = L.stride; dn.n = E; dn.xq = mq_hq; dn.bounds = rbounds; dn.ids = ident;
            dn.total_rows = n_ent; dn.max_rows = maxr; dn.dst = eparts; dn.ld_dst = N;
            mctx->run(dn, st);
            prow = rinv;
        } else {
            k_group<<<1, kThreads, 0, st>>>(ids, (int) n_ent, Kp, E, L.blobs, L.stride, grp_ptr, grp_start, n_groups, ent_dst,
                                            ent_tok);
            quant(h, N, T, xq);
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, std::min<int64_t>(n_ent, E), n_ent, xq,
                                     escr, eparts, st);
        }
        mm(L.sh_gate, h, sg, T);
        mm(L.sh_up, h, su, T);
        k_swiglu<<<grid1((int64_t) T * FS), kThreads, 0, st>>>(sg, su, shh, (int64_t) T * FS);
        mm(L.sh_down, shh, sy, T);
        if (dbg) {
            dump_n("  experts", 0, eparts, (int) n_ent, N);
            dump_n("  shared h", 0, shh, T, FS);
            dump_n("  shared", 0, sy, T, N);
            if (use_mmq) dump_n("  expert h", 0, mq_h, (int) n_ent, FF);
        }
        flk::k_moe_out<<<T, 256, 0, st>>>(x, eparts, prow, ewts, Kp, sy, wn, hout, N, m->eps);
        xq_ready = false;
    }

    /// The model over T rows (device `tok`, positions p0 .. p0 + T - 1, also at *pos0): `heads` rows from the end get
    /// logits (logits row r - (T - heads)) and greedy picks (`pick`).
    void forward(int T, int64_t p0, int heads) {
        const int N = m->N;
        if (T > kGemmT && dbounds && dbounds_T != T) {
            CK(cudaStreamSynchronize(st));   // (the pinned words are reused)
            hpin[20] = 0;
            hpin[21] = T;
            CK(cudaMemcpyAsync(dbounds, hpin + 20, 8, cudaMemcpyHostToDevice, st));
            dbounds_T = T;
        }
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok, T, N, x, st);
        k_rms<<<T, kThreads, 0, st>>>(x, m->layers[0].attn_norm, h, N, m->eps);
        xq_ready = false;
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const Layer& L = m->layers[l];
            const bool last = l + 1 == m->layers.size();
            const float* wn = last ? m->out_norm : m->layers[l + 1].attn_norm;
            float* hout = last ? hmain : h;
            const bool res = attention(L, l, T, p0);
            if (dbg && !res) dump("attn", l, y, T);
            if (!res) add_rms(L.ffn_norm, h, T);
            if (L.moe) moe(L, T, wn, hout);
            else dense(L, T, wn, hout);
            if (dbg) dump("x", l, x, T);
        }
        if (heads > 0) head_rows(hmain + (size_t) (T - heads) * N, heads, logits, pick);
    }
    bool dbg = std::getenv("FLINT_DEBUG") != nullptr;
    /// FLINT_DEBUG: a layer's T rows of n_embd on the host - the largest |value| and the non-finite count, per row group.
    void dump(const char* what, size_t l, const float* d, int T) { dump_n(what, l, d, T, m->N); }
    void dump_n(const char* what, size_t l, const float* d, int T, int n) {
        CK(cudaStreamSynchronize(st));
        std::vector<float> hv((size_t) T * n);
        CK(cudaMemcpy(hv.data(), d, hv.size() * 4, cudaMemcpyDeviceToHost));
        float mx = 0.f;
        size_t bad = 0, first_bad = SIZE_MAX;
        for (size_t i = 0; i < hv.size(); ++i) {
            if (!std::isfinite(hv[i])) { ++bad; if (first_bad == SIZE_MAX) first_bad = i / n; continue; }
            mx = std::max(mx, std::fabs(hv[i]));
        }
        std::fprintf(stderr, "FLINT_DEBUG T %d layer %zu %s: max |v| %.4g, non-finite %zu (first row %lld), row0[0..3] %.5g %.5g %.5g\n",
                     T, l, what, mx, bad, first_bad == SIZE_MAX ? -1LL : (long long) first_bad, hv[0], hv[1], hv[2]);
    }
    void head_rows(const float* hn, int n, float* lg, int32_t* pk) {
        mm(m->head, hn, lg, n);
        k_argmax<<<dim3(kArgBlocks, n), kThreads, 0, st>>>(lg, m->n_vocab, pk, amv, ami, false);
        k_argmax<<<dim3(1, n), kThreads, 0, st>>>(lg, kArgBlocks, pk, amv, ami, true);
    }

    template <class F> void graph_run(cudaGraphExec_t& gx, F body) {
        if (!use_graph) { body(); return; }
        if (!gx) {
            cudaGraph_t graph = nullptr;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            body();
            CK(cudaStreamEndCapture(st, &graph));
            CK(cudaGraphInstantiate(&gx, graph, 0));
            CK(cudaGraphDestroy(graph));
        }
        CK(cudaGraphLaunch(gx, st));
    }

    /// The tokens at the current position: a prompt in chunks (eager), one token from the decode graph.  The last token's
    /// logits stay in `logits` (its greedy pick in `pick`).  on_chunk(done) after every chunk; false ends the read there.
    template <class F> int64_t process(const int32_t* toks, int64_t n, F on_chunk_) {
        auto on_chunk = [&](int64_t done) -> bool {
            if constexpr (std::is_void_v<decltype(on_chunk_(done))>) {
                on_chunk_(done);
                return true;
            } else {
                return on_chunk_(done);
            }
        };
        if (n == 1) {
            CK(cudaStreamSynchronize(st));
            hpin[0] = toks[0];
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            graph_run(g1, [&] { forward(1, pos, 1); });
            pos += 1;
            on_chunk(1);
            return 1;
        }
        for (int64_t i = 0; i < n; i += tmax) {
            const int T = (int) std::min<int64_t>(tmax, n - i);
            CK(cudaStreamSynchronize(st));
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            forward(T, pos, i + T == n ? 1 : 0);
            pos += T;
            CK(cudaStreamSynchronize(st));
            if (!on_chunk(i + T)) return i + T;
        }
        return n;
    }
    int greedy() {
        CK(cudaMemcpyAsync(hpin + 2, pick, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return hpin[2];
    }
    void host_logits(std::vector<float>& out) {
        out.resize((size_t) m->n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
    void warmup() {
        std::vector<int32_t> t(64);
        for (size_t i = 0; i < t.size(); ++i) t[i] = (int32_t) (1000 + 7 * i);
        reset();
        process(t.data(), (int64_t) t.size(), [](int64_t) {});
        for (int i = 0; i < 4; ++i) {
            const int32_t one = greedy();
            process(&one, 1, [](int64_t) {});
        }
        CK(cudaStreamSynchronize(st));
        reset();
    }
};

// ------------------------------------------------------------------------------------------------ sampling, serving
struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    uint64_t seed = 42;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};
int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        const size_t n = s.last_n < 0 ? hist.size() : std::min(hist.size(), (size_t) s.last_n);
        std::vector<int> cnt;
        std::vector<int32_t> seen;
        for (size_t i = hist.size() - n; i < hist.size(); ++i) seen.push_back(hist[i]);
        std::sort(seen.begin(), seen.end());
        for (size_t i = 0; i < seen.size();) {
            size_t j = i;
            while (j < seen.size() && seen[j] == seen[i]) ++j;
            float& v = lg[(size_t) seen[i]];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) (j - i) * s.freq + s.present;
            i = j;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const size_t k = s.top_k > 0 ? std::min<size_t>((size_t) s.top_k, lg.size()) : lg.size();
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + (long) k, idx.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
    idx.resize(k);
    std::vector<double> p(k);
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < k; ++i) sum += p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature);
    size_t keep = k;
    if (s.top_p < 1.f) {
        double c = 0;
        for (size_t i = 0; i < k; ++i) {
            c += p[i] / sum;
            if (c >= s.top_p) { keep = i + 1; break; }
        }
    }
    if (s.min_p > 0.f) {
        size_t j = 0;
        while (j < keep && p[j] / p[0] >= s.min_p) ++j;
        keep = std::max<size_t>(1, j);
    }
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        r -= p[i];
        if (r <= 0) return idx[i];
    }
    return idx[keep - 1];
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) { ++s; continue; }
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

/// Flint's ends of an answer (generation_config.json): <|end_of_text|>, <|user|>, <|observation|>, <|developer|>.
bool is_eos(int t) { return t == 150001 || t == 150003 || t == 150005 || t == 150018; }

struct Serve {
    Engine& E;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::vector<int32_t> live;   // the tokens whose K / V the cache holds (cell i = live[i])
    std::vector<float> lg;
    explicit Serve(Engine& e) : E(e) {}

    void request(const std::string& line) {
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            std::fflush(stdout);
            return;
        }
        char* p = const_cast<char*>(line.c_str()) + 4;
        const long long max_new = std::strtoll(p, &p, 10);
        Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty()) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            return;
        }
        if ((int64_t) ids.size() + max_new > E.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx);
            std::fflush(stdout);
            return;
        }
        for (int32_t t : ids)
            if (t < 0 || t >= E.m->n_vocab) {
                std::printf("ERR a token id is outside the vocabulary\n");
                std::fflush(stdout);
                return;
            }
        stop.store(false);
        // the held cells continue when the request shares a start with them: the longest common start, at least one
        // token fresh, and the window ring still holding the 512 cells before it (it lost them once the old run went
        // more than R - 512 cells past it)
        size_t reuse = 0;
        while (reuse < live.size() && reuse + 1 < ids.size() && live[reuse] == ids[reuse]) ++reuse;
        if (reuse > 0 && (int64_t) live.size() - (int64_t) reuse >= (int64_t) E.R - (E.m->win - 1)) reuse = 0;
        live.resize(reuse);
        E.pos = (int64_t) reuse;
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        bool cancelled = false;
        const int64_t got = E.process(ids.data() + reuse, fresh, [&](int64_t done) -> bool {
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) fresh, ms_since(t0),
                        (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
            std::fflush(stdout);
            if (stop.load() && done < fresh) { cancelled = true; return false; }
            return true;
        });
        live.insert(live.end(), ids.begin() + (long) reuse, ids.begin() + (long) reuse + (long) got);
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reuse, (long long) got);
            std::fflush(stdout);
            return;
        }
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        const auto d0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";
        int nx = smp.needs_logits() ? -1 : E.greedy();
        for (;;) {
            if (smp.needs_logits()) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
            std::printf("T %d\n", nx);
            std::fflush(stdout);
            ++produced;
            hist.push_back(nx);
            if (is_eos(nx)) { finish = "stop"; break; }
            if (produced >= max_new) break;
            if (stop.load()) { finish = "cancel"; break; }
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            E.process(&one, 1, [](int64_t) {});
            live.push_back(nx);
            if (!smp.needs_logits()) nx = E.greedy();
        }
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms, decode_ms, finish,
                    reuse, (long long) fresh);
        std::fflush(stdout);
        std::fprintf(stderr, "strata-flint serve: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s), %lld "
                     "generated in %.0f ms (%.1f tok/s, %s)\n", ids.size(), reuse, (long long) fresh, prompt_ms,
                     prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, finish);
    }

    int run() {
        std::printf("READY %d stop\n", E.ctx);
        std::fflush(stdout);
        std::thread reader([&] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop.store(true); continue; }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        reader.detach();
        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (lines.empty()) return 0;
                line = lines.front();
                lines.pop_front();
            }
            if (line.empty() || line.rfind("BYIELD", 0) == 0) continue;
            request(line);
        }
    }
};

}  // namespace

int strata::flint::run(int argc, char** argv) {
    std::string model, tokens, force, dump_full;
    int ctx = 0, chunk = 0, n_gen = 64, bench_pp = 0, bench_tg = 0;   // 0: auto (what fits beside the weights)
    bool warm = true, serve_mode = false, graph = true, embd_gpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-flint: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--max-new") n_gen = std::atoi(next().c_str());
        else if (a == "--no-token-graph" || a == "--no-graph") graph = false;
        else if (a == "--gpu") (void) next();
        else if (a == "--greedy") {}
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context" || a == "--ctx") { const std::string v = next(); ctx = v == "auto" ? 0 : std::atoi(v.c_str()); }
        else if (a == "--chunk" || a == "--prefill") { const std::string v = next(); chunk = v == "auto" ? 0 : std::atoi(v.c_str()); }
        else if (a == "--tokens") tokens = next();
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--force") force = next();
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--bench-pp") bench_pp = std::atoi(next().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(next().c_str());
        else if (a == "--no-warmup") warm = false;
        else if (a == "--embd-gpu") embd_gpu = true;
        else if (serve_mode && a.rfind("--", 0) == 0) {
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-flint: ignoring %s%s%s (not an option of this engine)\n", a.c_str(), val ? " " : "",
                         val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-flint --model GGUF [--max-context N|auto] [--chunk N|auto] [--no-graph] [--embd-gpu]\n"
                         "       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F] | --bench-pp N --bench-tg N)\n");
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "strata-flint: --model is required\n"); return 2; }
    // LAZY module loading (not the Qwen3.6 path's EAGER): a 4-bit Flint leaves ~1 GiB beside its weights on a 24 GB card,
    // and loading every one of Strata's kernels with the context took 0.08 GiB of it; the warm-up loads the ones used
#if !defined(_WIN32)
    setenv("CUDA_MODULE_LOADING", "LAZY", 0);
    setenv("STRATA_MMQ_KQ", "1", 0);   // the K-quant MMQ prompt kernels
#else
    if (std::getenv("STRATA_MMQ_KQ") == nullptr) _putenv_s("STRATA_MMQ_KQ", "1");
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) _putenv_s("CUDA_MODULE_LOADING", "LAZY");
#endif
    if ((chunk != 0 && (chunk < 16 || chunk > 4096)) || (ctx != 0 && ctx < 1024)) {
        std::fprintf(stderr, "strata-flint: bad --chunk / --max-context\n");
        return 2;
    }
    Model m;
    std::string err;
    try {
        if (!load(model, m, err, embd_gpu)) { std::fprintf(stderr, "strata-flint: %s\n", err.c_str()); return 1; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-flint: %s\n", e.what());
        return 1;
    }
    {
        // auto sizes from the VRAM left beside the weights (RTX 4090, Q4_K_M: the engine's buffers 0.31 / 0.43 / 0.68 GiB at
        // a chunk of 512 / 1024 / 2048, K / V 28 KB per cell of the full layers plus the window layers' ring), 0.2 GiB kept
        // for what is allocated later (cuBLAS, MMQ, the graph).  A bigger chunk reads prompts faster (MMQ sees more rows per
        // expert: pp4096 1024 1.36x of 512 with flash attention), a longer context holds more conversation.
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const double GiB = 1073741824.0;
        int n_full = 0, n_swa = 0;
        for (const Layer& L : m.layers) (L.swa ? n_swa : n_full) += 1;
        const double cell = (double) n_full * 2 * m.HKV * m.HD * 2;
        auto ring = [&](int c) { return (double) std::max(2048, (m.win + c + 255) / 256 * 256) * n_swa * 2 * m.HKV * m.HD * 2; };
        auto bufs = [&](int c) { return 0.19 * GiB + (double) c * 245.0 * 1024.0; };
        const double room = (double) fr - 0.2 * GiB;
        auto fits = [&](int c, int x) { return bufs(c) + ring(c) + cell * x <= room; };
        if (chunk == 0) {
            const int want_ctx = ctx > 0 ? ctx : 16384;   // an auto context keeps 16K cells before it grows the chunk
            chunk = 256;
            for (int c : {2048, 1024, 512})
                if (fits(c, want_ctx)) { chunk = c; break; }
        }
        if (ctx == 0) ctx = std::max(4096, std::min(m.ctx_train, (int) ((room - bufs(chunk) - ring(chunk)) / cell) / 1024 * 1024));
        std::fprintf(stderr, "strata-flint: %.2f GiB of VRAM free beside the weights: context %d cells, prompt chunk %d%s\n",
                     fr / GiB, ctx, chunk, fits(chunk, ctx) ? "" : " (MORE than the estimate fits: expect an out-of-memory)");
    }
    Engine E;
    E.use_graph = graph;
    E.init(m, ctx, chunk);
    E.reset();
    if (warm) E.warmup();
    if (serve_mode) {
        Serve sv(E);
        return sv.run();
    }
    if (bench_pp > 0 || bench_tg > 0) {
        // llama-bench's shapes: a prompt of bench_pp tokens from a fresh state, then bench_tg tokens one at a time
        std::vector<int32_t> toks((size_t) std::max(bench_pp, 1));
        std::mt19937 rng(1234);
        for (auto& t : toks) t = (int32_t) (rng() % 150000u);
        for (int rep = 0; rep < 3; ++rep) {
            E.reset();
            const auto t0 = Clock::now();
            E.process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
            CK(cudaStreamSynchronize(E.st));
            const double pp_ms = ms_since(t0);
            const auto t1 = Clock::now();
            int nx = E.greedy();
            for (int i = 0; i < bench_tg; ++i) {
                const int32_t one = nx;
                E.process(&one, 1, [](int64_t) {});
                nx = E.greedy();
            }
            const double tg_ms = ms_since(t1);
            std::printf("bench %d: pp%d %.1f tok/s, tg%d %.1f tok/s\n", rep, (int) toks.size(),
                        (double) toks.size() / (pp_ms / 1000.0), bench_tg, bench_tg > 0 ? (double) bench_tg / (tg_ms / 1000.0) : 0.0);
            std::fflush(stdout);
        }
        return 0;
    }
    const std::vector<int32_t> prompt = parse_ids(tokens.c_str()), forced = parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "strata-flint: --tokens, --serve or --bench-pp is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    std::vector<float> lg;
    std::vector<int32_t> gen;
    const auto t0 = Clock::now();
    E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {});
    const double pp_ms = ms_since(t0);
    const auto t1 = Clock::now();
    for (int i = 0; i < n_gen; ++i) {
        const int a = E.greedy();
        gen.push_back(a);
        if (ff) {
            E.host_logits(lg);
            std::fwrite(lg.data(), 4, lg.size(), ff);
        }
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.process(&nx, 1, [](int64_t) {});
    }
    if (ff) std::fclose(ff);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-flint: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.1f tok/s)\n",
                 prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), ms_since(t1),
                 gen.size() / (ms_since(t1) / 1000.0));
    return 0;
}

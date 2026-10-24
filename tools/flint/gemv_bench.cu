// tools/flint/gemv_bench.cu - Flint's dense Q8_0 decode GEMVs (one column) at the engine's shapes through native_mmvq,
// GGUF layout and the packed qs / d planes (STRATA_Q8_PACKED=1 registers a copy), against the card's DRAM bandwidth.
// Enough copies of each matrix are cycled to miss the 72 MB L2, as a layer's call does.
//     nvcc -O3 -arch=sm_89 -std=c++17 -I include tools/flint/gemv_bench.cu build/libstrata_kernels.a -lcudart -o gemv_bench
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}

int main() {
    struct Shape { const char* what; int n_in, n_out; };
    const Shape shapes[] = {{"qkv 2048->5120", 2048, 5120}, {"wo 4096->2048", 4096, 2048}, {"sh gate/up 2048->1024", 2048, 1024},
                            {"sh down 1024->2048", 1024, 2048}, {"head 2048->150144", 2048, 150144}};
    const bool packed = std::getenv("STRATA_Q8_PACKED") != nullptr;
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    std::mt19937 rng(7);
    for (const Shape& s : shapes) {
        const size_t nb = (size_t) s.n_in / 32 * s.n_out, bytes = nb * 34;
        const int copies = (int) std::max<size_t>(2, (size_t) (256u << 20) / bytes + 1);
        std::vector<uint8_t> h(bytes);
        for (size_t b = 0; b < nb; ++b) {
            const uint16_t d = 0x2000 + (rng() & 0x3ff);   // a small positive half
            std::memcpy(&h[b * 34], &d, 2);
            for (int j = 0; j < 32; ++j) h[b * 34 + 2 + j] = (uint8_t) (rng() & 0xff);
        }
        std::vector<uint8_t*> w(copies), wp(copies);
        for (int c = 0; c < copies; ++c) {
            ck(cudaMalloc(&w[c], bytes), "w");
            ck(cudaMemcpy(w[c], h.data(), bytes, cudaMemcpyHostToDevice), "w copy");
            if (packed && K::native_q8_0_packed_eligible(s.n_in, s.n_out)) {
                std::vector<uint8_t> p(bytes);
                K::native_q8_0_pack_host(h.data(), p.data(), s.n_in, s.n_out);
                ck(cudaMalloc(&wp[c], bytes), "wp");
                ck(cudaMemcpy(wp[c], p.data(), bytes, cudaMemcpyHostToDevice), "wp copy");
                K::native_q8_0_packed_register(w[c], wp[c], s.n_in, s.n_out);
            }
        }
        std::vector<float> x(s.n_in);
        for (auto& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.0f;
        float *dx, *dy;
        void* q;
        ck(cudaMalloc(&dx, x.size() * 4), "x");
        ck(cudaMalloc(&dy, (size_t) s.n_out * 4), "y");
        ck(cudaMalloc(&q, K::native_q8_1_bytes(s.n_in, 1)), "q");
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x copy");
        K::native_quantize_q8_1(dx, q, s.n_in, 1, st);
        for (int c = 0; c < copies; ++c) K::native_mmvq(8, w[c], q, dy, s.n_in, s.n_out, 1, st);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const int iters = 200;
        cudaEventRecord(e0, st);
        for (int i = 0; i < iters; ++i) K::native_mmvq(8, w[i % copies], q, dy, s.n_in, s.n_out, 1, st);
        cudaEventRecord(e1, st);
        ck(cudaEventSynchronize(e1), "sync");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = ms * 1000.0 / iters;
        std::printf("%-24s %s: %7.2f us  %6.0f GB/s\n", s.what, packed ? "packed" : "gguf  ", us, bytes / (us * 1e3));
        for (int c = 0; c < copies; ++c) {
            if (packed && wp[c]) { K::native_q8_0_packed_unregister(w[c]); cudaFree(wp[c]); }
            cudaFree(w[c]);
        }
        cudaFree(dx); cudaFree(dy); cudaFree(q);
    }
    return 0;
}

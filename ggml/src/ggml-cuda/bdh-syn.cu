#include "bdh-syn.cuh"

#include <cstdint>

static __global__ void k_bdh_syn_head_y(
        const float * g, const float * q, float * y,
        const int64_t N, const int64_t D,
        const int64_t sg0, const int64_t sgb) {
    const int64_t d = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) {
        return;
    }
    float acc = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        acc += g[n * sg0 + d * sgb] * q[n];
    }
    y[d] = acc;
}

static __global__ void k_bdh_syn_head_update(
        float * g, const float * q, const float * x,
        const int64_t N, const int64_t D,
        const int64_t sg0, const int64_t sgb) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = N * D;
    if (idx >= total) {
        return;
    }
    const int64_t n = idx % N;
    const int64_t d = idx / N;
    g[n * sg0 + d * sgb] += q[n] * x[d];
}

void ggml_cuda_bdh_syn_head(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grid = dst->src[0];
    const ggml_tensor * qr   = dst->src[1];
    const ggml_tensor * x    = dst->src[2];

    GGML_ASSERT(grid->type == GGML_TYPE_F32 && qr->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32);

    const int64_t N   = grid->ne[0];
    const int64_t D   = grid->ne[1];
    const int64_t sg0 = grid->nb[0] / (int64_t) sizeof(float);
    const int64_t sgb = grid->nb[1] / (int64_t) sizeof(float);

    const float * g  = (const float *) grid->data;
    const float * q  = (const float *) qr->data;
    const float * xv = (const float *) x->data;
    float *       y  = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int tpb = 256;
    const int blocks_y = (int) ((D + tpb - 1) / tpb);
    k_bdh_syn_head_y<<<blocks_y, tpb, 0, stream>>>(g, q, y, N, D, sg0, sgb);

    float * g_mut = (float *) g;
    const int64_t total = N * D;
    const int blocks_u = (int) ((total + tpb - 1) / tpb);
    k_bdh_syn_head_update<<<blocks_u, tpb, 0, stream>>>(g_mut, q, xv, N, D, sg0, sgb);
}
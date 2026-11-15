#include "bdh-syn.cuh"
#include "common.cuh"

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

    cudaStream_t stream    = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    const float    alpha  = 1.0f;
    const float    beta   = 0.0f;
    const int      tpb    = 256;

    if (sg0 == 1 && sgb == N && N <= INT_MAX && D <= INT_MAX) {
        // Contiguous [N,D] row-major: y = Gᵀ·qr via cublas (op T).
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, (int) N, (int) D, &alpha, g, (int) N, q, 1, &beta, y, 1));
    } else {
        const int blocks_y = (int) ((D + tpb - 1) / tpb);
        k_bdh_syn_head_y<<<blocks_y, tpb, 0, stream>>>(g, q, y, N, D, sg0, sgb);
    }

    float * g_mut = (float *) g;
    const int64_t total = N * D;
    const int blocks_u = (int) ((total + tpb - 1) / tpb);
    k_bdh_syn_head_update<<<blocks_u, tpb, 0, stream>>>(g_mut, q, xv, N, D, sg0, sgb);
}

static __global__ void k_bdh_gated_decoder_y(
        const float * W, const float * xs, const float * ys, float * y,
        const int64_t K, const int64_t D,
        const int64_t sw0, const int64_t sw1) {
    const int64_t d = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) {
        return;
    }
    float acc = 0.0f;
    for (int64_t n = 0; n < K; ++n) {
        acc += W[n * sw0 + d * sw1] * xs[n] * ys[n];
    }
    y[d] = acc;
}

void ggml_cuda_bdh_gated_decoder(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * decoder = dst->src[0];
    const ggml_tensor * xsp     = dst->src[1];
    const ggml_tensor * ysp     = dst->src[2];

    GGML_ASSERT(decoder->type == GGML_TYPE_F32 && xsp->type == GGML_TYPE_F32 && ysp->type == GGML_TYPE_F32);

    const int64_t K   = xsp->ne[0];
    const int64_t D   = decoder->ne[1];
    const int64_t sw0 = decoder->nb[0] / (int64_t) sizeof(float);
    const int64_t sw1 = decoder->nb[1] / (int64_t) sizeof(float);

    const float * W  = (const float *) decoder->data;
    const float * xs = (const float *) xsp->data;
    const float * ys = (const float *) ysp->data;
    float *       y  = (float *) dst->data;

    cudaStream_t stream = ctx.stream();
    const int    tpb    = 256;
    const int blocks_y  = (int) ((D + tpb - 1) / tpb);
    k_bdh_gated_decoder_y<<<blocks_y, tpb, 0, stream>>>(W, xs, ys, y, K, D, sw0, sw1);
}
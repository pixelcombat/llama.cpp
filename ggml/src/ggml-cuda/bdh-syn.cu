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

static __global__ void k_bdh_relu_f32(float * y, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = y[i] > 0.0f ? y[i] : 0.0f;
    }
}

static __global__ void k_bdh_mul_f32(const float * a, const float * b, float * z, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = a[i] * b[i];
    }
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

    cudaStream_t    stream = ctx.stream();
    cublasHandle_t  handle = ctx.cublas_handle();
    const float     alpha  = 1.0f;
    const float     beta   = 0.0f;
    const int       tpb    = 256;

    if (sw0 == 1 && sw1 == K && K <= INT_MAX && D <= INT_MAX) {
        ggml_cuda_pool_alloc<float> z_alloc(ctx.pool(), (size_t) K);
        float * z = z_alloc.get();
        const int blocks_z = (int) ((K + tpb - 1) / tpb);
        k_bdh_mul_f32<<<blocks_z, tpb, 0, stream>>>(xs, ys, z, K);
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, (int) K, (int) D, &alpha, W, (int) K, z, 1, &beta, y, 1));
    } else {
        const int blocks_y = (int) ((D + tpb - 1) / tpb);
        k_bdh_gated_decoder_y<<<blocks_y, tpb, 0, stream>>>(W, xs, ys, y, K, D, sw0, sw1);
    }
}

static __global__ void k_bdh_enc_x_relu(
        const float * W, const float * x, float * y,
        const int64_t D, const int64_t K, const int64_t sld) {
    const int64_t k = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) {
        return;
    }
    float acc = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
        acc += W[d * sld + k] * x[d];
    }
    y[k] = acc > 0.0f ? acc : 0.0f;
}

static __global__ void k_bdh_enc_v_relu(
        const float * W, const float * Y, float * O,
        const int64_t D, const int64_t N, const int64_t nh, const int64_t plane) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = N * nh;
    if (idx >= total) {
        return;
    }
    const int64_t n = idx % N;
    const int64_t h = idx / N;
    const float * wh = W + h * plane + n * D;
    const float * yh = Y + h * D;
    float acc        = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
        acc += wh[d] * yh[d];
    }
    O[n + h * N] = acc > 0.0f ? acc : 0.0f;
}

void ggml_cuda_bdh_enc_x_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];

    GGML_ASSERT(w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32);

    const int64_t D = w->ne[0];
    const int64_t K = w->ne[1];
    const int64_t sw0 = w->nb[0] / (int64_t) sizeof(float);
    const int64_t sw1 = w->nb[1] / (int64_t) sizeof(float);

    const float * W  = (const float *) w->data;
    const float * xv = (const float *) x->data;
    float *       y  = (float *) dst->data;

    cudaStream_t stream = ctx.stream();
    const int    tpb    = 256;

    if (sw0 == 1 && sw1 == D) {
        const int blocks = (int) ((K + tpb - 1) / tpb);
        k_bdh_enc_x_relu<<<blocks, tpb, 0, stream>>>(W, xv, y, D, K, sw1);
    } else {
        GGML_ABORT("ggml_cuda_bdh_enc_x_relu: non-contiguous weights");
    }
}

void ggml_cuda_bdh_enc_v_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w    = dst->src[0];
    const ggml_tensor * y_kv = dst->src[1];

    GGML_ASSERT(w->type == GGML_TYPE_F32 && y_kv->type == GGML_TYPE_F32);

    const int64_t D  = w->ne[0];
    const int64_t N  = w->ne[1];
    const int64_t nh = w->ne[2];

    const int64_t sw0 = w->nb[0] / (int64_t) sizeof(float);
    const int64_t sw1 = w->nb[1] / (int64_t) sizeof(float);
    const int64_t sw2 = w->nb[2] / (int64_t) sizeof(float);

    float * O = (float *) dst->data;
    const int tpb = 256;

    if (sw0 == 1 && sw1 == D && sw2 == D * N) {
        const int64_t  plane = D * N;
        const float *  Wp    = (const float *) w->data;
        const float *  Yp    = (const float *) y_kv->data;
        const int64_t  total = N * nh;
        const int      blocks = (int) ((total + tpb - 1) / tpb);
        k_bdh_enc_v_relu<<<blocks, tpb, 0, ctx.stream()>>>(Wp, Yp, O, D, N, nh, plane);
    } else {
        GGML_ABORT("ggml_cuda_bdh_enc_v_relu: non-contiguous weights");
    }
}

void ggml_cuda_bdh_lm_head(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * output = dst->src[0];
    const ggml_tensor * x      = dst->src[1];

    GGML_ASSERT(output->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32);

    const int64_t D   = output->ne[0];
    const int64_t V   = output->ne[1];
    const int64_t sw0 = output->nb[0] / (int64_t) sizeof(float);
    const int64_t sw1 = output->nb[1] / (int64_t) sizeof(float);

    const float * W  = (const float *) output->data;
    const float * xv = (const float *) x->data;
    float *       y  = (float *) dst->data;

    cublasHandle_t handle = ctx.cublas_handle();
    const float    alpha  = 1.0f;
    const float    beta   = 0.0f;

    if (sw0 == 1 && sw1 == D && D <= INT_MAX && V <= INT_MAX) {
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, (int) D, (int) V, &alpha, W, (int) D, xv, 1, &beta, y, 1));
    } else {
        GGML_ABORT("ggml_cuda_bdh_lm_head: non-contiguous weights");
    }
}
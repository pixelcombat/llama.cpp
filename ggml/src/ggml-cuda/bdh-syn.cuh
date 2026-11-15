#pragma once

#include "common.cuh"

void ggml_cuda_bdh_syn_head(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_bdh_gated_decoder(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
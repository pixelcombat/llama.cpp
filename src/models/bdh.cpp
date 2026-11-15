#include "models.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

static ggml_tensor * g_bdh_debug_pick = nullptr;

static const char * bdh_debug_tag() {
    return std::getenv("BDH_DEBUG_OUT");
}

static void bdh_debug_select(ggml_tensor * t, const char * name) {
    const char * want = bdh_debug_tag();
    if (want && t && std::strcmp(want, name) == 0) {
        g_bdh_debug_pick = t;
    }
}

static void bdh_debug_select_il(ggml_tensor * t, const char * name, int il) {
    const char * want = bdh_debug_tag();
    if (!want || !t) {
        return;
    }
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s-%d", name, il);
    if (std::strcmp(want, name) == 0 || std::strcmp(want, buf) == 0) {
        g_bdh_debug_pick = t;
    }
}

static ggml_tensor * stack_heads_dim2(
        ggml_context *                    ctx,
        const std::vector<ggml_tensor *> & heads,
        int64_t                           ne0,
        int64_t                           ne1) {
    GGML_ASSERT(!heads.empty());

    ggml_tensor * out = ggml_reshape_3d(ctx, heads[0], ne0, ne1, 1);
    for (size_t i = 1; i < heads.size(); ++i) {
        ggml_tensor * h = ggml_reshape_3d(ctx, heads[i], ne0, ne1, 1);
        out             = ggml_concat(ctx, out, h, 2);
    }
    return out;
}

static ggml_tensor * stack_heads_dim0(
        ggml_context *                    ctx,
        const std::vector<ggml_tensor *> & heads,
        int64_t                           ne0,
        int64_t                           ne1) {
    GGML_ASSERT(!heads.empty());
    ggml_tensor * out = ggml_reshape_2d(ctx, ggml_dup(ctx, ggml_cont(ctx, heads[0])), ne0, ne1);
    for (size_t i = 1; i < heads.size(); ++i) {
        ggml_tensor * h = ggml_reshape_2d(ctx, ggml_dup(ctx, ggml_cont(ctx, heads[i])), ne0, ne1);
        out             = ggml_concat(ctx, out, h, 0);
    }
    return out;
}

static void bdh_rope_phases(
        ggml_context * ctx,
        ggml_tensor *  freqs,
        ggml_tensor *  inp_pos,
        int64_t        N,
        int64_t        n_tokens,
        ggml_tensor ** ph_cos,
        ggml_tensor ** ph_sin) {
    const bool fast_single = n_tokens == 1;
    ggml_tensor * pos_f32  = ggml_cast(ctx, inp_pos, GGML_TYPE_F32);
    ggml_tensor * pos_row  = fast_single
        ? ggml_reshape_2d(ctx, pos_f32, 1, n_tokens)
        : ggml_cont(ctx, ggml_reshape_2d(ctx, pos_f32, 1, n_tokens));
    ggml_tensor * freq_col = fast_single ? ggml_reshape_2d(ctx, freqs, N, 1)
                                         : ggml_cont(ctx, ggml_reshape_2d(ctx, freqs, N, 1));
    ggml_tensor * phases   = ggml_mul(ctx,
                                      ggml_repeat_4d(ctx, freq_col, N, n_tokens, 1, 1),
                                      ggml_repeat_4d(ctx, pos_row, N, n_tokens, 1, 1));
    phases                 = fast_single ? ggml_reshape_2d(ctx, phases, N, n_tokens)
                                         : ggml_cont(ctx, ggml_reshape_2d(ctx, phases, N, n_tokens));

    phases                 = ggml_sub(ctx, phases, ggml_floor(ctx, phases));
    phases                 = ggml_scale(ctx, phases, 2.0f * (float) M_PI);

    *ph_cos = fast_single ? ggml_cos(ctx, phases) : ggml_cont(ctx, ggml_cos(ctx, phases));
    *ph_sin = fast_single ? ggml_sin(ctx, phases) : ggml_cont(ctx, ggml_sin(ctx, phases));
}

static ggml_tensor * bdh_rope_apply(
        ggml_context * ctx,
        ggml_tensor *  ph_cos,
        ggml_tensor *  ph_sin,
        ggml_tensor *  v,
        bool           fast_single) {
    const int64_t N  = v->ne[0];
    const int64_t n2 = N / 2;

    // v [N,T]: torch pairs (2j,2j+1) → reshape [2,n2,T], nie [n2,2,T] (połówki widma).
    ggml_tensor * v_in  = fast_single ? v : ggml_cont(ctx, v);
    ggml_tensor * vp    = ggml_reshape_3d(ctx, v_in, 2, n2, v->ne[1]);
    ggml_tensor * even  = ggml_view_3d(ctx, vp, 1, n2, v->ne[1], vp->nb[1], vp->nb[2], 0);
    ggml_tensor * odd   = ggml_view_3d(ctx, vp, 1, n2, v->ne[1], vp->nb[1], vp->nb[2], vp->nb[0]);
    if (!fast_single) {
        even = ggml_cont(ctx, even);
        odd  = ggml_cont(ctx, odd);
    }
    ggml_tensor * v_rot = ggml_concat(ctx, ggml_neg(ctx, odd), even, /*dim=*/0);
    v_rot               = fast_single ? ggml_reshape_2d(ctx, v_rot, N, v->ne[1])
                                      : ggml_cont(ctx, ggml_reshape_2d(ctx, ggml_cont(ctx, v_rot), N, v->ne[1]));

    if (!fast_single) {
        v      = ggml_cont(ctx, v);
        ph_cos = ggml_cont(ctx, ph_cos);
        ph_sin = ggml_cont(ctx, ph_sin);
    }
    ggml_tensor * out = ggml_add(ctx, ggml_mul(ctx, v_in, ph_cos), ggml_mul(ctx, v_rot, ph_sin));
    return fast_single ? out : ggml_cont(ctx, out);
}

// GGUF encoder [N,D,nh] → jeden macierz [D, nh*N] (kolumny = transpozycje głów).
static ggml_tensor * bdh_encoder_weights_2d(
        ggml_context * ctx,
        ggml_tensor *  encoder,
        int64_t        n_head,
        int64_t        N,
        int64_t        D) {
    GGML_UNUSED(n_head);
    GGML_UNUSED(N);
    ggml_tensor * w = ggml_permute(ctx, encoder, 1, 0, 2, 3);
    w               = ggml_cont(ctx, w);
    return ggml_reshape_2d(ctx, w, D, n_head * N);
}

static ggml_tensor * bdh_encoder_fused(
        ggml_context * ctx,
        ggml_tensor *  encoder,
        ggml_tensor *  x,
        int64_t        n_head,
        int64_t        N,
        int64_t        D) {
    ggml_tensor * w = bdh_encoder_weights_2d(ctx, encoder, n_head, N, D);
    return ggml_relu(ctx, ggml_mul_mat(ctx, w, x));
}

// encoder_v [N,D,nh] → [D,N,nh]; y_kv [D,T,nh] — jeden mul_mat (broadcast ne2).
static ggml_tensor * bdh_encoder_v_fused(
        ggml_context * ctx,
        ggml_tensor *  encoder_v,
        ggml_tensor *  y_kv) {
    ggml_tensor * w = ggml_cont(ctx, ggml_permute(ctx, encoder_v, 1, 0, 2, 3));
    ggml_tensor * y = ggml_mul_mat(ctx, w, y_kv);
    return ggml_relu(ctx, y);
}

// Z (nh*N,T) → widok [N,T] jak stary stack_heads_dim2 (dim głów = 2).
static ggml_tensor * bdh_sparse_3d_nt(
        ggml_context * ctx,
        ggml_tensor *  fused,
        int64_t        N,
        int64_t        nh,
        int64_t        n_tokens,
        bool           fast_single = false) {
    ggml_tensor * t = ggml_reshape_3d(ctx, fused, N, nh, n_tokens);
    ggml_tensor * p = ggml_permute(ctx, t, 0, 2, 1, 3);
    return fast_single ? p : ggml_cont(ctx, p);
}

static ggml_tensor * bdh_head_from_fused(ggml_context * ctx, ggml_tensor * fused, int64_t h, int64_t N) {
    return ggml_view_2d(
        ctx, fused, N, fused->ne[1], fused->nb[1], (size_t) h * (size_t) N * ggml_element_size(fused));
}

static ggml_tensor * bdh_attn_head(
        ggml_context * ctx,
        ggml_tensor *  ph_cos,
        ggml_tensor *  ph_sin,
        ggml_tensor *  qk,
        ggml_tensor *  v,
        int64_t        n_tokens,
        int            head) {
    GGML_UNUSED(head);
    GGML_ASSERT(qk->ne[1] == n_tokens);
    GGML_ASSERT(v->ne[1] == n_tokens);
    ggml_tensor * qr     = bdh_rope_apply(ctx, ph_cos, ph_sin, qk, n_tokens == 1);
    ggml_tensor * kr     = qr;
    // Jak build_attn_mha: mul_mat(k,q) → ne0=klucz, ne1=zapytanie; LOWER = klucz < zapytanie (≈ torch tril(-1)).
    ggml_tensor * scores = ggml_cont(ctx, ggml_mul_mat(ctx, kr, qr));
    if (n_tokens <= 1) {
        scores = ggml_scale(ctx, scores, 0.0f);
    } else {
        scores = ggml_cont(ctx, scores);
        scores = ggml_cont(ctx, ggml_tri(ctx, scores, GGML_TRI_TYPE_LOWER));
    }
    if (head == 0) {
        bdh_debug_select(scores, "scores0_tri");
    }
    ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, ggml_cont(ctx, v)));
    ggml_tensor * y  = ggml_cont(ctx, ggml_mul_mat(ctx, vt, scores));
    if (head == 0) {
        bdh_debug_select(y, "attn0_raw");
    }
    return y;
}

static ggml_tensor * bdh_grid_outer_delta(
        ggml_context * ctx,
        ggml_tensor *  qr,
        ggml_tensor *  x,
        bool           fast_single = false) {
    // stamped (N,1) ⊗ x (D,) → (N,D) — jak streaming.py: unsqueeze(-1) * x
    if (fast_single) {
        return ggml_out_prod(ctx, qr, x);
    }
    const int64_t N = qr->ne[0];
    const int64_t D = x->ne[0];
    ggml_tensor * qr_w  = ggml_cont(ctx, ggml_repeat_4d(ctx, qr, N, D, 1, 1));
    ggml_tensor * x_col = ggml_cont(ctx, ggml_reshape_2d(ctx, x, D, 1));
    ggml_tensor * x_w   = ggml_cont(ctx, ggml_transpose(ctx, ggml_repeat_4d(ctx, x_col, D, N, 1, 1)));
    return ggml_cont(ctx, ggml_mul(ctx, qr_w, x_w));
}

static ggml_tensor * bdh_grid_head_write_nd(
        ggml_context * ctx,
        ggml_tensor *  grid_nd,
        ggml_tensor *  qr,
        ggml_tensor *  x,
        bool           fast_single = false) {
    ggml_tensor * delta = bdh_grid_outer_delta(ctx, qr, x, fast_single);
    ggml_tensor * sum   = ggml_add(ctx, grid_nd, delta);
    return fast_single ? sum : ggml_cont(ctx, sum);
}

} // namespace

void llama_model_bdh::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(std::string("bdh.sparse_dim_multiplier"), bdh_sparse_dim_multiplier, false);
    if (bdh_sparse_dim_multiplier <= 0) {
        bdh_sparse_dim_multiplier = 128;
    }
    hparams.bdh_sparse_dim_multiplier = (uint32_t) bdh_sparse_dim_multiplier;

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps, false);

    switch (hparams.n_layer()) {
        case 6:
            switch (hparams.n_embd) {
                case 512: type = LLM_TYPE_SMALL; break;
                default:  type = LLM_TYPE_UNKNOWN;
            } break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_bdh::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t nh = n_head;
    const int64_t N  = (int64_t) bdh_sparse_dim_multiplier * n_embd / nh;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    bdh_encoder    = create_tensor(tn(LLM_TENSOR_BDH_ENCODER),   { N, n_embd, nh }, 0);
    bdh_encoder_v  = create_tensor(tn(LLM_TENSOR_BDH_ENCODER_V), { N, n_embd, nh }, 0);
    bdh_decoder    = create_tensor(tn(LLM_TENSOR_BDH_DECODER),   { nh * N, n_embd }, 0);
    bdh_attn_freqs = create_tensor(tn(LLM_TENSOR_BDH_ATTN_FREQS), { N, 1, 1, 1 }, 0);
}

std::unique_ptr<llm_graph_context> llama_model_bdh::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_bdh::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    g_bdh_debug_pick = nullptr;

    const auto & m = static_cast<const llama_model_bdh &>(model);

    const int64_t D        = hparams.n_embd;
    const int64_t nh       = hparams.n_head(0);
    const int64_t N        = (int64_t) m.bdh_sparse_dim_multiplier * D / nh;
    const int64_t n_tokens = ubatch.n_tokens;
    const int64_t n_seqs   = ubatch.n_seqs;
    const bool    rope_fast = n_tokens == 1;

    llm_graph_input_rs * inp_rs = nullptr;
    const llama_memory_recurrent_context * mctx_recr = nullptr;
    if (mctx && !bdh_debug_tag()) {
        mctx_recr = static_cast<const llama_memory_recurrent_context *>(mctx);
    }
    const bool syn_step = mctx_recr != nullptr && params.bdh_synaptic && n_tokens == 1;
    const bool rs_fast  = syn_step && n_seqs == 1 && mctx_recr->get_n_rs() == (uint32_t) n_seqs &&
                          (!std::getenv("BDH_RS_DIRECT") || std::strcmp(std::getenv("BDH_RS_DIRECT"), "0") != 0);
    const char * syn_fused_env = std::getenv("BDH_SYN_FUSED");
    const bool   syn_fused_off = syn_fused_env && std::strcmp(syn_fused_env, "0") == 0;
    const bool   syn_fused     = syn_step && rs_fast && rope_fast && !syn_fused_off;
    if (syn_step && !rs_fast) {
        inp_rs = build_rs_inp();
    }
    const int32_t state_size = (int32_t) (nh * N * D);
    const auto    bdh_rs_view = [mctx_recr, state_size, n_seqs](ggml_context * ctx, ggml_tensor * states,
                                                                 ggml_tensor * ids) -> ggml_tensor * {
        GGML_UNUSED(ids);
        const uint32_t head = mctx_recr->get_head();
        return ggml_view_2d(ctx, states, state_size, n_seqs, states->nb[1], (size_t) head * states->nb[1]);
    };

    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cb(cur, "model.embed_tokens", -1);

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_build_forward_expand(gf, inp_pos);

    ggml_tensor * inp_out_ids = build_inp_out_ids();
    if (inp_out_ids != nullptr) {
        ggml_build_forward_expand(gf, inp_out_ids);
    }

    cur = build_norm(cur, nullptr, nullptr, LLM_NORM, -1);
    cb(cur, "model.ln_embed", -1);
    bdh_debug_select(cur, "ln_embed");
    GGML_ASSERT(cur->ne[1] == n_tokens);

    ggml_tensor * freqs = rope_fast ? ggml_reshape_2d(ctx0, m.bdh_attn_freqs, N, 1)
                                  : ggml_reshape_2d(ctx0, ggml_cont(ctx0, m.bdh_attn_freqs), N, 1);

    ggml_tensor * ph_cos = nullptr;
    ggml_tensor * ph_sin = nullptr;
    bdh_rope_phases(ctx0, freqs, inp_pos, N, n_tokens, &ph_cos, &ph_sin);
    bdh_debug_select(ph_cos, "ph_cos");

    ggml_tensor * w_enc   = bdh_encoder_weights_2d(ctx0, m.bdh_encoder, nh, N, D);
    ggml_tensor * w_enc_v = ggml_cont(ctx0, ggml_permute(ctx0, m.bdh_encoder_v, 1, 0, 2, 3));

    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> grid_pending_cpy;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * x = rope_fast ? cur : ggml_cont(ctx0, cur);

        ggml_tensor * xsp                         = ggml_relu(ctx0, ggml_mul_mat(ctx0, w_enc, x));
        ggml_tensor * x_sparse                    = bdh_sparse_3d_nt(ctx0, xsp, N, nh, n_tokens, rope_fast);
        cb(x_sparse, "model.layers.x_sparse", il);
        bdh_debug_select_il(x_sparse, "x_sparse", il);
        if (!bdh_debug_tag()) {
            bdh_debug_select_il(bdh_head_from_fused(ctx0, xsp, 0, N), "x_sparse_h", 0);
        }

        ggml_tensor * grid_s = nullptr;
        if (syn_step) {
            if (rs_fast) {
                ggml_tensor * states = ggml_reshape_2d(ctx0, mctx_recr->get_s_l(il), state_size, mctx_recr->get_s_l(il)->ne[1]);
                grid_s               = bdh_rs_view(ctx0, states, nullptr);
            } else if (inp_rs) {
                grid_s = build_rs(inp_rs, mctx_recr->get_s_l(il), state_size, (int32_t) n_seqs);
            }
        }

        std::vector<ggml_tensor *> attn_heads;
        std::vector<ggml_tensor *> grid_out_heads;
        attn_heads.reserve(nh);
        grid_out_heads.reserve(nh);
        const size_t head_grid_bytes = (size_t) N * (size_t) D * sizeof(float);
        for (int64_t h = 0; h < nh; ++h) {
            ggml_tensor * qk = bdh_head_from_fused(ctx0, xsp, h, N);
            ggml_tensor * y_h;
            if (syn_step && grid_s) {
                // Stan głowy (N,D) — zgodnie z streaming.py grids[layer,h]
                ggml_tensor * grid_h = ggml_view_2d(
                    ctx0, grid_s, N, D, N * (int64_t) sizeof(float), (size_t) h * head_grid_bytes);
                ggml_tensor * qr = bdh_rope_apply(ctx0, ph_cos, ph_sin, qk, rope_fast);
                if (syn_fused) {
                    y_h = ggml_bdh_syn_head(ctx0, grid_h, qr, x);
                } else {
                    y_h = rope_fast ? ggml_mul_mat(ctx0, grid_h, qr)
                                    : ggml_cont(ctx0, ggml_mul_mat(ctx0, grid_h, qr));
                    ggml_tensor * grid_w = bdh_grid_head_write_nd(ctx0, grid_h, qr, x, rope_fast);
                    grid_w               = ggml_add(ctx0, grid_w, ggml_scale(ctx0, y_h, 0.0f));
                    if (rs_fast) {
                        grid_pending_cpy.emplace_back(grid_w, grid_h);
                    } else {
                        grid_out_heads.push_back(grid_w);
                    }
                }
            } else {
                y_h = bdh_attn_head(ctx0, ph_cos, ph_sin, qk, x, n_tokens, (int) h);
            }
            if (h == 0) {
                bdh_debug_select_il(y_h, "attn0_raw", il);
            }
            ggml_tensor * y_ln = build_norm(y_h, nullptr, nullptr, LLM_NORM, il);
            if (h == 0) {
                bdh_debug_select_il(y_ln, "attn0_ln", il);
            }
            attn_heads.push_back(y_ln);
        }

        if (syn_step && !rs_fast && inp_rs && grid_s && grid_out_heads.size() == (size_t) nh) {
            const auto kv_head = mctx_recr->get_head();
            ggml_tensor * grid_out = stack_heads_dim2(ctx0, grid_out_heads, N, D);
            grid_out               = ggml_cont(ctx0, ggml_reshape_2d(ctx0, grid_out, state_size, n_seqs));
            ggml_tensor * s_slot   = mctx_recr->get_s_l(il);
            ggml_build_forward_expand(
                gf,
                ggml_cpy(ctx0,
                         grid_out,
                         ggml_view_1d(ctx0,
                                        s_slot,
                                        state_size * n_seqs,
                                        (size_t) state_size * kv_head * ggml_element_size(s_slot))));
        }

        ggml_tensor * y_kv = stack_heads_dim2(ctx0, attn_heads, D, n_tokens);
        cb(y_kv, "model.layers.y_kv", il);
        bdh_debug_select_il(y_kv, "y_kv", il);

        ggml_tensor * ysp_nt  = ggml_relu(ctx0, ggml_mul_mat(ctx0, w_enc_v, y_kv));
        ggml_tensor * ysp_nht = ggml_permute(ctx0, ysp_nt, 0, 2, 1, 3);
        ggml_tensor * ysp     = rope_fast ? ggml_reshape_2d(ctx0, ysp_nht, nh * N, n_tokens)
                                        : ggml_reshape_2d(ctx0, ggml_cont(ctx0, ysp_nht), nh * N, n_tokens);
        ggml_tensor * y_sparse  = bdh_sparse_3d_nt(ctx0, ysp, N, nh, n_tokens, rope_fast);
        bdh_debug_select_il(y_sparse, "y_sparse", il);
        ggml_tensor * gated_ntn = ggml_mul(ctx0, x_sparse, y_sparse);
        bdh_debug_select_il(gated_ntn, "gated_ntn", il);
        ggml_tensor * gated_dec = rope_fast ? ggml_mul(ctx0, xsp, ysp)
                                            : ggml_cont(ctx0, ggml_mul(ctx0, xsp, ysp));
        bdh_debug_select_il(gated_dec, "gated_dec", il);
        ggml_tensor * y_mlp = build_lora_mm(m.bdh_decoder, gated_dec, nullptr);
        cb(y_mlp, "model.layers.y_mlp", il);
        bdh_debug_select_il(y_mlp, "y_mlp", il);

        y_mlp = build_norm(y_mlp, nullptr, nullptr, LLM_NORM, il);
        cur   = build_norm(ggml_add(ctx0, x, y_mlp), nullptr, nullptr, LLM_NORM, il);
        bdh_debug_select_il(cur, "l_out", il);

        if (rs_fast && !grid_pending_cpy.empty()) {
            for (const auto & wh : grid_pending_cpy) {
                ggml_tensor * w_dep = ggml_add(ctx0, wh.first, ggml_scale(ctx0, cur, 0.0f));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, w_dep, wh.second));
            }
            grid_pending_cpy.clear();
        }

        if (il == n_layer - 1 && inp_out_ids && !bdh_debug_tag()) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
    }

    if (bdh_debug_tag() && g_bdh_debug_pick) {
        ggml_tensor * dbg = ggml_cont(ctx0, g_bdh_debug_pick);
        cb(dbg, "bdh_debug_out", -1);
        res->t_bdh_debug = dbg;
        res->t_embd      = nullptr;
        res->t_logits    = nullptr;
        ggml_build_forward_expand(gf, dbg);
        return;
    }

    cb(cur, "result_embd", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

}


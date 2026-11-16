# BDH fork changelog

[llama.cpp](https://github.com/ggml-org/llama.cpp) at tag **`v1.0.0`** (`b1b770d`), extended with **`LLM_ARCH_BDH`** (Dragon Hatchling, Pathway arXiv:2509.26507).

## 2026-11-16 — v1.0.0 integration

| Commit     | Summary |
|------------|---------|
| `bcdbac8`  | Merge upstream `v1.0.0`; retain BDH graph, server chat trim, fused ops. |
| `fd7371e`  | `GGML_OP_COUNT` → 104 (v1.0.0 + 5 BDH ops). |

## BDH feature set (vs upstream)

- Recurrent synaptic grid (`BDH_SYN_HEAD`, optional fused path).
- Tight prefill / micro-batch decode; `llama_context` graph node budgets for `LLM_ARCH_BDH`.
- `scripts/convert_bdh_pt_to_gguf.py`, `README-BDH.md`.
- Server: per-slot `n_ctx`, chat history trim for BDH models, UI `sw.js` hygiene.

## License

MIT, same as llama.cpp (`LICENSE`).
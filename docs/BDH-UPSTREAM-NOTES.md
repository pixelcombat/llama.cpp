# Notes for a future upstream PR (not submitted)

This fork is maintained separately. If BDH were proposed to ggml-org/llama.cpp, the logical split would be:

1. **ggml** — `GGML_OP_BDH_*` ops, CPU/CUDA kernels, `GGML_OP_COUNT` bump.
2. **llama** — `LLM_ARCH_BDH`, `llama_model_bdh`, context/graph budgets, synaptic toggle.
3. **tools** — optional server tweaks (context trim for recurrent models).

Upstream baseline for this tree: **`v1.0.0`**.

No upstream PR is opened from this publication.
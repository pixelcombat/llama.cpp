# BDH — Small Dragon Hatchling in llama.cpp

This tree is [llama.cpp](https://github.com/ggml-org/llama.cpp) at upstream **`v1.0.0`**, with **`LLM_ARCH_BDH`** wired through ggml, inference, and the HTTP server.

Pathway's MIT license and copyright notice are included in [licenses/LICENSE-pathway-bdh](licenses/LICENSE-pathway-bdh).

BDH is a compact recurrent language model: instead of a growing KV cache, a **synaptic grid** carries state across decode steps. Each new token updates that grid; the next step reads it back. Prefill can run as one **tight micro-batch** (ubatch length = prompt length) so the recurrent state is established before greedy or sampled decode.

## What is in this fork

| Layer | Role |
|-------|------|
| **ggml** | Dedicated ops (`BDH_SYN_HEAD`, gated decoder, encoder ReLU paths, LM head). Optional **fused** synaptic head on CPU/CUDA (`BDH_SYN_FUSED`). |
| **llama** | `llama_model_bdh`, graph budgets for sparse per-head loops, `bdh_synaptic` path on single-token decode. |
| **Server** | Slot `n_ctx`, chat history trim when the runtime context is smaller than the training window; standard `llama-server` API. |
| **Conversion** | `scripts/convert_bdh_pt_to_gguf.py` — PyTorch checkpoint → GGUF. |

Recurrent decode is on by default (`BDH_SYNAPTIC`); fused kernels and direct RS views are the fast paths most deployments use.

## Build

Branch: **`bdh-v1.0.0`**.

**Requirements:** Linux, GCC ≥ 11, CMake ≥ 3.20; Python 3.10+ and `torch` for GGUF export; CUDA toolkit optional (example arches `60;70`).

```bash
git checkout bdh-v1.0.0

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON -DGGML_BLAS=ON
cmake --build build -j --target llama-server

# CUDA
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON -DGGML_BLAS=ON -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="60;70"
cmake --build build-cuda -j --target llama-server
```

## Model weights

```bash
python3 scripts/convert_bdh_pt_to_gguf.py /path/to/bdh-100m-bpe8k.pt -o model.gguf
```

Uses this repository’s `gguf-py/`.

## Inference

```bash
./build-cuda/bin/llama-server -m model.gguf --ctx-size 2048 -ngl 99
```

Chat templates: prefer a simple user/assistant layout or `--chat-template-file` if the GGUF tokenizer metadata does not match your UI.

### Runtime toggles

| Variable | Default | Effect |
|----------|---------|--------|
| `BDH_SYN_FUSED` | on | Fused `BDH_SYN_HEAD` instead of decomposed matmul + grid write. |
| `BDH_RS_DIRECT` | on | Direct read/write path for recurrent state. |
| `BDH_SYNAPTIC` | on | Recurrent grid updates on decode (`0` = ablation). |
| `BDH_N_THREADS` | host CPUs | ggml CPU threads. |

## Verification

`docs/BDH-TEST-RESULTS.md` records the gate run before this release: PL chat-template parity, batch prefill vs tight ubatch, long prompts beyond the train window, CUDA staging build. Re-run with your own GGUF and a local PyTorch reference if you change kernels or schedulers.

## History

`CHANGELOG-BDH.md` — BDH-specific commits on `v1.0.0`.

## License

MIT, as upstream (`LICENSE`).
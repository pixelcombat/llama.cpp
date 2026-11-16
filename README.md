# Baby Dragon Hatchling (BDH) on llama.cpp

Branch **`bdh-v1.0.0`** — [llama.cpp](https://github.com/ggml-org/llama.cpp) at upstream tag **v1.0.0**, extended for the **BDH** recurrent architecture.

Architecture source: Pathway, [*The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain*](https://arxiv.org/abs/2509.26507) (arXiv:2509.26507).

Upstream project overview and general build options: **README-llama.cpp.md**.

---

## Idea

A transformer spends compute and memory on a **KV cache** that grows with each generated token. Attention re-reads that cache every step.

BDH keeps a fixed-size **synaptic grid**: state lives in a recurrent buffer updated after each token. Decode is a short graph (often one token per step) that reads the grid, runs a gated decoder, and writes back. There is no unbounded KV tensor in the familiar sense.

**Prefill** runs the prompt in a **tight micro-batch** (`n_ubatch` = prompt length) so the grid matches full-sequence training before autoregressive steps begin.

## Versus a standard decoder stack

| | Typical causal transformer | BDH in this fork |
|--|---------------------------|------------------|
| Memory across steps | KV cache (layers × heads × seq) | Synaptic grid (bounded) |
| Attention pattern | Full or windowed softmax over past | Per-head sparse routing on the grid |
| Prefill | Large batched matmuls | Tight ubatch; dedicated encoder ops |
| Decode hot path | KV read + attention | `GGML_OP_BDH_SYN_HEAD` (+ optional fuse) |

## What this fork adds

- **`LLM_ARCH_BDH`** — model type, tensors, `llama_model_bdh`, graph node budgets.
- **ggml ops** — `BDH_SYN_HEAD`, gated decoder, encoder ReLU paths, LM head; **`GGML_OP_COUNT`** extended accordingly.
- **Synaptic decode** — `bdh_synaptic` on single-token steps; `synchronize()` after tight prefill on GPU paths.
- **Fused path** — `BDH_SYN_FUSED` (CPU/CUDA) for decode throughput.
- **Server** — per-slot `n_ctx`, chat trim for BDH when runtime context &lt; train window.
- **Conversion** — `scripts/convert_bdh_pt_to_gguf.py`.

## Verified (2026-11-16)

Staging build: CUDA **sm_60 + sm_70**, OpenBLAS, `llama-server`.  
Integration weights: Polish **`bdh-100m-bpe8k-pl`** (~100M, BPE 8k).

| Check | Result |
|-------|--------|
| PL chat template — greedy top-1 vs PyTorch reference | pass |
| `llama-server` chat — no role/garbage regressions | 3× pass |
| Batch prefill vs tight ubatch greedy chain | pass |
| Prompt length beyond train window (2048 ctx + shift) | 2× pass |
| CUDA build (`llama-server` target) | 137/137 |

Details: [docs/BDH-TEST-RESULTS.md](docs/BDH-TEST-RESULTS.md).

## Test checkpoints

BDH (Baby Dragon Hatchling) is by Pathway: [arXiv:2509.26507](https://arxiv.org/abs/2509.26507), [pathwaycom/bdh](https://github.com/pathwaycom/bdh) (MIT). I ported BDH inference to llama.cpp in this fork and trained the files below for integration tests only.

Checkpoints trained locally for fork integration tests. Output quality is too low for production use, including general Polish generation. Training time was limited. Hardware: one V100; sustained jobs hit thermal throttling. Training a capable BDH model was not feasible under these constraints.

`bdh-100m-bpe8k-pl`: two epochs on a small exercise set from a local on-disk token cache (~17M training tokens in the run log; not a full pretraining corpus). This schedule does not yield a working BDH model.

Checkpoints are unsuitable for concrete tasks and do not answer questions meaningfully. Intended as a baseline for later work with this fork.

If you train a larger BDH model with this stack on your hardware, please tell me in [Issues](https://github.com/pixelcombat/llama.cpp/issues) or [Discussions](https://github.com/pixelcombat/llama.cpp/discussions) so I can try it.

I apologize that I lack hardware sufficient to train a sensible BDH model; I leave that work to the community, and for now I have done only as much as I could.

### Multi-token prediction (MTP)

[Multi-token prediction](https://arxiv.org/abs/2404.19737) adds auxiliary output heads (Transformer / DeepSeek V3 pattern). Compatible with BDH; not trained or run here. Heads require joint training with the model, preferably from initialization on the next larger run, not on the ~100M checkpoint.

Weights are not stored in git. Download from Hugging Face:

**[czescapk/bdh-100m-bpe8k-pl](https://huggingface.co/czescapk/bdh-100m-bpe8k-pl)**

| File | Role |
|------|------|
| `bdh-100m-bpe8k-pl.gguf` | Polish BPE 8k — inference with this fork (primary demo) |
| `bdh-100m-bpe8k-pl.pt` | Same run, PyTorch checkpoint |
| `checkpoints/bdh-100m-bpe8k.pt` | English BPE 8k base (~100M), same scale recipe |
| `checkpoints/bdh-100m-bpe-pl-ext-najlepszy.pt` | Extended Polish training — best saved step |
| `checkpoints/bdh-100m-bpe-pl-ext-epoka-1.pt` | Extended Polish — end of epoch 1 |
| `checkpoints/bdh-100m-bpe-pl-ext-init.pt` | Extended Polish — starting weights for that run |

```bash
pip install -U huggingface_hub
hf download czescapk/bdh-100m-bpe8k-pl --local-dir ./models/bdh-pl
```

Thanks to the Baby Dragon Hatchling (BDH) and llama.cpp authors for this integration work.

## Build

Requires Linux, GCC ≥ 11, CMake ≥ 3.20, optional CUDA for GPU.

```bash
git clone https://github.com/pixelcombat/llama.cpp.git
cd llama.cpp && git checkout bdh-v1.0.0

cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON -DGGML_BLAS=ON -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="60;70"
cmake --build build-cuda -j --target llama-server
```

CPU-only: omit `GGML_CUDA` and the CUDA arch flag; use `build/` instead of `build-cuda/`.

Convert your own checkpoint:

```bash
python3 scripts/convert_bdh_pt_to_gguf.py path/to/model.pt -o model.gguf
```

## Run (BDH example)

```bash
./build-cuda/bin/llama-server \
  -m ./models/bdh-pl/bdh-100m-bpe8k-pl.gguf \
  --ctx-size 2048 --context-shift -ngl 99 --parallel 1
```

Use a plain user/assistant template or `--chat-template-file` if tokenizer metadata does not match your UI.

| Variable | Default | Notes |
|----------|---------|--------|
| `BDH_SYN_FUSED` | on | Fused synaptic head |
| `BDH_RS_DIRECT` | on | Direct recurrent state I/O |
| `BDH_SYNAPTIC` | on | Grid updates on decode (`0` = off) |

## Changelog

[CHANGELOG-BDH.md](CHANGELOG-BDH.md) — commits on top of v1.0.0.

## License

**MIT** — same as [llama.cpp](https://github.com/ggml-org/llama.cpp) ([LICENSE](LICENSE)). Model weights are distributed separately on Hugging Face under the license stated in the model card there.
# BDH architecture (Small Dragon Hatchling)

Community fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) on branch **`bdh-v1.0.0`**, based on upstream tag **`v1.0.0`**.

Adds **`LLM_ARCH_BDH`**: recurrent synaptic grid state, tight micro-batch prefill (`n_ubatch` = prefix length), and fused decode op `GGML_OP_BDH_SYN_HEAD` (when enabled).

## Requirements

- Linux, GCC ≥ 11, CMake ≥ 3.20
- Python 3.10+ for GGUF conversion (`torch`)
- Optional: CUDA toolkit for GPU builds (sm_60+ and sm_70+ in sample scripts)

## Build

```bash
git checkout bdh-v1.0.0

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON -DGGML_BLAS=ON
cmake --build build -j --target llama-server

# CUDA example
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON -DGGML_BLAS=ON -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="60;70"
cmake --build build-cuda -j --target llama-server
```

## GGUF

```bash
python3 scripts/convert_bdh_pt_to_gguf.py /path/to/bdh-100m-bpe8k.pt -o model.gguf
```

Requires `torch` and this repo’s `gguf-py/`.

## Run server

```bash
./build-cuda/bin/llama-server -m model.gguf --ctx-size 2048 -ngl 99
```

For chat templates, use a plain user/assistant format (not ChatML baked into the GGUF) or pass `--chat-template-file`.

## Runtime env

| Variable | Effect |
|----------|--------|
| `BDH_SYN_FUSED` | Fused synaptic head (default **on**; set `0` to disable). |
| `BDH_RS_DIRECT` | Fast RS state view (default on). |
| `BDH_SYNAPTIC` | Set `0` to disable recurrent updates. |
| `BDH_N_THREADS` | ggml CPU thread count for probes |

## Regression tests

See `doc/BDH-TEST-RESULTS.md` (pytest harness used before release). You can reproduce parity checks with a local PyTorch reference and `llama-cli` / `llama-server` on a BDH GGUF.

## Changelog

`CHANGELOG-BDH.md` — BDH-specific commits on top of `v1.0.0`.

## License

MIT — same as llama.cpp (`LICENSE`).
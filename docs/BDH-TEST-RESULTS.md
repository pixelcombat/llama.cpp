# BDH regression results (2026-11-16)

Reproducibility (training, SHA-256, commands): [BDH-REPRODUCIBILITY.md](BDH-REPRODUCIBILITY.md).

Upstream base: **v1.0.0** + merge `bcdbac8`, fixes `fd7371e`, `510d025`.  
Build: `build-bdh-cuda-staging` (CUDA sm_60/70, OpenBLAS, `llama-server`).

Model: `bdh-100m-bpe8k-pl.gguf` (local weights, not shipped in repo).

## Pytest (Turbo OS harness)

| Test file | Result |
|-----------|--------|
| `test_bdh_pl_streaming_chat_top1.py` | **PASS** |
| `test_bdh_server_chat_coherent.py` (staging `:18298`) | **3× PASS** |
| `test_bdh_batch_prefill_generation.py` | **PASS** |
| `test_bdh_server_chat_long_context.py` | **2× PASS** |
| `test_bdh_cuda_staging_parity.py` | **SKIP** (English GGUF fixture absent on runner; PL parity covered above) |

Environment: `CUDA_VISIBLE_DEVICES=1`, `-ngl 99`, `BDH_SYN_FUSED=1`, PL chat template.

## Build

```
cmake --build build-bdh-cuda-staging -j --target llama-server  → OK (137/137)
```

## Performance

See operator notes in separate perf doc; this release gate targets **correctness** (logits top-1, chat coherence, long context >512 tokens), not TPS records.
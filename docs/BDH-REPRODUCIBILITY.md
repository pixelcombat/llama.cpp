# BDH reproducibility (published checkpoints)

Facts recorded in this repository and operator run logs. Where a field is not stored in the fork, it is marked **not recorded**.

## Test checkpoint `bdh-100m-bpe8k-pl` (Polish BPE 8k)

| Field | Value |
|-------|--------|
| Training script in this repo | **not recorded** (operator tooling outside fork) |
| Base init | English `bdh-100m-bpe8k.pt` checkpoint (Pathway-scale recipe; training provenance **not recorded** in fork) |
| Data | Local on-disk token memmap exercise set; **~17M training tokens** logged at run time (full corpus identity **not recorded** in fork) |
| Tokenizer | BPE 8k (`bpe8k.json`, shared with English BPE-8k recipe) |
| Epochs completed | **2** (job configured for 8; stopped after epoch 2) |
| Block size | 512 tokens |
| Optimizer | AdamW, weight decay 0.1, LR 1e-4, cosine schedule, warmup 100 updates |
| Micro-batch | 4 × grad accum 8 (effective 32 sequences per optimizer step) |
| Precision | FP32 training (no GradScaler) |
| Hardware | 1× NVIDIA Tesla V100-PCIE-32GB |
| Thermal throttling | observed on sustained runs (operator note) |
| Git commit (fork integration tests) | see branch `bdh-v1.0.0` at publish time |

## GGUF conversion

```bash
python3 scripts/convert_bdh_pt_to_gguf.py path/to/bdh-100m-bpe8k-pl.pt -o bdh-100m-bpe8k-pl.gguf
```

Tokenizer symlink or path to `bpe8k.json` must match the checkpoint (see script help).

## Inference (example)

```bash
./build-cuda/bin/llama-server \
  -m ./models/bdh-pl/bdh-100m-bpe8k-pl.gguf \
  --ctx-size 2048 --context-shift -ngl 99 --parallel 1
```

## Published weight files (SHA-256, local publish set)

| Hub path | Size (bytes) | SHA-256 |
|----------|--------------|---------|
| `bdh-100m-bpe8k-pl.gguf` | 436578144 | `1a37d9ad50708a93d9738b93d6aeed9959b7214780345ef5e6bd6a0547baf515` |
| `bdh-100m-bpe8k-pl.pt` | 436276621 | `9de4213eb93f2551be2b7808cea9a63195af6be945076d516347f9c62e07f9c6` |
| `checkpoints/bdh-100m-bpe8k.pt` | 436276649 | `fd691601b54802ec0b028c077cdf13f7c789b6caae793cedca6f63f2b6fd630e` |
| `checkpoints/bdh-100m-bpe-pl-ext-najlepszy.pt` | 453053777 | `0585bfdf95257006044af52c7f44a9b3714d6c11a12a07007cab8a139c97d961` |
| `checkpoints/bdh-100m-bpe-pl-ext-epoka-1.pt` | 453053753 | `a933888b78a6c9cd2d3e15babf8a05c5dd5adb670d195a0e3f85f6e52186ca5d` |
| `checkpoints/bdh-100m-bpe-pl-ext-init.pt` | 453053269 | `210c170265b4f46fe2360dd1473f29515fcad0137f1254cb49c682d3c010e27f` |

## Integration tests (fork)

See [BDH-TEST-RESULTS.md](BDH-TEST-RESULTS.md). Summary at time of `bdh-v1.0.0` publish gate:

- PL streaming greedy top-1 vs PyTorch: **PASS**
- `llama-server` chat coherence (staging): **3× PASS**
- Batch prefill generation chain: **PASS**
- Long context (>512, context shift): **2× PASS**
- CUDA `llama-server` build: **137/137** targets OK
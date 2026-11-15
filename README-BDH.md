# BDH architecture (Small Dragon Hatchling) in this fork

Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with **`LLM_ARCH_BDH`**: recurrent synaptic grid state, tight micro-batch (`n_ubatch` = prefix length), and optional fused decode op `GGML_OP_BDH_SYN_HEAD`.

## Build

```bash
cmake -S . -B build-bdh -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
cmake --build build-bdh -j --target llama llama-server

# CUDA (V100/P100 example)
./scripts/build-bdh-cuda.sh
```

## GGUF

```bash
python3 scripts/convert_bdh_pt_to_gguf.py /path/to/bdh-100m-bpe8k.pt -o model.gguf
```

Requires `torch` and this repo’s `gguf-py/`.

## Runtime env

| Variable | Effect |
|----------|--------|
| `BDH_SYN_FUSED` | Fused synaptic head (default **on**; set `0` to disable). |
| `BDH_RS_DIRECT` | Fast RS state view (default on). |
| `BDH_SYNAPTIC` | Set `0` to disable recurrent updates. |

## Tests (external harness)

See Turbo OS addon `dodatki/llama-cpp-bdh/` in the operator workspace: pytest parity, CUDA incremental, probes.

## License

Same as llama.cpp (MIT). See `LICENSE`.
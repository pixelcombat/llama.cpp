#!/usr/bin/env python3
"""Konwersja checkpointu BDH (.pt) do GGUF dla forka llama.cpp-bdh (architektura bdh)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

DEFAULT_FORK = Path.home() / "aik/apps/llama.cpp-bdh/src"
GGUF_PY = Path(os.environ.get("LLAMA_CPP_BDH_ROOT", DEFAULT_FORK)) / "gguf-py"
if str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))

from gguf import GGUFWriter, LlamaFileType  # noqa: E402
from gguf.vocab import BpeVocab, SpecialVocab  # noqa: E402


def strip_prefix(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        key = k.removeprefix("_orig_mod.")
        out[key] = v
    return out


def find_tokenizer_json(pt_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    d = pt_path.parent
    for name in ("tokenizer.json", "bpe8k.json"):
        p = d / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"Brak tokenizer.json / bpe8k.json obok {pt_path}")


def load_vocab_writer(writer: GGUFWriter, tokenizer_json: Path) -> None:
    tmp = tokenizer_json.parent
    use_tmp: Path | None = None
    if tokenizer_json.name != "tokenizer.json":
        import tempfile

        use_tmp = Path(tempfile.mkdtemp(prefix="bdh-tok-", dir=os.environ.get("TMPDIR", "/tmp")))
        (use_tmp / "tokenizer.json").write_bytes(tokenizer_json.read_bytes())
        tmp = use_tmp
    try:
        vocab = BpeVocab(tmp)
        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []
        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(int(toktype))
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("default")
        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        writer.add_token_types(toktypes)
        special = SpecialVocab(tmp, load_merges=True, n_vocab=len(tokens))
        special.add_to_gguf(writer, quiet=True)
    finally:
        if use_tmp is not None:
            (use_tmp / "tokenizer.json").unlink(missing_ok=True)
            use_tmp.rmdir()


def convert_pt_to_gguf(
    pt_path: Path,
    out_path: Path,
    tokenizer_json: Path | None = None,
    context_length: int = 512,
) -> dict[str, object]:
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck or "config" not in ck:
        raise ValueError(f"Nieznany format checkpointu: {pt_path}")
    cfg = ck["config"]
    sd = strip_prefix(ck["model"])

    n_layer = int(cfg["n_layer"])
    n_embd = int(cfg["n_embd"])
    n_head = int(cfg["n_head"])
    vocab_size = int(cfg["vocab_size"])
    sparse_mult = int(cfg.get("mlp_internal_dim_multiplier", 128))
    norm_eps = float(cfg.get("layer_norm_epsilon", 1e-5))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(out_path), "bdh")
    writer.add_name(pt_path.stem)
    writer.add_file_type(LlamaFileType.ALL_F32)
    writer.add_block_count(n_layer)
    writer.add_context_length(context_length)
    writer.add_embedding_length(n_embd)
    writer.add_feed_forward_length(0)
    writer.add_head_count(n_head)
    writer.add_vocab_size(vocab_size)
    writer.add_layer_norm_eps(norm_eps)
    writer.add_int32("bdh.sparse_dim_multiplier", sparse_mult)

    tok_json = find_tokenizer_json(pt_path, tokenizer_json)
    load_vocab_writer(writer, tok_json)

    # gguf-py zapisuje wymiary w odwrotnej kolejności względem numpy — układ jak w PyTorch.
    embed = sd["embed.weight"].float().contiguous().numpy()
    writer.add_tensor("token_embd.weight", embed)
    writer.add_tensor("output.weight", sd["lm_head"].float().t().contiguous().numpy())

    enc = sd["encoder"].float().contiguous().numpy()
    enc_v = sd["encoder_v"].float().contiguous().numpy()
    dec = sd["decoder"].float().contiguous().numpy()
    freqs = sd["attn.freqs"].float().contiguous().numpy()

    writer.add_tensor("bdh.encoder", enc)
    writer.add_tensor("bdh.encoder_v", enc_v)
    # gguf-py odwraca wymiary względem numpy: zapis dec.T [D,nh·N] → meta (nh·N,D) w loaderze
    writer.add_tensor("bdh.decoder", np.ascontiguousarray(dec.T))
    writer.add_tensor("bdh.attn.freqs", freqs)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    meta = {
        "pt": str(pt_path),
        "gguf": str(out_path),
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": n_head,
        "vocab_size": vocab_size,
        "sparse_mult": sparse_mult,
        "tokenizer": str(tok_json),
        "bytes": out_path.stat().st_size,
    }
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pt", type=Path, help="Ścieżka do .pt (tylko odczyt)")
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="GGUF wynik (domyślnie obok .pt, ta sama nazwa .gguf)",
    )
    ap.add_argument("--tokenizer", type=Path, default=None, help="tokenizer.json / bpe8k.json")
    ap.add_argument("--ctx", type=int, default=512, help="context length w metadanych")
    ap.add_argument("--meta-json", type=Path, default=None, help="Zapisz metadane konwersji jako JSON")
    args = ap.parse_args()

    pt_path = args.pt.expanduser().resolve()
    if not pt_path.is_file():
        print(f"BLOKADA: brak pliku {pt_path}", file=sys.stderr)
        return 2
    out_path = args.out or pt_path.with_suffix(".gguf")
    meta = convert_pt_to_gguf(pt_path, out_path, args.tokenizer, args.ctx)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    if args.meta_json:
        args.meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
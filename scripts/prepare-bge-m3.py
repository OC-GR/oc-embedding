#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare existing BGE-M3 weights for offline SentenceTransformers loading."""

import argparse
import json
import os
from pathlib import Path
import shutil


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output == source or source in output.parents:
        parser.error("--output must be outside the original model directory")
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be new or empty")

    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "xlm-roberta" or config.get("hidden_size") != 1024:
        parser.error("expected BGE-M3: XLM-RoBERTa with hidden_size=1024")
    if not (source / "tokenizer.json").is_file():
        parser.error("missing tokenizer.json")
    if not any((source / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        parser.error("missing model.safetensors or pytorch_model.bin")
    max_positions = config.get("max_position_embeddings", 8194) - 2
    if not 1 <= args.max_seq_length <= max_positions:
        parser.error(f"--max-seq-length must be between 1 and {max_positions}")

    output.mkdir(parents=True, exist_ok=True)
    for original in source.rglob("*"):
        relative = original.relative_to(source)
        if any(part.startswith(".") for part in relative.parts) or not original.is_file():
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Only immutable large files are linked. Configuration is copied so
        # editing the prepared view can never alter the original config.
        if original.stat().st_size > 1024 * 1024:
            try:
                os.link(original, destination)
            except OSError:
                shutil.copy2(original, destination)
        else:
            shutil.copy2(original, destination)

    write_json(output / "modules.json", [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    ])
    write_json(output / "1_Pooling/config.json", {
        "word_embedding_dimension": 1024,
        "pooling_mode_cls_token": True,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
    })
    write_json(output / "sentence_bert_config.json", {
        "max_seq_length": args.max_seq_length,
        "do_lower_case": False,
    })
    write_json(output / "oc-model.json", {
        "model": "BAAI/bge-m3",
        "weights_source": str(source),
        "pooling": "cls",
        "max_seq_length": args.max_seq_length,
        "normalization": "L2 normalization performed by Infinity",
        "weights_downloaded": False,
    })
    print(json.dumps({"status": "PREPARED", "model_dir": str(output),
                      "pooling": "cls", "max_seq_length": args.max_seq_length}))


if __name__ == "__main__":
    main()

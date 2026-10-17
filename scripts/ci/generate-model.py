#!/usr/bin/env python3
"""Create a tiny deterministic XLM-RoBERTa fixture without downloading a model.

The generated weights are synthetic and only exercise model loading and inference.
They do not measure BGE-M3 accuracy or performance. Python's standard library is enough.
"""

import argparse
import json
import math
from pathlib import Path
import random
import struct


DIMENSIONS = 16
MAX_LENGTH = 32
VOCAB = [
    "<s>", "<pad>", "</s>", "<unk>", "hello", "world", "another",
    "embedding", "text", "account", "password", "reset", "order", "parcel",
    "track", "test", "example", "different", "账户", "密码", "重置",
]


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Fixture directory must be empty: {output}")

    config = {
        "architectures": ["XLMRobertaModel"], "model_type": "xlm-roberta",
        "vocab_size": len(VOCAB), "hidden_size": DIMENSIONS,
        "num_hidden_layers": 1, "num_attention_heads": 4,
        "intermediate_size": DIMENSIONS * 2, "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": MAX_LENGTH + 2, "type_vocab_size": 1,
        "initializer_range": 0.02, "layer_norm_eps": 1e-5,
        "pad_token_id": 1, "bos_token_id": 0, "eos_token_id": 2,
        "position_embedding_type": "absolute", "torch_dtype": "float32",
    }
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [
            {"id": i, "content": token, "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": True}
            for i, token in enumerate(VOCAB[:4])
        ],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                       {"Sequence": {"id": "A", "type_id": 0}},
                       {"SpecialToken": {"id": "</s>", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                     {"Sequence": {"id": "A", "type_id": 0}},
                     {"SpecialToken": {"id": "</s>", "type_id": 0}},
                     {"SpecialToken": {"id": "</s>", "type_id": 0}},
                     {"Sequence": {"id": "B", "type_id": 0}},
                     {"SpecialToken": {"id": "</s>", "type_id": 0}}],
            "special_tokens": {
                token: {"id": token, "ids": [i], "tokens": [token]}
                for token, i in (("<s>", 0), ("</s>", 2))
            },
        },
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": dict(zip(VOCAB, range(len(VOCAB)))),
                  "unk_token": "<unk>"},
    }
    rng = random.Random(20261017)
    tensors = {}

    def tensor(name, shape, values=None):
        count = math.prod(shape)
        if values is None:
            values = [rng.uniform(-0.1, 0.1) for _ in range(count)]
        assert len(values) == count
        tensors[name] = (shape, struct.pack(f"<{count}f", *values))

    def layer_norm(prefix):
        tensor(prefix + ".weight", [DIMENSIONS], [1.0] * DIMENSIONS)
        tensor(prefix + ".bias", [DIMENSIONS], [0.0] * DIMENSIONS)

    tensor("embeddings.word_embeddings.weight", [len(VOCAB), DIMENSIONS])
    tensor("embeddings.position_embeddings.weight", [MAX_LENGTH + 2, DIMENSIONS])
    tensor("embeddings.token_type_embeddings.weight", [1, DIMENSIONS])
    layer_norm("embeddings.LayerNorm")
    prefix = "encoder.layer.0."
    for name in ("query", "key", "value"):
        values = None
        if name == "value":
            values = [float(i == j) for i in range(DIMENSIONS) for j in range(DIMENSIONS)]
        tensor(prefix + f"attention.self.{name}.weight", [DIMENSIONS, DIMENSIONS], values)
        tensor(prefix + f"attention.self.{name}.bias", [DIMENSIONS], [0.0] * DIMENSIONS)
    tensor(prefix + "attention.output.dense.weight", [DIMENSIONS, DIMENSIONS],
           [0.5 * (i == j) for i in range(DIMENSIONS) for j in range(DIMENSIONS)])
    tensor(prefix + "attention.output.dense.bias", [DIMENSIONS], [0.0] * DIMENSIONS)
    layer_norm(prefix + "attention.output.LayerNorm")
    tensor(prefix + "intermediate.dense.weight", [DIMENSIONS * 2, DIMENSIONS])
    tensor(prefix + "intermediate.dense.bias", [DIMENSIONS * 2], [0.0] * (DIMENSIONS * 2))
    tensor(prefix + "output.dense.weight", [DIMENSIONS, DIMENSIONS * 2])
    tensor(prefix + "output.dense.bias", [DIMENSIONS], [0.0] * DIMENSIONS)
    layer_norm(prefix + "output.LayerNorm")

    header = {"__metadata__": {"format": "pt", "purpose": "synthetic CI fixture"}}
    chunks = []
    offset = 0
    for name, (shape, data) in sorted(tensors.items()):
        header[name] = {"dtype": "F32", "shape": shape,
                        "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    (output / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))
    for name, value in {
        "config.json": config, "tokenizer.json": tokenizer,
        "sentence_bert_config.json": {"max_seq_length": MAX_LENGTH},
        "config_sentence_transformers.json": {"prompts": {}},
    }.items():
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(f"Generated synthetic {DIMENSIONS}-dimensional XLM-RoBERTa fixture: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    generate(parser.parse_args().output.resolve())

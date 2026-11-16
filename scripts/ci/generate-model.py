#!/usr/bin/env python3
"""Create tiny, untrained BERT weights for offline runtime checks (stdlib only)."""

import argparse
import json
import math
from pathlib import Path
import random
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit("Refusing to overwrite a nonempty fixture directory")

    rng = random.Random(20261116)
    dim = 16
    vocab_words = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "reset",
        "account", "password", "track", "order", "parcel", "a", "document", "about",
        "semantic", "search", "this", "is", "test", "another", "sentence", ".",
        "你", "好", "世", "界",
    ]
    vocab = {word: i for i, word in enumerate(vocab_words)}

    def write_json(path, value):
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

    write_json("config.json", {
        "architectures": ["BertModel"], "model_type": "bert", "hidden_size": dim,
        "num_hidden_layers": 1, "num_attention_heads": 4, "intermediate_size": 32,
        "vocab_size": len(vocab), "max_position_embeddings": 64, "type_vocab_size": 2,
        "hidden_act": "gelu", "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0, "initializer_range": 0.02,
        "layer_norm_eps": 1e-12, "pad_token_id": 0, "torch_dtype": "float32",
    })
    write_json("tokenizer.json", {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": i, "content": word, "single_word": False,
                          "lstrip": False, "rstrip": False, "normalized": False,
                          "special": True} for i, word in enumerate(vocab_words[:5])],
        "normalizer": {"type": "BertNormalizer", "clean_text": True,
                       "handle_chinese_chars": True, "strip_accents": None,
                       "lowercase": True},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                       {"Sequence": {"id": "A", "type_id": 0}},
                       {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                     {"Sequence": {"id": "A", "type_id": 0}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                     {"Sequence": {"id": "B", "type_id": 1}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 1}}],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]},
            },
        },
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": True},
        "model": {"type": "WordPiece", "unk_token": "[UNK]",
                  "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
                  "vocab": vocab},
    })
    special = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]",
               "cls_token": "[CLS]", "mask_token": "[MASK]"}
    write_json("tokenizer_config.json", {
        "tokenizer_class": "BertTokenizerFast", "model_max_length": 32,
        "do_lower_case": True, **special,
    })
    write_json("special_tokens_map.json", special)
    (output / "vocab.txt").write_text("\n".join(vocab_words) + "\n")
    write_json("sentence_bert_config.json", {"max_seq_length": 32, "do_lower_case": False})
    write_json("config_sentence_transformers.json", {"prompts": {}, "default_prompt_name": None})
    write_json("modules.json", [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
    ])
    (output / "2_Normalize").mkdir()
    write_json("1_Pooling/config.json", {
        "word_embedding_dimension": dim, "pooling_mode_cls_token": True,
        "pooling_mode_mean_tokens": False, "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False, "pooling_mode_lasttoken": False,
        "include_prompt": True,
    })

    header = {"__metadata__": {"format": "pt"}}
    blobs = []
    offset = 0

    def tensor(name, shape, kind="random", scale=0.08):
        nonlocal offset
        size = math.prod(shape)
        if kind == "ones":
            values = [1.0] * size
        elif kind == "zeros":
            values = [0.0] * size
        elif kind == "identity":
            values = [scale if i == j else 0.0
                      for i in range(shape[0]) for j in range(shape[1])]
        else:
            values = [rng.uniform(-scale, scale) for _ in range(size)]
        data = struct.pack("<" + "f" * size, *values)
        header[name] = {"dtype": "F32", "shape": shape,
                        "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)

    tensor("embeddings.word_embeddings.weight", [len(vocab), dim], scale=0.2)
    tensor("embeddings.position_embeddings.weight", [64, dim])
    tensor("embeddings.token_type_embeddings.weight", [2, dim])
    tensor("embeddings.LayerNorm.weight", [dim], "ones")
    tensor("embeddings.LayerNorm.bias", [dim], "zeros")
    prefix = "encoder.layer.0."
    for name in ["query", "key", "value"]:
        tensor(prefix + "attention.self." + name + ".weight", [dim, dim],
               "identity" if name == "value" else "random", scale=1.0 if name == "value" else 0.08)
        tensor(prefix + "attention.self." + name + ".bias", [dim], "zeros")
    tensor(prefix + "attention.output.dense.weight", [dim, dim], "identity", scale=0.5)
    tensor(prefix + "attention.output.dense.bias", [dim], "zeros")
    tensor(prefix + "attention.output.LayerNorm.weight", [dim], "ones")
    tensor(prefix + "attention.output.LayerNorm.bias", [dim], "zeros")
    tensor(prefix + "intermediate.dense.weight", [32, dim])
    tensor(prefix + "intermediate.dense.bias", [32], "zeros")
    tensor(prefix + "output.dense.weight", [dim, 32])
    tensor(prefix + "output.dense.bias", [dim], "zeros")
    tensor(prefix + "output.LayerNorm.weight", [dim], "ones")
    tensor(prefix + "output.LayerNorm.bias", [dim], "zeros")
    tensor("pooler.dense.weight", [dim, dim], "identity", scale=1.0)
    tensor("pooler.dense.bias", [dim], "zeros")
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    (output / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))
    write_json("fixture.json", {"kind": "untrained-offline-BERT", "dimension": dim,
                                "scope": "runtime and API correctness; not semantic quality"})
    print(json.dumps({"path": str(output.resolve()), "dimension": dim,
                      "weights_bytes": (output / "model.safetensors").stat().st_size}))


if __name__ == "__main__":
    main()

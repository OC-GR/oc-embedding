#!/usr/bin/env python3
"""Run the upstream CPU suite with a local model, without editing upstream tests."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


class LocalModel:
    def __init__(self, model_dir):
        self.model_dir = str(model_dir)

    @pytest.hookimpl(trylast=True)
    def pytest_configure(self, config):
        # The upstream conftest initializes DEFAULT_BERT_MODEL before this hook;
        # test modules consume it afterwards, at collection time.
        pytest.DEFAULT_BERT_MODEL = self.model_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve(strict=True)
    assert (model_dir / "model.safetensors").is_file()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    os.environ["INFINITY_ANONYMOUS_USAGE_STATS"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "2"
    package_dir = Path(__file__).resolve().parents[2] / "libs" / "infinity_emb"
    tests = [
        "tests/unit_test/test_args.py",
        "tests/unit_test/test_cli.py",
        "tests/unit_test/test_infinity_server.py",
        "tests/unit_test/test_log_handler.py",
        "tests/unit_test/test_sync_engine.py",
        "tests/unit_test/test_engine.py::test_async_api_debug",
        "tests/unit_test/test_engine.py::test_async_api_torch_double_launch",
        "tests/unit_test/test_engine.py::test_args_between_array_and_engine_same",
        "tests/unit_test/fastapi_schemas",
        "tests/unit_test/inference/test_caching_layer.py",
        "tests/unit_test/transformer/test_utils.py",
        "tests/end_to_end/test_authentication.py",
        "tests/end_to_end/test_api_with_dummymodel.py",
        "tests/end_to_end/test_sentence_transformers.py",
    ]
    # EngineArgs derives the public model id from the final two path parts.
    # A relative local alias lets the original /models assertion remain intact.
    previous_dir = Path.cwd()
    with TemporaryDirectory(prefix="oc-embedding-tests-") as temp_dir:
        model_alias = Path(temp_dir) / "models" / "ci-model"
        model_alias.parent.mkdir()
        model_alias.symlink_to(model_dir, target_is_directory=True)
        try:
            os.chdir(temp_dir)
            result = pytest.main(
                [*(str(package_dir / test) for test in tests), "-v", "-m", "not performance",
                 "-k", "not test_cli_preload and not test_sync_engine_on_model"],
                plugins=[LocalModel("models/ci-model")],
            )
        finally:
            os.chdir(previous_dir)
    raise SystemExit(result)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify a CPU image using a real, local SentenceTransformer forward pass."""

import argparse
import base64
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

INPUTS = ["hello world", "reset account password", "track order parcel", "hello world"]
MODEL_NAME = "ci-model"
API_KEY = "offline-ci-test-token"
EPS = 1e-5


def docker(*args, check=True):
    result = subprocess.run(["docker", *args], capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result


def exists(kind, name):
    return docker(kind, "inspect", name, check=False).returncode == 0


def request(base, path, payload=None, authenticated=True, expected=200):
    headers = {"Content-Type": "application/json"}
    if authenticated:
        headers["Authorization"] = "Bearer " + API_KEY
    req = urllib.request.Request(base + path, headers=headers,
                                 data=None if payload is None else json.dumps(payload).encode())
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read()
    if status != expected:
        raise AssertionError(f"{path}: expected HTTP {expected}, got {status}: {body[:1000]!r}")
    return json.loads(body)


def check_vectors(vectors, dimension):
    assert len(vectors) == len(INPUTS), len(vectors)
    for vector in vectors:
        assert len(vector) == dimension, len(vector)
        assert all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector)
        assert abs(math.sqrt(sum(x * x for x in vector)) - 1.0) < 1e-4
    assert max(abs(a - b) for a, b in zip(vectors[0], vectors[-1])) < EPS
    assert max(abs(a - b) for a, b in zip(vectors[0], vectors[1])) > 1e-4


def max_error(actual, expected):
    assert len(actual) == len(expected)
    assert all(len(av) == len(ev) for av, ev in zip(actual, expected))
    return max(abs(a - b) for av, ev in zip(actual, expected) for a, b in zip(av, ev))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--expected-architecture", choices=["amd64", "arm64"], required=True)
    parser.add_argument("--expected-version", default="0.0.77")
    parser.add_argument("--expected-revision")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve(strict=True)
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.container_name), "invalid container name"
    names = [args.container_name, args.container_name + "-reference", args.container_name + "-version"]
    for name in names:
        if exists("container", name):
            raise SystemExit(f"Refusing to replace existing container: {name}")
    info = json.loads(docker("image", "inspect", args.image).stdout)[0]
    assert info["Os"] == "linux" and info["Architecture"] == args.expected_architecture
    if args.expected_revision:
        assert (info["Config"].get("Labels") or {}).get("org.opencontainers.image.revision") == args.expected_revision
    dimension = json.loads((model_dir / "config.json").read_text())["hidden_size"]
    platform = "linux/" + args.expected_architecture
    offline = ["--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
               "--env", "HF_HUB_DISABLE_TELEMETRY=1", "--env", "INFINITY_ANONYMOUS_USAGE_STATS=0",
               "--env", "TOKENIZERS_PARALLELISM=false", "--env", "OMP_NUM_THREADS=2"]
    mount = ["--mount", f"type=bind,source={model_dir},target=/models/ci-model,readonly"]
    try:
        version = docker("run", "--rm", "--pull", "never", "--platform", platform,
                         "--network", "none", "--name", names[2], "--entrypoint", "python",
                         args.image, "-c", "import importlib.metadata; print(importlib.metadata.version('infinity-emb'))").stdout.strip()
        assert version == args.expected_version, version
        reference_code = (
            "import json; from sentence_transformers import SentenceTransformer; "
            "m=SentenceTransformer('/models/ci-model',device='cpu'); "
            f"v=m.encode({INPUTS!r},normalize_embeddings=True,show_progress_bar=False); "
            "print('REFERENCE_JSON='+json.dumps(v.tolist()))"
        )
        reference_result = docker("run", "--rm", "--pull", "never", "--platform", platform,
                                  "--network", "none", "--name", names[1], *offline, *mount,
                                  "--entrypoint", "python", args.image, "-c", reference_code)
        reference_lines = [line.removeprefix("REFERENCE_JSON=") for line in reference_result.stdout.splitlines()
                           if line.startswith("REFERENCE_JSON=")]
        assert len(reference_lines) == 1, reference_result.stdout
        reference = json.loads(reference_lines[0])
        check_vectors(reference, dimension)

        docker("run", "--detach", "--pull", "never", "--platform", platform,
               "--name", names[0], "--label", "io.oc-embedding.ci=true", *offline, *mount,
               "--publish", "127.0.0.1::7997", args.image, "v2",
               "--model-id", "/models/ci-model", "--served-model-name", MODEL_NAME,
               "--engine", "torch", "--device", "cpu", "--dtype", "float32",
               "--pooling-method", "cls", "--batch-size", "2", "--no-bettertransformer",
               "--api-key", API_KEY, "--host", "0.0.0.0", "--port", "7997")
        published = docker("port", names[0], "7997/tcp").stdout.strip()
        assert re.fullmatch(r"127\.0\.0\.1:\d+", published), published
        base = "http://" + published
        deadline = time.monotonic() + 120
        while True:
            try:
                health = request(base, "/health", authenticated=False)
                assert isinstance(health.get("unix"), (float, int)), health
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Infinity did not become ready within 120 seconds")
                assert docker("inspect", "--format", "{{.State.Running}}", names[0]).stdout.strip() == "true", "container exited"
                time.sleep(0.5)

        request(base, "/models", authenticated=False, expected=401)
        models = request(base, "/models")["data"]
        assert len(models) == 1 and models[0]["id"] == MODEL_NAME, models
        # ModelInfo filters out dtype/revision even though the route builds them.
        assert models[0]["backend"] == "torch" and "embed" in models[0]["capabilities"], models
        assert isinstance(models[0]["stats"], dict), models
        payload = {"model": MODEL_NAME, "input": INPUTS, "encoding_format": "float"}
        response = request(base, "/embeddings", payload)
        assert response["model"] == MODEL_NAME and response["object"] == "list", response
        assert [item["index"] for item in response["data"]] == list(range(len(INPUTS)))
        vectors = [item["embedding"] for item in response["data"]]
        check_vectors(vectors, dimension)
        reference_error = max_error(vectors, reference)
        assert reference_error < EPS, reference_error
        single = request(base, "/embeddings", {"model": MODEL_NAME, "input": INPUTS[0]})
        assert len(single["data"]) == 1
        assert max_error([single["data"][0]["embedding"]], [vectors[0]]) < EPS
        encoded = request(base, "/embeddings", {**payload, "encoding_format": "base64"})
        decoded = [list(struct.unpack("<" + "f" * dimension, base64.b64decode(item["embedding"])))
                   for item in encoded["data"]]
        base64_error = max_error(decoded, vectors)
        assert base64_error < EPS, base64_error
        request(base, "/embeddings", {"model": MODEL_NAME, "input": []}, expected=422)
        request(base, "/embeddings", {**payload, "encoding_format": "hex"}, expected=422)
        # Upstream resolves any requested name when only one engine is loaded.
        fallback = request(base, "/embeddings", {"model": "not-deployed", "input": INPUTS[0]})
        assert fallback["model"] == MODEL_NAME
        assert max_error([fallback["data"][0]["embedding"]], [vectors[0]]) < EPS
        print(json.dumps({"status": "PASS", "platform": platform, "version": version,
                          "source_sha": args.expected_revision, "input_count": len(INPUTS),
                          "dimension": dimension, "reference_max_abs_error": reference_error,
                          "base64_max_abs_error": base64_error,
                          "scope": "untrained offline fixture; real Torch forward and API contract"}, indent=2))
    except BaseException:
        if exists("container", names[0]):
            logs = docker("logs", names[0], check=False)
            print(logs.stdout + logs.stderr, file=sys.stderr)
        raise
    finally:
        for name in names:
            if exists("container", name):
                docker("container", "rm", "--force", name)
            assert not exists("container", name), f"test container remains: {name}"


if __name__ == "__main__":
    main()

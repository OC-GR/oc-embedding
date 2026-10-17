#!/usr/bin/env python3
"""Run real offline inference against a temporary CPU image and synthetic fixture."""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


INPUTS = ["hello world", "reset account password", "track order parcel", "hello world"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def docker(*args, check=True, timeout=60):
    return subprocess.run(["docker", *args], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=check, timeout=timeout)


def request(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=15) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def check_vectors(vectors, dimensions):
    require(isinstance(vectors, list) and len(vectors) == len(INPUTS), "Incorrect vector count")
    for vector in vectors:
        require(isinstance(vector, list) and len(vector) == dimensions, "Incorrect vector dimension")
        require(all(type(x) in (int, float) and math.isfinite(x) for x in vector),
                "Non-finite or nonnumeric vector value")
        require(abs(math.sqrt(sum(x * x for x in vector)) - 1.0) < 1e-4,
                "Vector is not normalized")
    require(max(abs(x - y) for x, y in zip(vectors[0], vectors[3])) < 1e-5,
            "Repeated input produced inconsistent vectors")
    require(max(abs(x - y) for x, y in zip(vectors[0], vectors[1])) > 1e-4,
            "Distinct known-vocabulary inputs produced indistinguishable vectors")


def run(args):
    require(bool(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", args.container_name)),
            "Invalid temporary container name")
    model = args.model_dir.resolve()
    require(model.is_dir() and "," not in str(model), "Invalid model directory")
    config = json.loads((model / "config.json").read_text())
    dimensions = config["hidden_size"]
    image = json.loads(docker("image", "inspect", args.image).stdout)[0]
    require(image["Os"] == "linux" and image["Architecture"] == args.expected_architecture,
            "Image platform does not match the requested architecture")
    require(image["Config"]["Labels"].get("org.opencontainers.image.revision") == args.expected_revision,
            "Image revision label does not match the build commit")
    require(docker("container", "inspect", args.container_name, check=False).returncode != 0,
            "Temporary container name is already in use; refusing to replace it")

    try:
        version = docker("run", "--rm", "--pull=never", "--name", args.container_name,
                         "--platform", "linux/" + args.expected_architecture,
                         args.image, "--version").stdout.strip()
        require("text-embeddings-router" in version, "Image executable version check failed")
        print(version, flush=True)
        docker("run", "--detach", "--pull=never", "--name", args.container_name,
               "--label", "io.oc-embedding.ci=true", "--platform", "linux/" + args.expected_architecture,
               "--publish", "127.0.0.1::80", "--workdir", "/models",
               "--mount", f"type=bind,source={model},target=/models/ci-model,readonly",
               "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
               args.image, "--model-id", "ci-model", "--revision", args.expected_revision,
               "--dtype", "float32", "--pooling", "cls", "--max-batch-tokens", "128",
               "--max-batch-requests", "1", "--tokenization-workers", "2", "--auto-truncate", "false")
        address = docker("port", args.container_name, "80/tcp").stdout.strip()
        require(bool(re.fullmatch(r"127\.0\.0\.1:\d+", address)), "Unexpected container port binding")
        base = "http://" + address
        deadline = time.monotonic() + args.startup_timeout
        while True:
            try:
                request(base, "/health")
                break
            except (URLError, OSError) as error:
                state = json.loads(docker("inspect", "--format", "{{json .State}}", args.container_name).stdout)
                require(state["Running"], f"Container exited during startup: {state}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Image did not become healthy") from error
                time.sleep(1)
        info = request(base, "/info")
        require(info["sha"] == args.expected_revision, "Binary source SHA does not match the build commit")
        require(info["model_id"] == "ci-model", "Service did not load the local fixture")
        require(info["model_dtype"] == "float32", "Unexpected model dtype")
        require(info["model_type"]["embedding"]["pooling"] == "cls", "Unexpected pooling")
        require(info["max_input_length"] == 32, "Unexpected fixture input limit")

        native = request(base, "/embed", {"inputs": INPUTS, "normalize": True, "truncate": False})
        check_vectors(native, dimensions)
        compatible = request(base, "/v1/embeddings", {"input": INPUTS, "model": "ci-model"})
        rows = compatible["data"]
        require(sorted(row["index"] for row in rows) == list(range(len(INPUTS))),
                "OpenAI-compatible response has incorrect indices")
        openai = [row["embedding"] for row in sorted(rows, key=lambda row: row["index"])]
        check_vectors(openai, dimensions)
        difference = max(abs(x - y) for a, b in zip(native, openai) for x, y in zip(a, b))
        require(difference < 1e-5, f"API embedding outputs disagree: {difference}")
        try:
            request(base, "/embed", {"inputs": ["hello " * 100], "truncate": False})
        except HTTPError as error:
            require(error.code == 422, f"Expected 422 for an overlong input; got {error.code}")
        else:
            raise ValueError("Overlong input was accepted despite truncate=false")
        print(json.dumps({"status": "PASS", "image": args.image,
                          "architecture": args.expected_architecture, "source_sha": info["sha"],
                          "dimensions": dimensions, "inputs": len(INPUTS),
                          "api_max_absolute_error": difference,
                          "scope": "synthetic offline model loading and HTTP inference, not BGE-M3 quality"}, indent=2))
    except Exception:
        logs = docker("logs", args.container_name, check=False)
        print(logs.stdout + logs.stderr, flush=True)
        raise
    finally:
        cleanup = docker("rm", "--force", args.container_name, check=False)
        remaining = docker("container", "inspect", args.container_name, check=False)
        require(remaining.returncode != 0, "Temporary container was not removed: " + cleanup.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--expected-architecture", required=True, choices=("arm64", "amd64"))
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--startup-timeout", type=int, default=90)
    run(parser.parse_args())

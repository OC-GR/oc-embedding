#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare a running Infinity service with saved reference embedding outputs."""

import argparse
import json
import math
from pathlib import Path
import urllib.request


def top_three(vectors):
    rankings = []
    for i, query in enumerate(vectors):
        scores = [(sum(a * b for a, b in zip(query, doc)), j)
                  for j, doc in enumerate(vectors) if j != i]
        rankings.append([j for _, j in sorted(scores, reverse=True)[:3]])
    return rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--url", required=True, help="Base URL, such as http://127.0.0.1:18080")
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--label", default="rebuilt-image")
    parser.add_argument("--max-error", type=float, default=1e-5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    expected = reference["vectors"]
    inputs = reference["inputs"]
    request = urllib.request.Request(
        args.url.rstrip("/") + "/embeddings",
        data=json.dumps({"model": args.model, "input": inputs,
                         "encoding_format": "float"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.load(response)
    data = sorted(body["data"], key=lambda row: row["index"])
    if [row["index"] for row in data] != list(range(len(inputs))):
        raise AssertionError("Missing, duplicate or out-of-range result indices")
    actual = [row["embedding"] for row in data]
    if len(expected) != len(actual):
        raise AssertionError("Embedding count differs from reference")
    for old, new in zip(expected, actual):
        if len(old) != len(new) or not all(math.isfinite(x) for x in new):
            raise AssertionError("Invalid embedding shape or non-finite values")
        if abs(sum(x * x for x in new) - 1) > 1e-4:
            raise AssertionError("Embedding is not L2 normalized")
    max_error = max(abs(a - b) for old, new in zip(expected, actual)
                    for a, b in zip(old, new))
    min_cosine = min(sum(a * b for a, b in zip(old, new)) /
                     math.sqrt(sum(a * a for a in old) * sum(b * b for b in new))
                     for old, new in zip(expected, actual))
    ranks_match = top_three(expected) == top_three(actual)
    report = {
        "status": "PASS" if max_error <= args.max_error and ranks_match
                  and min_cosine >= 1 - 1e-6 else "FAIL",
        "label": args.label,
        "inputs": len(inputs),
        "dimensions": len(actual[0]),
        "reference_image": reference.get("reference_image"),
        "engine": reference.get("engine"),
        "dtype": reference.get("dtype"),
        "pooling": reference.get("pooling"),
        "max_absolute_error": max_error,
        "minimum_cosine_similarity": min_cosine,
        "retrieval_top3_matches": ranks_match,
        "scope": "Sampled dense embeddings under the same model and launch settings",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

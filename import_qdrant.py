#!/usr/bin/env python3
"""Import generated judgment embeddings into Qdrant and verify the count."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


POINT_NAMESPACE = uuid.UUID("b5f41fcf-b5eb-4c52-8647-6406746ddd21")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import QuestaCase vectors into Qdrant")
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://embed-storage:6333"))
    parser.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION", "judgment_chunks"))
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY", ""))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def request_json(
    method: str,
    url: str,
    api_key: str,
    body: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["api-key"] = api_key
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_for_qdrant(url: str, api_key: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            request_json("GET", f"{url}/collections", api_key, timeout=10)
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(2)
    raise TimeoutError(f"Qdrant did not become ready: {last_error}")


def ensure_collection(
    url: str, collection: str, api_key: str, dimension: int, timeout: int
) -> None:
    endpoint = f"{url}/collections/{collection}"
    try:
        payload = request_json("GET", endpoint, api_key, timeout=30)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        request_json(
            "PUT",
            endpoint,
            api_key,
            {
                "vectors": {"size": dimension, "distance": "Cosine"},
                "on_disk_payload": True,
            },
            timeout,
        )
        return

    configured = payload.get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
    configured_size = configured.get("size")
    if configured_size != dimension:
        raise RuntimeError(
            f"Qdrant collection dimension is {configured_size}, expected {dimension}"
        )


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    data = json.loads(args.embeddings.read_text(encoding="utf-8"))
    doc = data.get("document", {})
    doc_id = str(doc.get("document_id", "")).strip()
    dimension = int(data.get("embedding", {}).get("dimension", 0))
    chunks = data.get("chunks", [])
    if not doc_id or dimension < 1 or not chunks:
        raise ValueError("Embedding output is missing document ID, dimension, or chunks")

    base_url = args.url.rstrip("/")
    wait_for_qdrant(base_url, args.api_key, args.timeout)
    ensure_collection(base_url, args.collection, args.api_key, dimension, args.timeout)

    started = time.perf_counter()
    for start in range(0, len(chunks), args.batch_size):
        points = []
        for chunk in chunks[start : start + args.batch_size]:
            chunk_id = str(chunk["chunk_id"])
            points.append(
                {
                    "id": str(uuid.uuid5(POINT_NAMESPACE, chunk_id)),
                    "vector": chunk["embedding"],
                    "payload": {
                        "document_id": doc_id,
                        "chunk_id": chunk_id,
                        "page": chunk["page"],
                        "chunk_index": chunk["chunk_index"],
                        "word_start": chunk["word_start"],
                        "text": chunk["text"],
                        "filename": doc.get("source_file", ""),
                    },
                }
            )
        request_json(
            "PUT",
            f"{base_url}/collections/{args.collection}/points?wait=true",
            args.api_key,
            {"points": points},
            args.timeout,
        )

    count = request_json(
        "POST",
        f"{base_url}/collections/{args.collection}/points/count",
        args.api_key,
        {
            "filter": {
                "must": [
                    {"key": "document_id", "match": {"value": doc_id}}
                ]
            },
            "exact": True,
        },
        args.timeout,
    ).get("result", {}).get("count")
    if count != len(chunks):
        raise RuntimeError(
            f"Qdrant verification mismatch: expected {len(chunks)}, got {count}"
        )
    elapsed = time.perf_counter() - started
    print(
        f"Imported and verified {count} vectors in Qdrant for {doc_id} "
        f"in {elapsed:.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

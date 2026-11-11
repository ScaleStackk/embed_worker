#!/usr/bin/env python3
"""Resolve container inputs and launch the embedding worker."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def find_input_pdf(data_dir: Path) -> Path:
    explicit = os.environ.get("INPUT_PDF")
    if explicit:
        return Path(explicit)
    preferred = data_dir / "input.pdf"
    if preferred.is_file():
        return preferred
    candidates = sorted(data_dir.glob("*.pdf"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("No PDF found in /data; mount one or set INPUT_PDF")
    raise RuntimeError("Multiple PDFs found in /data; set INPUT_PDF explicitly")


def main() -> None:
    mode = os.environ.get("WORKER_MODE", "oneshot").strip().lower()
    if mode == "kafka":
        os.execv(
            sys.executable,
            [sys.executable, "/app/kafka_consumer.py", *sys.argv[1:]],
        )
    if mode != "oneshot":
        raise ValueError("WORKER_MODE must be 'oneshot' or 'kafka'")

    if len(sys.argv) > 1:
        arguments = sys.argv[1:]
    else:
        pdf = find_input_pdf(Path("/data"))
        output = os.environ.get("OUTPUT_JSON", "/data/judgment_embeddings.json")
        arguments = [str(pdf), "--output", output]
        document_id = os.environ.get("DOCUMENT_ID", "").strip()
        if document_id:
            arguments.extend(["--document-id", document_id])

    completed = subprocess.run(
        [sys.executable, "/app/embed_judgment.py", *arguments], check=False
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)

    if os.environ.get("QDRANT_URL"):
        output_path = os.environ.get("OUTPUT_JSON", "/data/judgment_embeddings.json")
        for index, argument in enumerate(arguments[:-1]):
            if argument == "--output":
                output_path = arguments[index + 1]
                break
        imported = subprocess.run(
            [sys.executable, "/app/import_qdrant.py", output_path], check=False
        )
        if imported.returncode:
            raise SystemExit(imported.returncode)


if __name__ == "__main__":
    main()

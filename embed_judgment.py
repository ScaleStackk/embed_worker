#!/usr/bin/env python3
"""Create retrieval-ready, page-grounded embeddings from a judgment PDF."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import fitz
from fastembed import TextEmbedding

from document_identity import resolve_document_identity


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_WORDS = 220
DEFAULT_OVERLAP_WORDS = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract page-aware chunks from a PDF and embed them locally."
    )
    parser.add_argument("pdf", type=Path, help="Judgment PDF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("judgment_embeddings.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EMBED_MODEL", DEFAULT_MODEL),
        help="FastEmbed model name",
    )
    parser.add_argument(
        "--model-cache",
        default=os.environ.get("MODEL_CACHE", "/models"),
        help="Directory containing the bundled model",
    )
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--document-id",
        help=(
            "Canonical ID assigned by the upload API. If omitted, the PDF SHA-256 "
            "is used for backwards-compatible one-file runs."
        ),
    )
    return parser.parse_args()


def clean_page_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_words(text: str, size: int, overlap: int) -> Iterable[tuple[int, str]]:
    words = text.split()
    if not words:
        return
    step = size - overlap
    for start in range(0, len(words), step):
        chunk = words[start : start + size]
        if not chunk:
            break
        yield start, " ".join(chunk)
        if start + size >= len(words):
            break


def normalized(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [round(value / magnitude, 8) for value in vector]


def extract_chunks(
    pdf_path: Path, doc_id: str, chunk_size: int, overlap: int
) -> tuple[list[dict], int]:
    chunks: list[dict] = []
    with fitz.open(pdf_path) as document:
        page_count = document.page_count
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            text = clean_page_text(page.get_text("text"))
            for chunk_index, (word_start, content) in enumerate(
                chunk_words(text, chunk_size, overlap)
            ):
                chunk_id = f"{doc_id[:16]}-p{page_number:04d}-c{chunk_index:03d}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "word_start": word_start,
                        "text": content,
                    }
                )
    return chunks, page_count


def main() -> int:
    args = parse_args()
    if args.chunk_words < 1:
        raise ValueError("--chunk-words must be positive")
    if args.overlap_words < 0 or args.overlap_words >= args.chunk_words:
        raise ValueError("--overlap-words must be >= 0 and smaller than --chunk-words")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not args.pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    started = time.perf_counter()
    doc_id, pdf_sha256 = resolve_document_identity(args.pdf, args.document_id)
    chunks, page_count = extract_chunks(
        args.pdf, doc_id, args.chunk_words, args.overlap_words
    )
    if not chunks:
        raise RuntimeError(
            "No embedded PDF text was found. OCR the document before embedding it."
        )

    print(
        f"Embedding {len(chunks)} chunks from {page_count} pages with {args.model}",
        flush=True,
    )
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    model = TextEmbedding(
        model_name=args.model,
        cache_dir=args.model_cache,
        local_files_only=offline,
    )
    vectors = model.embed(
        [chunk["text"] for chunk in chunks], batch_size=args.batch_size
    )

    dimension = 0
    for chunk, vector in zip(chunks, vectors, strict=True):
        vector_values = vector.tolist()
        dimension = len(vector_values)
        chunk["embedding"] = normalized(vector_values)

    elapsed = time.perf_counter() - started
    output = {
        "schema_version": "1.0",
        "document": {
            "document_id": doc_id,
            "content_sha256": pdf_sha256,
            "source_file": args.pdf.name,
            "page_count": page_count,
        },
        "embedding": {
            "model": args.model,
            "dimension": dimension,
            "normalized": True,
        },
        "chunking": {
            "strategy": "page_bounded_word_window",
            "chunk_words": args.chunk_words,
            "overlap_words": args.overlap_words,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    print(
        f"Wrote {len(chunks)} vectors ({dimension} dimensions) to {args.output} "
        f"in {elapsed:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Embedding failed: {exc}", file=sys.stderr, flush=True)
        raise

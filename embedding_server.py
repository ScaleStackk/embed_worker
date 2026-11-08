#!/usr/bin/env python3
"""Small HTTP service for producing query vectors with the bundled model."""

from __future__ import annotations

import json
import math
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastembed import TextEmbedding


MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MODEL_CACHE = os.environ.get("MODEL_CACHE", "/models")
API_KEY = os.environ.get("EMBEDDING_API_KEY", "").strip()
MAX_TEXT_CHARACTERS = int(os.environ.get("MAX_TEXT_CHARACTERS", "8000"))
PORT = int(os.environ.get("PORT", "8081"))


def normalize(values: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude == 0:
        return values
    return [round(value / magnitude, 8) for value in values]


print(f"Loading embedding model {MODEL_NAME} from {MODEL_CACHE}", flush=True)
MODEL = TextEmbedding(
    model_name=MODEL_NAME,
    cache_dir=MODEL_CACHE,
    local_files_only=True,
)
MODEL_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "QuestCaseEmbedding/1.0"

    def log_message(self, message: str, *args: object) -> None:
        print(f"{self.address_string()} - {message % args}", flush=True)

    def send_json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def authorized(self) -> bool:
        return not API_KEY or self.headers.get("Authorization") == f"Bearer {API_KEY}"

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/health":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self.send_json(
            HTTPStatus.OK,
            {"status": "ok", "model": MODEL_NAME},
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/embed":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if not self.authorized():
            self.send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 64 * 1024:
                raise ValueError("Invalid request size.")
            payload = json.loads(self.rfile.read(length))
            text = str(payload.get("text", "")).strip()
            requested_model = str(payload.get("model", MODEL_NAME)).strip()
            if not text:
                raise ValueError("text is required.")
            if len(text) > MAX_TEXT_CHARACTERS:
                raise ValueError(f"text may not exceed {MAX_TEXT_CHARACTERS} characters.")
            if requested_model != MODEL_NAME:
                raise ValueError(f"Only {MODEL_NAME} is available.")

            with MODEL_LOCK:
                vector = next(MODEL.embed([text])).tolist()
            embedding = normalize(vector)
            self.send_json(
                HTTPStatus.OK,
                {
                    "model": MODEL_NAME,
                    "dimension": len(embedding),
                    "normalized": True,
                    "embedding": embedding,
                },
            )
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_request", "message": str(error)},
            )
        except Exception as error:  # pragma: no cover - defensive service boundary
            print(f"Embedding request failed: {error}", flush=True)
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "embedding_failed"},
            )


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Embedding API listening on 0.0.0.0:{PORT}", flush=True)
    server.serve_forever()

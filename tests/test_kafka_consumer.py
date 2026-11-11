import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from kafka_consumer import UploadJob, parse_upload_event, run_pipeline


class FakeS3Client:
    def download_file(self, bucket: str, key: str, destination: str) -> None:
        self.request = (bucket, key)
        Path(destination).write_bytes(b"%PDF-test")


class UploadEventTests(unittest.TestCase):
    def test_parses_webapi_upload_event(self) -> None:
        event = {
            "schemaVersion": "1.0",
            "eventType": "judgment.uploaded",
            "eventId": "event-1",
            "document": {
                "documentId": "document-1",
                "originalFilename": "judgment.pdf",
                "contentType": "application/pdf",
                "sizeBytes": 42,
            },
            "storage": {
                "bucket": "judgments",
                "key": "judgments/document-1/source.pdf",
                "uri": "s3://judgments/judgments/document-1/source.pdf",
            },
        }

        job = parse_upload_event(json.dumps(event).encode())

        self.assertEqual(job.document_id, "document-1")
        self.assertEqual(job.bucket, "judgments")
        self.assertEqual(job.size_bytes, 42)

    def test_rejects_non_pdf_event(self) -> None:
        event = {
            "eventType": "judgment.uploaded",
            "eventId": "event-1",
            "document": {
                "documentId": "document-1",
                "originalFilename": "notes.txt",
                "contentType": "text/plain",
                "sizeBytes": 42,
            },
            "storage": {"bucket": "judgments", "key": "notes.txt"},
        }
        with self.assertRaisesRegex(ValueError, "non-empty PDF"):
            parse_upload_event(event)

    def test_downloads_embeds_and_imports_vectors(self) -> None:
        job = UploadJob(
            event_id="event-1",
            document_id="document-1",
            original_filename="judgment.pdf",
            size_bytes=len(b"%PDF-test"),
            bucket="judgments",
            key="judgments/document-1/source.pdf",
            uri="s3://judgments/judgments/document-1/source.pdf",
        )
        client = FakeS3Client()

        def fake_run(command: list[str], check: bool) -> None:
            self.assertTrue(check)
            if "/app/embed_judgment.py" in command:
                output = Path(command[command.index("--output") + 1])
                output.write_text(
                    json.dumps(
                        {
                            "chunk_count": 3,
                            "embedding": {"dimension": 384},
                        }
                    ),
                    encoding="utf-8",
                )

        with patch.dict(os.environ, {"QDRANT_URL": "http://qdrant:6333"}), patch(
            "kafka_consumer.subprocess.run", side_effect=fake_run
        ) as run:
            result = run_pipeline(job, client)

        self.assertEqual(client.request, (job.bucket, job.key))
        self.assertEqual(result, {"chunks": 3, "dimension": 384})
        self.assertEqual(run.call_count, 2)
        embedding_command = run.call_args_list[0].args[0]
        self.assertEqual(
            embedding_command[embedding_command.index("--document-id") + 1],
            job.document_id,
        )


if __name__ == "__main__":
    unittest.main()

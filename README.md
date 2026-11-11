# Legal judgment embedding worker

This image extracts text from a judgment PDF, creates overlapping page-bounded
chunks, and embeds those chunks locally with `BAAI/bge-small-en-v1.5`. The model
is bundled into the image, so the container makes no model-server or internet
calls at runtime.

Each output record contains the chunk text, page number, stable document/chunk
IDs, and a normalized 384-dimensional vector. Keeping text and page provenance
next to the vector allows later GraphRAG answers to cite the original judgment.

## Build

On an Apple Silicon Mac:

```bash
chmod +x build-image.sh
./build-image.sh
```

For standard x86-64 EC2 instances:

```bash
PLATFORM=linux/amd64 \
OUTPUT_TAR=legal-embedding-worker-amd64.tar \
./build-image.sh
```

The resulting archive can be copied to a matching host and imported with
`docker load --input <archive>`.

## Run

```bash
docker run --rm \
  --volume "$PWD/data:/data" \
  --env INPUT_PDF=/data/input.pdf \
  --env OUTPUT_JSON=/data/judgment_embeddings.json \
  legal-embedding-worker:latest
```

Or pass paths and chunk settings directly:

```bash
docker run --rm --volume "$PWD/data:/data" legal-embedding-worker:latest \
  /data/judgment.pdf \
  --output /data/judgment-embeddings.json \
  --chunk-words 220 \
  --overlap-words 40
```

Scanned PDFs must be OCRed first.

## Run as the Kafka ingestion worker

Set `WORKER_MODE=kafka` to consume `judgment.uploaded` events continuously. The
consumer downloads each PDF from the event's private S3 location, runs the
existing page-bounded embedding pipeline, imports its vectors into Qdrant, and
commits the Kafka offset only after the import succeeds. It uses the independent
consumer group `questacase-embedding-worker-v1`, so the knowledge-graph group
receives the same event.

On EC2, S3 credentials come from the instance IAM role through boto3. Do not put
long-lived AWS access keys in the image. For local S3-compatible storage, set
`S3_ENDPOINT` and `S3_FORCE_PATH_STYLE=true`.

Start Kafka and Qdrant on the shared `questacase` Docker network, then:

```bash
docker compose -f compose.yaml -f compose.kafka.yaml up --build
```

The consumer publishes `started` and `completed` records to `processing.status`.
After its bounded retries are exhausted, it publishes the error to
`processing.failed` before committing the poison event.

The `document.documentId` UUID from the upload event is passed to
`embed_judgment.py --document-id` and stored as `document.document_id`. The PDF
hash is retained separately as `document.content_sha256`. For manual one-file
runs, omitting `--document-id` preserves the original hash-derived ID behavior.

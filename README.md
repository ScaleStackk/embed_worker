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

Scanned PDFs must be OCRed first. In the eventual AWS pipeline, a Kafka consumer
adapter can invoke this worker logic for an S3 object and send the generated
records to a dedicated vector-index writer.

#!/usr/bin/env bash
set -euo pipefail

image_name="${IMAGE_NAME:-legal-embedding-worker:latest}"
platform="${PLATFORM:-linux/arm64}"
output_tar="${OUTPUT_TAR:-legal-embedding-worker-arm64.tar}"

docker buildx build \
  --platform "${platform}" \
  --load \
  --tag "${image_name}" \
  .

docker image save --output "${output_tar}" "${image_name}"

echo "Created ${output_tar} (${platform})"

#!/usr/bin/env bash
set -euo pipefail

image="$1"
docker build --file Dockerfile.aws --tag "$image" .
docker run --rm --network none --entrypoint sh \
  --volume "$PWD/scripts/verify-image.sh:/checks/verify-image.sh:ro" \
  --volume "$PWD/data:/sample:ro" "$image" \
  -ec 'python -m mlops_demo.aws.container --help && sh /checks/verify-image.sh /sample/deliveries.csv'

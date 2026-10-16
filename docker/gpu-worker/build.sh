#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: build.sh <base-image@sha256:digest> <audioenv-wheel> <output-tag>" >&2
  exit 2
fi
base_image=$1
wheel=$2
tag=$3
if [[ ! $base_image =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "base image must be pinned by an immutable sha256 digest" >&2
  exit 2
fi
if [[ ! -f $wheel ]]; then
  echo "AudioEnv wheel does not exist: $wheel" >&2
  exit 2
fi

docker build \
  --build-arg "BASE_IMAGE=$base_image" \
  --build-arg "AUDIOENV_WHEEL=$wheel" \
  --label "org.opencontainers.image.source=https://github.com/SSDeb-RR/AudioEnv" \
  --tag "$tag" \
  -f docker/gpu-worker/Dockerfile .

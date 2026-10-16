#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: build.sh <base-image@sha256:digest> <audioenv-wheel> <output-tag> [receipt.json]" >&2
  exit 2
fi
base_image=$1
wheel=$2
tag=$3
receipt=${4:-dist/gpu-worker-image-receipt.json}
lock=docker/gpu-worker/dependencies.lock.json
if [[ ! $base_image =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "base image must be pinned by an immutable sha256 digest" >&2
  exit 2
fi
if [[ ! -f $wheel ]]; then
  echo "AudioEnv wheel does not exist: $wheel" >&2
  exit 2
fi
if [[ ! $tag =~ ^[A-Za-z0-9._/:@-]+$ ]]; then
  echo "output tag contains unsupported characters" >&2
  exit 2
fi
if [[ ! -f $lock ]]; then
  echo "dependency lock does not exist: $lock" >&2
  exit 2
fi

IFS=$'\t' read -r audioenv_version verl_omni_commit verl_commit vllm_omni_commit tau2_commit < <(
  python3 - "$lock" <<'PY'
import json, pathlib, sys
lock = json.loads(pathlib.Path(sys.argv[1]).read_text())
keys = ("audioenv_version", "verl_omni_commit", "verl_commit", "vllm_omni_commit", "tau2_commit")
values = [lock.get(key) for key in keys]
if not all(isinstance(value, str) and value for value in values):
    raise SystemExit("dependency lock is missing required release pins")
print("\t".join(values))
PY
)
python3 - "$wheel" "$audioenv_version" <<'PY'
import pathlib, sys, zipfile
wheel, expected = pathlib.Path(sys.argv[1]), sys.argv[2]
with zipfile.ZipFile(wheel) as archive:
    metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    if len(metadata) != 1:
        raise SystemExit("AudioEnv wheel must contain exactly one dist-info/METADATA file")
    version = next(
        (line.removeprefix("Version: ") for line in archive.read(metadata[0]).decode().splitlines()
         if line.startswith("Version: ")), None,
    )
if version != expected:
    raise SystemExit(f"AudioEnv wheel version {version!r} does not match dependency lock {expected!r}")
PY
lock_sha=$(python3 - "$lock" <<'PY'
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)
wheel_sha=$(python3 - "$wheel" <<'PY'
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)
iidfile=$(mktemp)
trap 'rm -f "$iidfile"' EXIT

docker build \
  --build-arg "BASE_IMAGE=$base_image" \
  --build-arg "AUDIOENV_WHEEL=$wheel" \
  --build-arg "VERL_OMNI_COMMIT=$verl_omni_commit" \
  --build-arg "VERL_COMMIT=$verl_commit" \
  --build-arg "VLLM_OMNI_COMMIT=$vllm_omni_commit" \
  --build-arg "TAU2_COMMIT=$tau2_commit" \
  --label "org.opencontainers.image.source=https://github.com/SSDeb-RR/AudioEnv" \
  --label "ai.audioenv.dependency-lock-sha256=$lock_sha" \
  --label "ai.audioenv.wheel-sha256=$wheel_sha" \
  --iidfile "$iidfile" \
  --tag "$tag" \
  -f docker/gpu-worker/Dockerfile .

image_id=$(tr -d '\r\n' < "$iidfile")
if [[ ! $image_id =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Docker did not return an immutable sha256 image ID" >&2
  exit 1
fi
mkdir -p "$(dirname "$receipt")"
python3 - "$receipt" "$tag" "$image_id" "$base_image" "$lock_sha" "$wheel_sha" \
  "$audioenv_version" "$verl_omni_commit" "$verl_commit" "$vllm_omni_commit" "$tau2_commit" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
keys = (
    "image", "image_id", "base_image", "dependency_lock_sha256", "wheel_sha256",
    "audioenv_version", "verl_omni_commit", "verl_commit", "vllm_omni_commit", "tau2_commit",
)
payload = {"schema_version": "audioenv.gpu-worker-image-receipt/v1", **dict(zip(keys, sys.argv[2:], strict=True))}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(path)
PY

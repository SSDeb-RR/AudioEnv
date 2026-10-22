"""Fail the image build on dependency conflicts except one pinned upstream override."""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys


EXPECTED_VLLM_OMNI_VERSION = "0.24.1.dev26+gfe478a95a"
MINIMUM_ACCELERATE_VERSION = (1, 14, 0)


def _version_tuple(value: str) -> tuple[int, ...]:
    release = value.split("+", 1)[0].split(".")
    return tuple(int(part) for part in release[:3])


def main() -> int:
    vllm_omni = importlib.metadata.version("vllm-omni")
    accelerate = importlib.metadata.version("accelerate")
    if vllm_omni != EXPECTED_VLLM_OMNI_VERSION:
        raise SystemExit(
            f"unexpected vllm-omni version {vllm_omni!r}; "
            f"expected {EXPECTED_VLLM_OMNI_VERSION!r}"
        )
    if _version_tuple(accelerate) < MINIMUM_ACCELERATE_VERSION:
        raise SystemExit(f"accelerate {accelerate!r} does not satisfy VeRL-Omni >=1.14.0")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    conflicts = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    allowed = re.compile(
        r"vllm-omni 0\.24\.1\.dev26\+gfe478a95a (?:has requirement|requires) "
        rf"accelerate==1\.12\.0, but you have accelerate {re.escape(accelerate)}\."
    )
    if len(conflicts) != 1 or allowed.fullmatch(conflicts[0]) is None:
        details = "\n".join(conflicts) or result.stderr.strip() or "pip check returned no details"
        raise SystemExit(f"unexpected dependency-check result:\n{details}")

    print(
        "Accepted pinned upstream compatibility override: "
        f"vllm-omni {vllm_omni} metadata requests accelerate 1.12.0; "
        f"VeRL-Omni requires and resolved accelerate {accelerate}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Entrypoint that runs an uploaded Omni driver inside an isolated runtime."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

from ..audio import DeterministicWavFixtureRenderer
from ..spec import AudioObservation, AudioProfile
from .omni_driver import ConformanceFixture, ModelConformanceRunner
from .package import inspect_driver_package


def execute(package_path: Path, expected_model_id: str) -> dict[str, object]:
    """Load untrusted driver code; callers must place this process in a sandbox."""

    payload = package_path.read_bytes()
    inspection = inspect_driver_package(payload, expected_model_id=expected_model_id)
    module_name, class_name = str(inspection.manifest["module"]).split(":", 1)
    # Bind trusted runner code before importing the untrusted package. The OCI
    # boundary protects the host; this binding also prevents ordinary import-
    # time replacement of the runner method from redefining the gate logic.
    run_conformance = ModelConformanceRunner().run
    with tempfile.TemporaryDirectory(prefix="audioenv-driver-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(root)
        sys.path.insert(0, str(root))
        try:
            module = importlib.import_module(module_name)
            driver_type = getattr(module, class_name)
            driver = driver_type()
            audio = DeterministicWavFixtureRenderer().render(
                "Please check order A-1042.", AudioProfile(), 7,
            )
            report = run_conformance(
                driver, ConformanceFixture(AudioObservation(audio=audio)),
            )
        finally:
            sys.path.remove(str(root))
    return report.model_dump(mode="json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an Omni driver conformance suite")
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args(argv)
    diagnostics = io.StringIO()
    try:
        with redirect_stdout(diagnostics), redirect_stderr(diagnostics):
            report = execute(args.package, args.model_id)
        response = {"ok": True, "report": report}
    except Exception as exc:
        response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:1000]}
    sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if response["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

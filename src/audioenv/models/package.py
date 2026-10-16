"""Static intake validation for user-supplied Omni driver source packages."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
from pathlib import PurePosixPath
import re
import stat
import zipfile
from typing import Any


REQUIRED_CONFORMANCE = (
    "dependency_probe", "native_audio_inference", "structured_action",
    "token_logprob_alignment", "multimodal_positions", "mask_isolation",
    "declared_gradients", "lora_update", "checkpoint_reload", "deterministic_probe",
)
_MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
_NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".whl", ".pyc", ".pyd"}


@dataclass(frozen=True, slots=True)
class DriverPackageInspection:
    sha256: str
    manifest: dict[str, Any]
    files: tuple[str, ...]
    uncompressed_bytes: int


def inspect_driver_package(
    payload: bytes, *, expected_model_id: str,
) -> DriverPackageInspection:
    if not payload or len(payload) > 4 * 1024 * 1024:
        raise ValueError("driver package must be a non-empty ZIP no larger than 4 MiB")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > 200:
                raise ValueError("driver package must contain between 1 and 200 files")
            names: list[str] = []
            total = 0
            for info in infos:
                path = PurePosixPath(info.filename)
                if (
                    path.is_absolute() or ".." in path.parts or not path.parts
                    or "\\" in info.filename or info.flag_bits & 0x1
                ):
                    raise ValueError(f"unsafe driver package member: {info.filename}")
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError("driver package cannot contain symbolic links")
                if info.is_dir():
                    continue
                if path.suffix.casefold() in _NATIVE_SUFFIXES:
                    raise ValueError(f"compiled driver artifact is forbidden: {info.filename}")
                total += info.file_size
                if info.file_size > 2 * 1024 * 1024 or total > 16 * 1024 * 1024:
                    raise ValueError("driver package exceeds uncompressed size limits")
                names.append(path.as_posix())
            if len(names) != len(set(names)):
                raise ValueError("driver package contains duplicate file names")
            if "manifest.json" not in names:
                raise ValueError("driver package requires manifest.json")
            try:
                manifest = json.loads(archive.read("manifest.json"))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("driver package manifest is unreadable") from exc
            if not isinstance(manifest, dict) or manifest.get("schema_version") != "audioenv.model-driver-scaffold/v2":
                raise ValueError("unsupported driver package manifest schema")
            if manifest.get("model_id") != expected_model_id:
                raise ValueError("driver package model_id does not match the registered model")
            module = manifest.get("module")
            if not isinstance(module, str) or not _MODULE.fullmatch(module):
                raise ValueError("driver package module must use package.module:ClassName")
            module_name, _class_name = module.split(":", 1)
            module_path = module_name.replace(".", "/") + ".py"
            package_init = module_name.split(".", 1)[0] + "/__init__.py"
            if module_path not in names or package_init not in names:
                raise ValueError("driver package module source or package initializer is missing")
            if tuple(manifest.get("required_conformance") or ()) != REQUIRED_CONFORMANCE:
                raise ValueError("driver package cannot weaken or reorder required conformance gates")
            if manifest.get("registration_state") != "generated_needs_review":
                raise ValueError("uploaded driver packages must remain generated_needs_review")
            if "requirements.lock" in names:
                try:
                    requirements = archive.read("requirements.lock").decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("requirements.lock must be UTF-8 text") from exc
                _validate_requirements_lock(requirements)
    except zipfile.BadZipFile as exc:
        raise ValueError("driver package is not a valid ZIP") from exc
    return DriverPackageInspection(
        sha256=sha256(payload).hexdigest(), manifest=manifest,
        files=tuple(sorted(names)), uncompressed_bytes=total,
    )


def _validate_requirements_lock(value: str) -> None:
    for number, raw in enumerate(value.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") or " @ " in line or "==" not in line or "--hash=sha256:" not in line:
            raise ValueError(
                f"requirements.lock line {number} must be an exact hash-pinned package"
            )

"""Per-rank evidence that a VeRL optimizer step updates only Thinker LoRA."""

from __future__ import annotations

from functools import wraps
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any


ALLOWED_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def install_gradient_audit(engine_class=None) -> bool:
    """Patch the pinned FSDP engine once, inside each VeRL Ray worker."""

    if engine_class is None:
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
        engine_class = FSDPEngine
    if getattr(engine_class, "_audioenv_gradient_audit", False):
        return False
    original = engine_class.optimizer_step

    @wraps(original)
    def optimizer_step(self, *args, **kwargs):
        module = self.module
        gradients = tuple(sorted(
            name for name, parameter in module.named_parameters()
            if parameter.grad is not None and _nonzero(parameter.grad)
        ))
        unexpected = tuple(name for name in gradients if not _allowed(name))
        trainable = tuple(sorted(name for name, parameter in module.named_parameters() if parameter.requires_grad))
        undeclared_trainable = tuple(name for name in trainable if not _allowed(name))
        before = _parameter_hash(module)
        grad_norm = original(self, *args, **kwargs)
        after = _parameter_hash(module)
        evidence = {
            "schema_version": "audioenv.gradient-evidence/v1",
            "rank": _rank(), "gradient_parameters": gradients,
            "trainable_parameters": trainable,
            "unexpected_gradient_parameters": unexpected,
            "undeclared_trainable_parameters": undeclared_trainable,
            "before_hash": before, "after_hash": after,
            "parameters_changed": before != after,
            "grad_norm": float(grad_norm), "finite_grad_norm": math.isfinite(float(grad_norm)),
            "allowed_targets": ALLOWED_LORA_TARGETS,
        }
        _write_evidence(evidence)
        if not gradients:
            raise RuntimeError("AudioEnv gradient audit observed no non-zero gradients")
        if unexpected or undeclared_trainable:
            raise RuntimeError("AudioEnv gradient audit detected parameters outside declared Thinker LoRA targets")
        if not evidence["finite_grad_norm"] or not evidence["parameters_changed"]:
            raise RuntimeError("AudioEnv gradient audit detected a non-finite or ineffective optimizer step")
        return grad_norm

    engine_class.optimizer_step = optimizer_step
    engine_class._audioenv_gradient_audit = True
    return True


def validate_gradient_evidence(directory: Path, *, expected_ranks: int | None = None) -> dict[str, Any]:
    paths = sorted(directory.glob("rank-*-step-*.json"))
    if not paths:
        raise ValueError("no optimizer gradient evidence was produced")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for row in rows:
        if row.get("schema_version") != "audioenv.gradient-evidence/v1":
            raise ValueError("unsupported optimizer gradient evidence")
        if row.get("unexpected_gradient_parameters") or row.get("undeclared_trainable_parameters"):
            raise ValueError("gradient evidence contains parameters outside Thinker LoRA")
        if not row.get("finite_grad_norm") or not row.get("parameters_changed"):
            raise ValueError("gradient evidence does not prove a finite parameter update")
    ranks = tuple(sorted({int(row["rank"]) for row in rows}))
    if expected_ranks is not None and ranks != tuple(range(expected_ranks)):
        raise ValueError(f"gradient evidence must cover ranks 0..{expected_ranks - 1}; found {ranks}")
    return {
        "schema_version": "audioenv.gradient-summary/v1", "evidence_files": len(paths),
        "ranks": ranks, "expected_ranks": expected_ranks,
        "all_lora_only": True, "all_finite": True,
        "all_steps_changed": True,
        "summary_sha256": sha256(b"".join(path.read_bytes() for path in paths)).hexdigest(),
    }


def _allowed(name: str) -> bool:
    return "lora_" in name and any(target in name for target in ALLOWED_LORA_TARGETS) and not any(
        frozen in name for frozen in ("talker", "code2wav", "codec", "audio_tower", "visual")
    )


def _nonzero(tensor: Any) -> bool:
    value = tensor.detach()
    if hasattr(value, "to_local"):
        value = value.to_local()
    return bool(value.count_nonzero().item())


def _parameter_hash(module: Any) -> str:
    import torch

    digest = sha256()
    found = False
    for name, parameter in sorted(module.named_parameters()):
        if not parameter.requires_grad:
            continue
        found = True
        value = parameter.detach()
        if hasattr(value, "to_local"):
            value = value.to_local()
        value = value.cpu().contiguous()
        digest.update(name.encode()); digest.update(b"\0")
        digest.update(str(value.dtype).encode()); digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode()); digest.update(b"\0")
        # NumPy cannot represent torch.bfloat16. Hash the exact underlying
        # bytes through a uint8 view so the release-configured dtype works and
        # no lossy cast can hide a parameter update.
        digest.update(value.view(torch.uint8).numpy().tobytes())
    if not found:
        raise RuntimeError("AudioEnv gradient audit found no trainable parameters")
    return digest.hexdigest()


def _rank() -> int:
    try:
        import torch.distributed as distributed
        if distributed.is_available() and distributed.is_initialized():
            return int(distributed.get_rank())
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _write_evidence(payload: dict[str, Any]) -> None:
    root_value = os.environ.get("AUDIOENV_GRADIENT_EVIDENCE_DIR")
    if not root_value:
        raise RuntimeError("AUDIOENV_GRADIENT_EVIDENCE_DIR is required")
    root = Path(root_value); root.mkdir(parents=True, exist_ok=True)
    rank = int(payload["rank"])
    sequence = len(tuple(root.glob(f"rank-{rank}-step-*.json"))) + 1
    destination = root / f"rank-{rank}-step-{sequence:06d}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)

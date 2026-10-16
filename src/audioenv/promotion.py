"""Backend-issued, tamper-evident environment promotion evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import secrets
from typing import Any

from .contracts import PromotionEvidenceV1, PromotionReportV1


REQUIRED_PROMOTION_GATES = (
    "sandbox_build", "oracle_pass", "noop_fail", "adversarial_fail",
    "determinism", "state_isolation", "leakage", "reward_integrity",
)


class PromotionEvidenceAuthority:
    def __init__(self, key_path: Path, *, worker_id: str = "audioenv-local-validator-v1"):
        self.key_path = key_path.resolve()
        self.worker_id = worker_id
        self._key = self._load_or_create_key()
        self.key_id = sha256(self._key).hexdigest()[:16]

    def issue(
        self, *, environment_hash: str, verifier_hash_before: str,
        verifier_hash_after: str, validation: dict[str, Any],
    ) -> PromotionEvidenceV1:
        checks = aggregate_validation_checks(validation)
        payload = {
            "schema_version": "audioenv.promotion-evidence/v1",
            "environment_hash": environment_hash,
            "verifier_hash_before": verifier_hash_before,
            "verifier_hash_after": verifier_hash_after,
            "checks": checks,
            "worker_id": self.worker_id,
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "nonce": secrets.token_hex(16),
            "key_id": self.key_id,
        }
        payload["signature"] = self._sign(payload)
        return PromotionEvidenceV1.model_validate(payload)

    def verify(self, evidence: PromotionEvidenceV1) -> PromotionReportV1:
        if evidence.key_id != self.key_id:
            raise ValueError("promotion evidence key is not trusted")
        payload = evidence.model_dump(mode="json")
        supplied = payload.pop("signature")
        if not hmac.compare_digest(supplied, self._sign(payload)):
            raise ValueError("promotion evidence signature is invalid")
        return PromotionReportV1.evaluate(
            environment_hash=evidence.environment_hash,
            verifier_hash_before=evidence.verifier_hash_before,
            verifier_hash_after=evidence.verifier_hash_after,
            checks=evidence.checks,
            worker_id=evidence.worker_id,
        )

    def _sign(self, payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hmac.new(self._key, encoded, sha256).hexdigest()

    def _load_or_create_key(self) -> bytes:
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        if self.key_path.exists():
            key = self.key_path.read_bytes()
            if len(key) != 32:
                raise RuntimeError("invalid promotion evidence key")
            return key
        key = secrets.token_bytes(32)
        descriptor = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, key)
        finally:
            os.close(descriptor)
        return key


def aggregate_validation_checks(validation: dict[str, Any]) -> dict[str, bool]:
    rows = validation.get("checks")
    if validation.get("passed") is not True or not isinstance(rows, list):
        return {name: False for name in REQUIRED_PROMOTION_GATES}
    by_name = {
        str(row.get("name")): bool(row.get("passed"))
        for row in rows if isinstance(row, dict) and row.get("name")
    }

    def all_suffix(suffix: str, *, required: bool = True) -> bool:
        values = [passed for name, passed in by_name.items() if name.endswith(suffix)]
        return bool(values or not required) and all(values)

    build_names = [
        name for name in by_name
        if name == "tool_handlers_complete" or name.startswith("audio_profile:")
        or name.endswith(":sandbox_build")
    ]
    sandbox_build = bool(build_names) and all(by_name[name] for name in build_names)
    oracle = all_suffix(":oracle_pass")
    noop = all_suffix(":noop_fail")
    mutations = [passed for name, passed in by_name.items() if name.endswith(":reward_mutation") or name.endswith(":wrong_order")]
    adversarial = bool(mutations) and all(mutations)
    deterministic = all_suffix(":deterministic_replay")
    isolated = all_suffix(":state_isolation")
    leakage = all_suffix(":no_text_leak")
    return {
        "sandbox_build": sandbox_build,
        "oracle_pass": oracle,
        "noop_fail": noop,
        "adversarial_fail": adversarial,
        "determinism": deterministic,
        "state_isolation": isolated,
        "leakage": leakage,
        "reward_integrity": oracle and noop and adversarial and deterministic,
    }

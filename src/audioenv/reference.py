"""Frozen reference V2 corpus for credential-free product acceptance.

The corpus contains 48 deterministic tasks with immutable, disjoint train,
validation, and test memberships.  Oracle trajectories remain authoring-only
validation fixtures and are never consumed by the trainable policy.
"""

from __future__ import annotations

from hashlib import sha256
import json

from .adapter import EnvironmentPackage, ToolResult
from .contracts import SplitManifestV1
from .spec import AudioProfile, EnvironmentSpec, SpeakAction, TaskSpec, ToolCallAction, ToolExpectation, ToolSpec


def _lookup_order(state, args):
    order = state.get("orders", {}).get(str(args["order_id"]))
    return ToolResult(order is not None, {"order": order} if order else {"error": "not_found"})


def _refund_order(state, args):
    order = state.get("orders", {}).get(str(args["order_id"]))
    if not order or order["customer_zip"] != str(args["customer_zip"]):
        return ToolResult(False, {"error": "identity_mismatch"})
    order["status"] = "refunded"
    order["refund_amount"] = order["amount"]
    return ToolResult(True, {"refund_id": order["refund_id"], "amount": order["amount"]})


def _lookup_line(state, args):
    line = state.get("lines", {}).get(str(args["phone_number"]))
    return ToolResult(line is not None, {"line": line} if line else {"error": "not_found"})


def _send_device_step(state, args):
    line = state.get("lines", {}).get(str(args["phone_number"]))
    if not line:
        return ToolResult(False, {"error": "not_found"})
    line["instructions_sent"] = True
    return ToolResult(True, {"instruction": "enable_data_roaming"})


def _confirm_device_state(state, args):
    line = state.get("lines", {}).get(str(args["phone_number"]))
    if not line or not line.get("instructions_sent") or args["enabled"] is not True:
        return ToolResult(False, {"error": "prerequisite_missing"})
    line["data_roaming"] = True
    return ToolResult(True, {"data_roaming": True})


def _change_plan(state, args):
    line = state.get("lines", {}).get(str(args["phone_number"]))
    if not line:
        return ToolResult(False, {"error": "not_found"})
    line["plan"] = str(args["plan"])
    return ToolResult(True, {"plan": line["plan"]})


def _retail_task(index: int, split: str) -> TaskSpec:
    order_id = "A-1042" if index == 1 else f"R-{split[0].upper()}{index:04d}"
    zip_code = "94107" if index == 1 else f"{21000 + index:05d}"
    amount = 79 if index == 1 else 25 + index
    refund_id = "RF-1001" if index == 1 else f"RF-{5000 + index}"
    task_id = "retail-refund-001" if index == 1 else f"retail-refund-{split}-{index:03d}"
    oracle = (
        ToolCallAction("lookup_order", {"order_id": order_id}),
        ToolCallAction("refund_order", {"order_id": order_id, "customer_zip": zip_code}),
        SpeakAction(
            b"agent-audio",
            f"Your refund {refund_id} for ${amount} has been completed.",
            True,
        ),
    )
    return TaskSpec(
        id=task_id,
        split=split,  # type: ignore[arg-type]
        user_goal="Refund a delivered order after identity verification.",
        hidden_facts={"order_id": order_id, "customer_zip": zip_code},
        initial_state={
            "orders": {
                order_id: {
                    "status": "delivered",
                    "amount": amount,
                    "customer_zip": zip_code,
                    "refund_id": refund_id,
                }
            }
        },
        expected_state={"orders": {order_id: {"status": "refunded", "refund_amount": amount}}},
        initial_utterance=f"Please refund order {order_id}. My ZIP code is {zip_code}.",
        required_tools=(
            ToolExpectation("lookup_order", {"order_id": order_id}, "order_verified"),
            ToolExpectation(
                "refund_order",
                {"order_id": order_id, "customer_zip": zip_code},
                "refund_completed",
            ),
        ),
        required_communications=(refund_id, f"${amount}"),
        forbidden_tools=("change_plan",),
        oracle_actions=oracle,
    )


def _telecom_task(index: int, split: str, *, legacy: bool = False) -> TaskSpec:
    phone = "+1-202-555-0198" if legacy else f"+1-303-555-{1000 + index:04d}"
    task_id = "telecom-roaming-001" if legacy else f"telecom-roaming-{split}-{index:03d}"
    oracle = (
        ToolCallAction("lookup_line", {"phone_number": phone}),
        ToolCallAction("send_device_step", {"phone_number": phone}),
        SpeakAction(b"agent-audio", "Please enable data roaming now."),
        ToolCallAction("confirm_device_state", {"phone_number": phone, "enabled": True}),
        SpeakAction(b"agent-audio", "Data roaming is enabled.", True),
    )
    return TaskSpec(
        id=task_id,
        split=split,  # type: ignore[arg-type]
        user_goal="Guide the caller through enabling data roaming without changing their plan.",
        hidden_facts={"phone_number": phone},
        initial_state={
            "lines": {
                phone: {"plan": "standard", "data_roaming": False, "instructions_sent": False}
            }
        },
        expected_state={
            "lines": {
                phone: {"plan": "standard", "data_roaming": True, "instructions_sent": True}
            }
        },
        initial_utterance=f"My phone number is {phone} and mobile data stopped working abroad.",
        caller_turns=("I followed that step and enabled the setting.",),
        required_tools=(
            ToolExpectation("lookup_line", {"phone_number": phone}, "line_verified"),
            ToolExpectation("send_device_step", {"phone_number": phone}, "instruction_delivered"),
            ToolExpectation(
                "confirm_device_state", {"phone_number": phone, "enabled": True}, "device_state_confirmed"
            ),
        ),
        required_communications=("data roaming is enabled",),
        forbidden_tools=("change_plan",),
        oracle_actions=oracle,
    )


def build_reference_package() -> EnvironmentPackage:
    tasks: list[TaskSpec] = []
    offsets = {"train": 0, "validation": 100, "test": 200}
    counts = {"train": 16, "validation": 4, "test": 4}
    for split, count in counts.items():
        for local_index in range(1, count + 1):
            index = offsets[split] + local_index
            tasks.append(_retail_task(index, split))
            tasks.append(_telecom_task(index, split, legacy=(split == "test" and local_index == 1)))
    spec = EnvironmentSpec(
        id="audioenv/reference-agentic",
        version="0.2.0",
        description="Frozen 48-task retail and dual-control telecom acceptance corpus.",
        policies=(
            "Verify identifiers before state-changing actions.",
            "Never change a plan without explicit authorization.",
            "Communicate confirmation identifiers and amounts.",
        ),
        tools=(
            ToolSpec("lookup_order", "Read an order.", ("order_id",)),
            ToolSpec("refund_order", "Refund a verified order.", ("order_id", "customer_zip"), True),
            ToolSpec("lookup_line", "Read a telecom line.", ("phone_number",)),
            ToolSpec("send_device_step", "Give the caller a device action.", ("phone_number",)),
            ToolSpec("confirm_device_state", "Record caller-confirmed device state.", ("phone_number", "enabled"), True),
            ToolSpec("change_plan", "Change the subscription plan.", ("phone_number", "plan"), True),
        ),
        audio_profiles=(
            AudioProfile(),
            AudioProfile(id="degraded-en", noise_snr_db=12.0, codec="pcm_s16le", speaking_rate=1.15),
            AudioProfile(id="fast-en", codec="pcm_s16le", speaking_rate=1.35),
        ),
        max_steps=12,
    )
    handlers = {
        "lookup_order": _lookup_order,
        "refund_order": _refund_order,
        "lookup_line": _lookup_line,
        "send_device_step": _send_device_step,
        "confirm_device_state": _confirm_device_state,
        "change_plan": _change_plan,
    }
    candidates = {}
    for task in tasks:
        if "orders" in task.initial_state:
            identifier = next(iter(task.initial_state["orders"]))
            progress = ToolCallAction("lookup_order", {"order_id": identifier})
            forbidden = ToolCallAction("change_plan", {"phone_number": "unknown", "plan": "premium"})
        else:
            identifier = next(iter(task.initial_state["lines"]))
            progress = ToolCallAction("lookup_line", {"phone_number": identifier})
            forbidden = ToolCallAction("change_plan", {"phone_number": identifier, "plan": "premium"})
        candidates[task.id] = (
            SpeakAction(b"agent-audio", "I cannot complete that request.", True),
            progress,
            forbidden,
        )
    return EnvironmentPackage(spec, tuple(tasks), handlers, candidates)


def build_reference_split_manifest(package: EnvironmentPackage | None = None) -> SplitManifestV1:
    """Build the immutable task/seed manifest bound to the reference package."""
    package = package or build_reference_package()
    from .evaluation import package_hashes

    environment_hash, _ = package_hashes(package)
    groups = {
        split: tuple(sorted(task.id for task in package.tasks if task.split == split))
        for split in ("train", "validation", "test")
    }
    payload = {
        "schema_version": "audioenv.split/v1",
        "environment_hash": environment_hash,
        **groups,
        "seeds": {
            "train": tuple(range(1_000, 1_000 + len(groups["train"]))),
            "validation": tuple(range(20_000, 20_000 + len(groups["validation"]))),
            "test": tuple(range(30_000, 30_000 + len(groups["test"]))),
        },
    }
    content_hash = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SplitManifestV1(**payload, content_hash=content_hash)


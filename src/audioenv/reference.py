"""Two fictional reference domains: retail and dual-control telecom."""

from __future__ import annotations

from .adapter import EnvironmentPackage, ToolResult
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
    return ToolResult(True, {"refund_id": "RF-1001", "amount": order["amount"]})


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


def build_reference_package() -> EnvironmentPackage:
    retail_oracle = (
        ToolCallAction("lookup_order", {"order_id": "A-1042"}),
        ToolCallAction("refund_order", {"order_id": "A-1042", "customer_zip": "94107"}),
        SpeakAction(b"agent-audio", "Your refund RF-1001 for $79 has been completed.", True),
    )
    telecom_oracle = (
        ToolCallAction("lookup_line", {"phone_number": "+1-202-555-0198"}),
        ToolCallAction("send_device_step", {"phone_number": "+1-202-555-0198"}),
        SpeakAction(b"agent-audio", "Please enable data roaming now."),
        ToolCallAction("confirm_device_state", {"phone_number": "+1-202-555-0198", "enabled": True}),
        SpeakAction(b"agent-audio", "Data roaming is enabled.", True),
    )
    tasks = (
        TaskSpec(
            id="retail-refund-001",
            split="train",
            user_goal="Refund a delivered order after identity verification.",
            hidden_facts={"order_id": "A-1042", "customer_zip": "94107"},
            initial_state={
                "orders": {"A-1042": {"status": "delivered", "amount": 79, "customer_zip": "94107"}}
            },
            expected_state={"orders": {"A-1042": {"status": "refunded", "refund_amount": 79}}},
            initial_utterance="Please refund order A-1042. My ZIP code is 94107.",
            required_tools=(
                ToolExpectation("lookup_order", {"order_id": "A-1042"}, "order_verified"),
                ToolExpectation(
                    "refund_order",
                    {"order_id": "A-1042", "customer_zip": "94107"},
                    "refund_completed",
                ),
            ),
            required_communications=("RF-1001", "$79"),
            forbidden_tools=("change_plan",),
            oracle_actions=retail_oracle,
        ),
        TaskSpec(
            id="telecom-roaming-001",
            split="test",
            user_goal="Guide the caller through enabling data roaming without changing their plan.",
            hidden_facts={"phone_number": "+1-202-555-0198"},
            initial_state={
                "lines": {
                    "+1-202-555-0198": {
                        "plan": "standard",
                        "data_roaming": False,
                        "instructions_sent": False,
                    }
                }
            },
            expected_state={
                "lines": {
                    "+1-202-555-0198": {
                        "plan": "standard",
                        "data_roaming": True,
                        "instructions_sent": True,
                    }
                }
            },
            initial_utterance="My phone number is +1-202-555-0198 and mobile data stopped working abroad.",
            caller_turns=("I followed that step and enabled the setting.",),
            required_tools=(
                ToolExpectation("lookup_line", {"phone_number": "+1-202-555-0198"}, "line_verified"),
                ToolExpectation(
                    "send_device_step", {"phone_number": "+1-202-555-0198"}, "instruction_delivered"
                ),
                ToolExpectation(
                    "confirm_device_state",
                    {"phone_number": "+1-202-555-0198", "enabled": True},
                    "device_state_confirmed",
                ),
            ),
            required_communications=("data roaming is enabled",),
            forbidden_tools=("change_plan",),
            oracle_actions=telecom_oracle,
        ),
    )
    spec = EnvironmentSpec(
        id="audioenv/reference-agentic",
        version="0.1.0",
        description="Fictional retail and dual-control telecom tasks.",
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
    return EnvironmentPackage(spec, tasks, handlers)


"""Versioned JSON-lines worker executed inside the pinned τ² Python 3.12 env."""

from __future__ import annotations

import json
import sys
from typing import Any
from uuid import uuid4


PROTOCOL_VERSION = "audioenv.tau2-worker/v1"


class RetailWorker:
    def __init__(self):
        from tau2.domains.retail.environment import get_environment, get_tasks, get_tasks_split

        self.get_environment = get_environment
        self.tasks = {task.id: task for task in get_tasks(None)}
        self.splits = get_tasks_split()
        self.sessions: dict[str, dict[str, Any]] = {}

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("operation")
        if operation == "probe":
            return {"protocol": PROTOCOL_VERSION, "domain": "retail", "task_count": len(self.tasks)}
        if operation == "discover":
            # tau2 also publishes an aggregate ``base`` list containing every
            # task. It is not a training split and must not overwrite the
            # official train/test membership.
            split_by_id = {
                str(task_id): name
                for name in ("train", "test")
                for task_id in self.splits.get(name, ())
            }
            return {"tasks": [{"id": task.id, "split": split_by_id.get(task.id, "unknown")} for task in self.tasks.values()]}
        if operation == "reset":
            task = self.tasks[str(request["task_id"])]
            environment = self.get_environment()
            initial = task.initial_state
            environment.set_state(
                initialization_data=initial.initialization_data if initial else None,
                initialization_actions=initial.initialization_actions if initial else None,
                message_history=list(initial.message_history or []) if initial else [],
            )
            session_id = str(uuid4())
            self.sessions[session_id] = {
                "task": task, "environment": environment, "calls": [],
                "messages": list(initial.message_history or []) if initial else [],
                "terminated": False,
            }
            return {
                "session_id": session_id,
                "task_id": task.id,
                "tool_names": [tool.name for tool in environment.get_tools()],
                "initial_db_hash": environment.get_db_hash(),
            }
        if operation == "caller_context":
            # This endpoint is intentionally separate from reset/tool results:
            # only the frozen caller adapter may consume the private scenario.
            session = self.sessions[str(request["session_id"])]
            return {"user_scenario": _jsonable(session["task"].user_scenario)}
        if operation == "speak":
            from tau2.data_model.message import AssistantMessage, UserMessage

            session = self.sessions[str(request["session_id"])]
            role = str(request["role"])
            text = str(request.get("semantic_text") or "").strip()
            if role not in {"assistant", "user"} or not text:
                raise ValueError("speak requires an assistant/user role and semantic text")
            message = AssistantMessage.text(text) if role == "assistant" else UserMessage.text(text)
            session["messages"].append(message)
            if bool(request.get("end_episode")):
                session["terminated"] = True
            return {"recorded": True, "message_count": len(session["messages"])}
        if operation == "tool":
            from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage

            session = self.sessions[str(request["session_id"])]
            name = str(request["name"])
            arguments = dict(request.get("arguments") or {})
            call_id = str(uuid4())
            session["messages"].append(AssistantMessage.text(
                "", tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments, requestor="assistant")],
            ))
            result = session["environment"].make_tool_call(name, requestor="assistant", **arguments)
            json_result = _jsonable(result)
            session["messages"].append(ToolMessage(
                id=call_id, role="tool", content=json.dumps(json_result, sort_keys=True), requestor="assistant",
            ))
            session["calls"].append({"id": call_id, "name": name, "arguments": arguments})
            return {"result": json_result, "db_hash": session["environment"].get_db_hash()}
        if operation == "verify":
            session = self.sessions[str(request["session_id"])]
            task = session["task"]
            predicted = session["environment"]
            gold = self.get_environment()
            initial = task.initial_state
            gold.set_state(
                initialization_data=initial.initialization_data if initial else None,
                initialization_actions=initial.initialization_actions if initial else None,
                message_history=list(initial.message_history or []) if initial else [],
            )
            criteria = task.evaluation_criteria
            for action in criteria.actions or [] if criteria else []:
                try:
                    gold.make_tool_call(action.name, requestor=action.requestor, **action.arguments)
                except Exception:
                    pass
            assertions = criteria.env_assertions or [] if criteria else []
            assertion_results = [predicted.run_env_assertion(item, raise_assertion_error=False) for item in assertions]
            db_match = predicted.get_db_hash() == gold.get_db_hash()
            user_db_match = predicted.get_user_db_hash() == gold.get_user_db_hash()
            result = {
                "upstream_environment_reward": float(db_match and user_db_match and all(assertion_results)),
                "db_match": db_match,
                "user_db_match": user_db_match,
                "env_assertions": assertion_results,
                "calls": session["calls"],
            }
            if session["terminated"]:
                result.update(self._official_reward(session))
            else:
                result.update({"upstream_reward": 0.0, "reward_info": {
                    "info": {"note": "episode has not terminated"},
                }})
            return result
        raise ValueError(f"unsupported operation: {operation}")

    @staticmethod
    def _official_reward(session: dict[str, Any]) -> dict[str, Any]:
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        simulation = SimulationRun(
            id=str(uuid4()), task_id=session["task"].id,
            start_time="1970-01-01T00:00:00+00:00", end_time="1970-01-01T00:00:00+00:00",
            duration=0.0, termination_reason=TerminationReason.AGENT_STOP,
            messages=session["messages"], seed=0,
        )
        reward = evaluate_simulation(
            simulation=simulation, task=session["task"], evaluation_type=EvaluationType.ALL,
            solo_mode=False, domain="retail", strict_replay=True,
        )
        return {"upstream_reward": float(reward.reward), "reward_info": reward.model_dump(mode="json")}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def main() -> int:
    worker = RetailWorker()
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            if request.get("protocol") != PROTOCOL_VERSION:
                raise ValueError("unsupported protocol version")
            response = {"protocol": PROTOCOL_VERSION, "request_id": request_id, "ok": True, "result": worker.handle(request)}
        except Exception as exc:
            response = {"protocol": PROTOCOL_VERSION, "request_id": request_id, "ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(response, sort_keys=True) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
            self.sessions[session_id] = {"task": task, "environment": environment, "calls": []}
            return {
                "session_id": session_id,
                "task_id": task.id,
                "tool_names": [tool.name for tool in environment.get_tools()],
                "initial_db_hash": environment.get_db_hash(),
            }
        if operation == "tool":
            session = self.sessions[str(request["session_id"])]
            name = str(request["name"])
            arguments = dict(request.get("arguments") or {})
            result = session["environment"].make_tool_call(name, requestor="assistant", **arguments)
            session["calls"].append({"name": name, "arguments": arguments})
            return {"result": _jsonable(result), "db_hash": session["environment"].get_db_hash()}
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
            return {
                "upstream_environment_reward": float(db_match and user_db_match and all(assertion_results)),
                "db_match": db_match,
                "user_db_match": user_db_match,
                "env_assertions": assertion_results,
                "calls": session["calls"],
            }
        raise ValueError(f"unsupported operation: {operation}")


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

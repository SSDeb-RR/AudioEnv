"""Controller for the isolated, pinned tau2 JSON-lines worker."""

from __future__ import annotations

from collections import deque
import json
import os
import queue
import signal
import subprocess
import threading
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .tau2_worker import PROTOCOL_VERSION


class Tau2WorkerError(RuntimeError):
    """Raised when the isolated worker rejects or corrupts a request."""


class Tau2WorkerTimeout(TimeoutError):
    """Raised when the isolated worker does not answer within the deadline."""


class Tau2WorkerClient:
    """Own one stateful tau2 worker process and serialize its requests.

    The command is executed directly (never through a shell). Only an explicit
    environment mapping is added to the inherited environment, so callers can
    pass non-secret runtime settings without serializing secret values into an
    AudioEnv experiment record.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float = 30.0,
        environment: Mapping[str, str] | None = None,
        stderr_lines: int = 100,
    ):
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("worker command must contain non-empty string arguments")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.command = tuple(command)
        self.timeout_seconds = float(timeout_seconds)
        self.environment = dict(environment or {})
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=stderr_lines)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        return tuple(self._stderr)

    def start(self) -> "Tau2WorkerClient":
        if self._process is not None and self._process.poll() is None:
            return self
        process_environment = {**os.environ, **self.environment}
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=process_environment,
            start_new_session=True,
        )
        threading.Thread(target=self._read_stdout, name="tau2-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="tau2-stderr", daemon=True).start()
        return self

    def request(self, operation: str, **payload: Any) -> dict[str, Any]:
        if not operation:
            raise ValueError("operation is required")
        with self._lock:
            self.start()
            process = self._require_process()
            request_id = str(uuid4())
            request = {
                "protocol": PROTOCOL_VERSION,
                "request_id": request_id,
                "operation": operation,
                **payload,
            }
            assert process.stdin is not None
            try:
                process.stdin.write(json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise Tau2WorkerError(self._terminated_message(process)) from exc
            try:
                line = self._responses.get(timeout=self.timeout_seconds)
            except queue.Empty as exc:
                self.close()
                raise Tau2WorkerTimeout(
                    f"tau2 worker timed out after {self.timeout_seconds:g}s during {operation}"
                ) from exc
            if line is None:
                raise Tau2WorkerError(self._terminated_message(process))
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                self.close()
                raise Tau2WorkerError("tau2 worker emitted invalid JSON") from exc
            if response.get("protocol") != PROTOCOL_VERSION:
                self.close()
                raise Tau2WorkerError("tau2 worker protocol mismatch")
            if response.get("request_id") != request_id:
                self.close()
                raise Tau2WorkerError("tau2 worker response correlation mismatch")
            if response.get("ok") is not True:
                raise Tau2WorkerError(str(response.get("error") or "tau2 worker rejected request"))
            result = response.get("result")
            if not isinstance(result, dict):
                raise Tau2WorkerError("tau2 worker result must be an object")
            return result

    def probe(self) -> dict[str, Any]:
        return self.request("probe")

    def discover(self) -> dict[str, Any]:
        return self.request("discover")

    def reset(self, task_id: str) -> dict[str, Any]:
        return self.request("reset", task_id=task_id)

    def caller_context(self, session_id: str) -> dict[str, Any]:
        """Return private user instructions for the frozen caller only."""
        return self.request("caller_context", session_id=session_id)

    def speak(
        self, session_id: str, role: str, semantic_text: str, *, end_episode: bool = False,
    ) -> dict[str, Any]:
        if role not in {"assistant", "user"}:
            raise ValueError("tau2 speech role must be assistant or user")
        return self.request(
            "speak", session_id=session_id, role=role,
            semantic_text=semantic_text, end_episode=end_episode,
        )

    def tool(self, session_id: str, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self.request("tool", session_id=session_id, name=name, arguments=dict(arguments))

    def verify(self, session_id: str) -> dict[str, Any]:
        return self.request("verify", session_id=session_id)

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    def __enter__(self) -> "Tau2WorkerClient":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read_stdout(self) -> None:
        process = self._require_process()
        assert process.stdout is not None
        for line in process.stdout:
            self._responses.put(line)
        self._responses.put(None)

    def _read_stderr(self) -> None:
        process = self._require_process()
        assert process.stderr is not None
        for line in process.stderr:
            self._stderr.append(line.rstrip())

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise RuntimeError("tau2 worker is not running")
        return self._process

    def _terminated_message(self, process: subprocess.Popen[str]) -> str:
        status = process.poll()
        detail = " | ".join(self.stderr_tail[-3:])
        suffix = f"; stderr: {detail}" if detail else ""
        return f"tau2 worker terminated with status {status}{suffix}"

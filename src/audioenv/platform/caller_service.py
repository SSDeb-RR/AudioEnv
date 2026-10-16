"""Loopback HTTP service owning one frozen Qwen caller checkpoint."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from ..models.qwen3_omni import Qwen3OmniCallerAdapter


@dataclass(slots=True)
class _Session:
    history: list[dict[str, Any]]
    turn_index: int
    seed: int


class FrozenCallerSessionManager:
    """Serializes model inference while keeping per-episode state isolated."""

    def __init__(self, adapter: Qwen3OmniCallerAdapter):
        if adapter.capabilities.trainable:
            raise ValueError("caller service requires a frozen adapter")
        self.adapter = adapter
        self.sessions: dict[str, _Session] = {}
        self.lock = Lock()

    def create(self, private_goal: str, seed: int) -> dict[str, Any]:
        if not private_goal.strip():
            raise ValueError("private caller goal is required")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("caller seed must be a non-negative integer")
        with self.lock:
            _seed_frozen_inference(seed, 0)
            turn, history = self.adapter.start_isolated_session(private_goal, seed)
            session_id = str(uuid4())
            self.sessions[session_id] = _Session(history=history, turn_index=turn.turn_index, seed=seed)
        return {"session_id": session_id, **_turn_payload(turn)}

    def respond(self, session_id: str, agent_audio: bytes) -> dict[str, Any]:
        if not agent_audio:
            raise ValueError("agent speech audio is required")
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError("caller session not found")
            _seed_frozen_inference(session.seed, session.turn_index + 1)
            turn, history = self.adapter.continue_isolated_session(session.history, agent_audio)
            self.sessions[session_id] = _Session(
                history=history, turn_index=turn.turn_index, seed=session.seed,
            )
        return _turn_payload(turn)

    def delete(self, session_id: str) -> bool:
        with self.lock:
            return self.sessions.pop(session_id, None) is not None


class FrozenCallerHTTPClient:
    """Strict loopback client used from VeRL agent-loop workers."""

    def __init__(self, base_url: str = "http://127.0.0.1:8788", *, timeout_seconds: float = 900):
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("frozen caller client accepts loopback HTTP only")
        if timeout_seconds <= 0:
            raise ValueError("caller timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def create(self, private_goal: str, seed: int) -> dict[str, Any]:
        return self._request("POST", "/v1/sessions", {"private_goal": private_goal, "seed": seed})

    def respond(self, session_id: str, agent_audio: bytes) -> dict[str, Any]:
        return self._request("POST", f"/v1/sessions/{session_id}/respond", {
            "agent_audio_base64": base64.b64encode(agent_audio).decode(),
        })

    def delete(self, session_id: str) -> None:
        self._request("DELETE", f"/v1/sessions/{session_id}", None)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        request = Request(
            self.base_url + path, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - loopback enforced above
            result = json.loads(response.read())
        if not isinstance(result, dict):
            raise RuntimeError("caller service response must be an object")
        return result


def create_server(manager: FrozenCallerSessionManager, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ready": True, "model_id": manager.adapter.model_id, "frozen": True})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                payload = self._payload()
                if self.path == "/v1/sessions":
                    result = manager.create(
                        str(payload.get("private_goal") or ""), payload.get("seed", 0),
                    )
                elif self.path.startswith("/v1/sessions/") and self.path.endswith("/respond"):
                    session_id = self.path.removeprefix("/v1/sessions/").removesuffix("/respond")
                    result = manager.respond(session_id, base64.b64decode(payload.get("agent_audio_base64") or "", validate=True))
                else:
                    self._json(404, {"error": "not found"}); return
                self._json(200, result)
            except KeyError as exc:
                self._json(404, {"error": str(exc)})
            except Exception as exc:
                self._json(422, {"error": str(exc)})

        def do_DELETE(self):
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/v1/sessions/"):
                self._json(404, {"error": "not found"}); return
            session_id = parsed.path.removeprefix("/v1/sessions/")
            self._json(200 if manager.delete(session_id) else 404, {"deleted": session_id})

        def _payload(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000_000:
                raise ValueError("invalid request size")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("request body must be an object")
            return value

        def _json(self, status: int, payload: dict[str, Any]):
            body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002 - stdlib API
            return

    return ThreadingHTTPServer((host, port), Handler)


def _turn_payload(turn) -> dict[str, Any]:
    return {
        "audio_base64": base64.b64encode(turn.audio).decode(),
        "semantic_text": turn.semantic_text,
        "source": turn.source,
        "turn_index": turn.turn_index,
    }


def _seed_frozen_inference(seed: int, turn_index: int) -> int:
    """Set a session/turn-specific RNG state while inference is serialized."""

    import torch

    material = f"audioenv-frozen-caller:{seed}:{turn_index}".encode()
    derived = int.from_bytes(sha256(material).digest()[:8], "big") % (2**63)
    torch.manual_seed(derived)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(derived)
    return derived


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audioenv-qwen-caller")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("caller service binds to loopback only")
    adapter = Qwen3OmniCallerAdapter(args.checkpoint, allow_download=False)
    adapter.load()
    create_server(FrozenCallerSessionManager(adapter), args.host, args.port).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

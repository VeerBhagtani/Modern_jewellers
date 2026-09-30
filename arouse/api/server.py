"""Local HTTP API for Arouse (stdlib only) + a minimal chat UI at `/`.

    GET  /               chat UI
    GET  /v1/health      status + model info
    POST /v1/chat        {"messages":[{"role","content"}], "max_tokens", "temperature", "top_k", "top_p", "seed", "stream"}
    POST /v1/generate    {"prompt", "allow_special", ...same sampling fields..., "stream"}
    POST /v1/agent       next structured action for a client-managed conversation (MDA)
    POST /v1/agent/run   local demo: server session + sandbox tools, runs to a terminal action
    POST /v1/agent/reset forget a demo session

Streaming responses are Server-Sent Events: `data: {"delta": "..."}` ... `data: {"done": true, ...}`.
One model, one generation at a time (a lock serialises requests).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from arouse.api.agent_service import AgentService
from arouse.api.errors import ApiError
from arouse.config import ConfigError
from arouse.inference import InferenceEngine, Message, PromptTooLong, SamplingParams, StreamEvent

API_VERSION = "v1"
MAX_BODY_BYTES = 1_000_000
STATIC_DIR = Path(__file__).parent / "static"
_PARAM_KEYS = {"max_tokens": "max_new_tokens", "temperature": "temperature", "top_k": "top_k", "top_p": "top_p", "seed": "seed"}
_ROUTE_KEYS = {
    "chat": {"messages", "stream", "model", *_PARAM_KEYS},
    "generate": {"prompt", "allow_special", "stream", "model", *_PARAM_KEYS},
}  # "model" is accepted and ignored (one model per server)


class ArouseService:
    """Transport-independent request handling (the HTTP handler is a thin shell)."""

    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine
        self.lock = threading.Lock()
        self.agent = AgentService(engine, self.lock)

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "api_version": API_VERSION, "model": self.engine.info()}

    @staticmethod
    def sampling_params(body: dict[str, Any]) -> SamplingParams:
        raw = {_PARAM_KEYS[k]: v for k, v in body.items() if k in _PARAM_KEYS}
        try:
            return SamplingParams.from_dict(raw)
        except ConfigError as e:
            raise ApiError(400, str(e)) from e

    @staticmethod
    def messages(body: dict[str, Any]) -> list[Message]:
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ApiError(400, "'messages' must be a non-empty list")
        out = []
        for i, m in enumerate(msgs):
            if not isinstance(m, dict):
                raise ApiError(400, f"messages[{i}] must be an object")
            try:
                out.append(Message(m.get("role"), m.get("content")))
            except ValueError as e:
                raise ApiError(400, f"messages[{i}]: {e}") from e
        return out

    def prompt_ids(self, route: str, body: dict[str, Any], params: SamplingParams) -> list[int]:
        if route == "chat":
            try:
                return self.engine.chat_prompt(self.messages(body), params)
            except PromptTooLong as e:
                raise ApiError(400, str(e)) from e
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ApiError(400, "'prompt' must be a non-empty string")
        allow = body.get("allow_special", False)
        if not isinstance(allow, bool):
            raise ApiError(400, "'allow_special' must be a boolean")
        ids = self.engine.tokenizer.encode(prompt, allow_special=allow, add_bos=True)
        if len(ids) >= self.engine.context_length:
            raise ApiError(400, f"prompt is {len(ids)} tokens; context_length is {self.engine.context_length}")
        return ids

    def run(self, prompt_ids: list[int], params: SamplingParams) -> Iterator[StreamEvent]:
        with self.lock:
            yield from self.engine.stream(prompt_ids, params)


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


class Handler(BaseHTTPRequestHandler):
    server_version = "Arouse/0.1"
    server: ArouseHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.server.verbose:
            super().log_message(fmt, *args)

    # --- responses -------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: dict[str, Any]) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": {"status": status, "message": message}})

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as e:
            raise ApiError(400, "bad Content-Length") from e
        if length > MAX_BODY_BYTES:
            raise ApiError(413, f"body larger than {MAX_BODY_BYTES} bytes")
        try:
            body = json.loads(self.rfile.read(length) or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ApiError(400, f"invalid JSON: {e}") from e
        if not isinstance(body, dict):
            raise ApiError(400, "body must be a JSON object")
        return body

    # --- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/v1/health":
            self._json(200, self.server.service.health())
        else:
            self._error(404, f"no route GET {path}")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        routes = {"/v1/chat": "chat", "/v1/generate": "generate"}
        agent_routes = {"/v1/agent": "next_action", "/v1/agent/run": "run", "/v1/agent/reset": "reset"}
        try:
            self._post(path, routes, agent_routes)
        except ApiError as e:
            self._error(e.status, e.message)
        except Exception as e:  # never drop the connection without an answer
            self.log_error("internal error: %r", e)
            self._error(500, "internal error")

    def _post(self, path: str, routes: dict[str, str], agent_routes: dict[str, str]) -> None:
        if path in agent_routes:
            body = self._read_json()
            self._json(200, getattr(self.server.service.agent, agent_routes[path])(body))
            return
        if path not in routes:
            raise ApiError(404, f"no route POST {path}")
        body = self._read_json()
        unknown = sorted(set(body) - _ROUTE_KEYS[routes[path]])
        if unknown:
            raise ApiError(400, f"unknown fields: {unknown}")
        svc = self.server.service
        params = svc.sampling_params(body)
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise ApiError(400, "'stream' must be a boolean")
        prompt = svc.prompt_ids(routes[path], body, params)
        events = svc.run(prompt, params)
        if stream:
            self._stream(routes[path], events, len(prompt))
        else:
            self._complete(routes[path], events, len(prompt))

    def _complete(self, route: str, events: Iterator[StreamEvent], n_prompt: int) -> None:
        parts, n, reason = [], 0, "stop"
        for ev in events:
            parts.append(ev.text)
            n += ev.token_id is not None and ev.finish_reason != "stop"
            reason = ev.finish_reason or reason
        text = "".join(parts)
        out: dict[str, Any] = {
            "id": f"arouse-{uuid.uuid4().hex[:12]}",
            "created": int(time.time()),
            "model": self.server.service.engine.info()["name"],
            "finish_reason": reason,
            "usage": _usage(n_prompt, n),
        }
        if route == "chat":
            out["message"] = {"role": "assistant", "content": text}
        else:
            out["text"] = text
        self._json(200, out)

    def _stream(self, route: str, events: Iterator[StreamEvent], n_prompt: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        n = 0
        try:
            for ev in events:
                n += ev.token_id is not None and ev.finish_reason != "stop"
                if ev.text:
                    self._sse({"delta": ev.text})
                if ev.finish_reason:
                    self._sse({"done": True, "finish_reason": ev.finish_reason, "usage": _usage(n_prompt, n)})
        except (BrokenPipeError, ConnectionResetError):
            pass  # client pressed Stop / went away: stop generating
        finally:
            events.close()  # type: ignore[attr-defined]  # releases the model lock

    def _sse(self, obj: dict[str, Any]) -> None:
        self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
        self.wfile.flush()


class ArouseHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: ArouseService, verbose: bool = False) -> None:
        super().__init__(address, Handler)
        self.service = service
        self.verbose = verbose


def make_server(engine: InferenceEngine, host: str = "127.0.0.1", port: int = 8000, verbose: bool = False) -> ArouseHTTPServer:
    return ArouseHTTPServer((host, port), ArouseService(engine), verbose)

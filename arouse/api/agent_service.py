"""Agent endpoints.

POST /v1/agent        stateless next-action (for MDA): the client owns the conversation and
                      executes tools itself, then sends the observation back as an event.
POST /v1/agent/run    local demo: server-side session + built-in sandbox tools; runs until
                      Arouse finishes, asks, or fails.
POST /v1/agent/reset  forget a demo session.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Any

from arouse.agent.context import build_context
from arouse.agent.episode import Header, validate_events
from arouse.agent.runtime import AgentRuntime
from arouse.agent.tools import REGISTRY, Sandbox
from arouse.api.errors import ApiError
from arouse.inference import InferenceEngine
from arouse.protocol import PROTOCOL_VERSION, ProtocolError

DEMO_FILES = {
    "sales.csv": "date,product,quantity,price\n2026-09-01,milk_1l,120,56\n2026-09-02,curd_500g,40,35\n"
                 "2026-09-03,paneer_200g,25,90\n2026-09-04,ghee_500ml,10,410",
    "milk_records.csv": "date,cow,litres\n2026-09-01,cow_12,14\n2026-09-01,cow_17,11\n2026-09-02,cow_12,15",
    "notes.txt": "Vet visit on Tuesday at 10.\nOrder mineral blocks this week.",
    "staff.csv": "name,role,phone\nRavi,parlour,98xxxxxx01\nMeena,records,98xxxxxx02",
}
MAX_SESSIONS = 100


class Session:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.sandbox = Sandbox(datetime.now().replace(second=0, microsecond=0), DEMO_FILES)


class AgentService:
    def __init__(self, engine: InferenceEngine, lock: threading.Lock) -> None:
        self.engine = engine
        self.lock = lock
        self.runtime = AgentRuntime(engine, REGISTRY)
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        self._sessions_lock = threading.Lock()

    @staticmethod
    def _header(context: dict[str, Any] | None, memory: str | None, state: dict[str, Any] | None,
                tools: list[str] | None) -> Header:
        if tools is not None:
            unknown = [t for t in tools if t not in REGISTRY.specs]
            if unknown:
                raise ApiError(400, f"unknown tools {unknown}; available: {REGISTRY.names()}")
        now = datetime.now().replace(second=0, microsecond=0)
        return Header(context=context or build_context(now, "local"), tools=tools or REGISTRY.names(),
                      memory=memory, state=state)

    # --- stateless (MDA) ------------------------------------------------------

    def next_action(self, body: dict[str, Any]) -> dict[str, Any]:
        allowed = {"events", "context", "memory", "state", "tools"}
        extra = sorted(set(body) - allowed)
        if extra:
            raise ApiError(400, f"unknown fields: {extra}")
        events = body.get("events")
        if not isinstance(events, list) or not events:
            raise ApiError(400, "'events' must be a non-empty list")
        for key, typ in (("context", dict), ("memory", str), ("state", dict), ("tools", list)):
            if body.get(key) is not None and not isinstance(body[key], typ):
                raise ApiError(400, f"'{key}' has the wrong type")
        try:
            validate_events(events)
        except (ValueError, ProtocolError) as e:
            raise ApiError(400, str(e)) from e
        header = self._header(body.get("context"), body.get("memory"), body.get("state"), body.get("tools"))
        try:
            with self.lock:
                r = self.runtime.next_turn(header, events)
        except ProtocolError as e:
            raise ApiError(400, str(e)) from e
        return {"protocol": PROTOCOL_VERSION, **r.turn.to_dict(), "valid": r.valid, "attempts": r.attempts,
                "guarded": r.guarded}

    # --- demo sessions ----------------------------------------------------------

    def _session(self, sid: str | None) -> tuple[str, Session]:
        with self._sessions_lock:
            if sid and sid in self.sessions:
                self.sessions.move_to_end(sid)
                return sid, self.sessions[sid]
            sid = sid or uuid.uuid4().hex[:16]
            self.sessions[sid] = Session()
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)
            return sid, self.sessions[sid]

    def run(self, body: dict[str, Any]) -> dict[str, Any]:
        extra = sorted(set(body) - {"session_id", "message"})
        if extra:
            raise ApiError(400, f"unknown fields: {extra}")
        msg = body.get("message")
        if not isinstance(msg, str) or not msg.strip():
            raise ApiError(400, "'message' must be a non-empty string")
        if len(msg) > 2000:
            raise ApiError(400, "'message' is too long (max 2000 characters)")
        sid = body.get("session_id")
        if sid is not None and (not isinstance(sid, str) or not sid.isalnum() or len(sid) > 64):
            raise ApiError(400, "'session_id' must be alphanumeric")
        sid, sess = self._session(sid)
        now = datetime.now().replace(second=0, microsecond=0)
        sess.sandbox.now = now
        header = Header(context=build_context(now, "local"), tools=REGISTRY.names())
        sess.events.append({"type": "user", "content": msg.strip()})
        try:
            with self.lock:
                res = self.runtime.run(header, sess.events, sess.sandbox.execute)
        except ProtocolError:  # conversation no longer fits: start over, keep tools state
            sess.events = [sess.events[-1]]
            try:
                with self.lock:
                    res = self.runtime.run(header, sess.events, sess.sandbox.execute)
            except ProtocolError as e:
                sess.events = []
                raise ApiError(400, f"message is too long for this model: {e}") from e
        sess.events.extend(res.events)
        steps = []
        for ev in res.events:
            steps.append(ev["turn"] if ev["type"] == "arouse" else {"observation": ev["type"], "content": ev["content"]})
        return {
            "protocol": PROTOCOL_VERSION,
            "session_id": sid,
            "final": res.final.to_dict(),
            "steps": steps,
            "valid": all(t.valid for t in res.turns),
            "guarded": any(t.guarded for t in res.turns),
            "state": sess.sandbox.snapshot(),
        }

    def reset(self, body: dict[str, Any]) -> dict[str, Any]:
        sid = body.get("session_id")
        with self._sessions_lock:
            existed = self.sessions.pop(sid, None) is not None if isinstance(sid, str) else False
        return {"reset": existed}

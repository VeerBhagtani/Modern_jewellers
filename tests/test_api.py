import http.client
import json
import threading
import urllib.request

import pytest

torch = pytest.importorskip("torch")

from arouse.api.server import MAX_BODY_BYTES, make_server  # noqa: E402
from arouse.inference import InferenceEngine  # noqa: E402
from arouse.model.config import get_preset  # noqa: E402
from arouse.model.transformer import ArouseTransformer  # noqa: E402

CHAT = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 6, "seed": 1}


@pytest.fixture(scope="module")
def server(tiny_tokenizer):
    torch.manual_seed(0)
    engine = InferenceEngine(ArouseTransformer(get_preset("arouse-tiny")), tiny_tokenizer)
    srv = make_server(engine, port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}", srv
    srv.shutdown()
    srv.server_close()


def call(base, path, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method="POST" if data is not None else "GET")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.headers.get("Content-Type"), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read().decode()


def sse_events(text):
    return [json.loads(line[6:]) for line in text.split("\n\n") if line.startswith("data: ")]


def test_index_page(server):
    status, ctype, body = call(server[0], "/")
    assert status == 200 and ctype.startswith("text/html")
    assert "<title>Arouse Chat</title>" in body and "/v1/chat" in body


def test_health(server):
    status, _, body = call(server[0], "/v1/health")
    h = json.loads(body)
    assert status == 200 and h["status"] == "ok" and h["api_version"] == "v1"
    assert h["model"]["name"] == "arouse-tiny" and h["model"]["trained"] is False


def test_chat_complete(server):
    status, ctype, body = call(server[0], "/v1/chat", CHAT)
    r = json.loads(body)
    assert status == 200 and ctype.startswith("application/json")
    assert r["message"]["role"] == "assistant" and isinstance(r["message"]["content"], str)
    assert r["finish_reason"] in ("stop", "length")
    assert r["usage"]["prompt_tokens"] == 6 and 0 <= r["usage"]["completion_tokens"] <= 6


def test_chat_stream_matches_complete(server):
    _, _, full = call(server[0], "/v1/chat", CHAT)
    status, ctype, body = call(server[0], "/v1/chat", {**CHAT, "stream": True})
    events = sse_events(body)
    assert status == 200 and ctype.startswith("text/event-stream")
    assert events[-1]["done"] is True
    streamed = "".join(e.get("delta", "") for e in events)
    assert streamed == json.loads(full)["message"]["content"]  # same seed -> same text


def test_generate(server):
    status, _, body = call(server[0], "/v1/generate", {"prompt": "Every Monday", "max_tokens": 4, "temperature": 0})
    r = json.loads(body)
    assert status == 200 and isinstance(r["text"], str) and r["usage"]["prompt_tokens"] > 1


@pytest.mark.parametrize(
    "path,body,raw,code,msg",
    [
        ("/v1/chat", None, b"{not json", 400, "invalid JSON"),
        ("/v1/chat", [1, 2], None, 400, "JSON object"),
        ("/v1/chat", {"messages": []}, None, 400, "non-empty list"),
        ("/v1/chat", {"messages": [{"role": "tool", "content": "x"}]}, None, 400, "role"),
        ("/v1/chat", {**CHAT, "temperature": -1}, None, 400, "temperature"),
        ("/v1/chat", {**CHAT, "temprature": 1}, None, 400, "unknown fields"),
        ("/v1/chat", {**CHAT, "stream": "yes"}, None, 400, "stream"),
        ("/v1/chat", {"messages": [{"role": "user", "content": "word " * 400}]}, None, 400, "tokens"),
        ("/v1/generate", {"prompt": ""}, None, 400, "prompt"),
        ("/v1/generate", {"prompt": "x", "allow_special": "no"}, None, 400, "allow_special"),
        ("/v1/agent", {}, None, 501, "Milestone 6"),
        ("/v1/nope", {}, None, 404, "no route"),
    ],
)
def test_bad_requests(server, path, body, raw, code, msg):
    status, _, text = call(server[0], path, body, raw)
    assert status == code and msg in json.loads(text)["error"]["message"]


def test_unknown_get_route(server):
    assert call(server[0], "/v1/missing")[0] == 404


def test_body_too_large(server):
    """Rejected from the Content-Length header alone, before reading the body."""
    host, port = server[0].removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=10)
    conn.putrequest("POST", "/v1/chat")
    conn.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
    conn.endheaders()
    resp = conn.getresponse()
    assert resp.status == 413 and "larger than" in json.loads(resp.read())["error"]["message"]
    conn.close()


def test_concurrent_requests_are_serialised(server):
    results = []

    def worker():
        results.append(call(server[0], "/v1/chat", CHAT)[0])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert results == [200] * 4


def test_client_disconnect_releases_model(server):
    base, srv = server
    host, port = base.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=30)
    body = json.dumps({**CHAT, "max_tokens": 200, "stream": True})
    conn.request("POST", "/v1/chat", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read(20)  # read a little, then hang up mid-stream
    conn.close()
    status, _, _ = call(base, "/v1/chat", CHAT)  # would deadlock if the lock leaked
    assert status == 200

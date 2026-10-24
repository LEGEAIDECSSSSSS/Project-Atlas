"""Test harness: a scripted fake Ollama server plus an isolated data directory."""
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Must be set before the app (and config) is imported.
_tmp = tempfile.mkdtemp(prefix="lc-test-")
_sock = socket.socket()
_sock.bind(("127.0.0.1", 0))
_port = _sock.getsockname()[1]
_sock.close()

os.environ["LC_DATA_DIR"] = _tmp
os.environ["LC_OLLAMA_URL"] = f"http://127.0.0.1:{_port}"
os.environ["LC_CODE_MODEL"] = "qwen2.5-coder:7b"
os.environ["LC_CHAT_MODEL"] = "qwen2.5:7b"


class FakeOllama:
    """Scripted replies. Each call to /api/chat pops the next reply from `script`."""

    def __init__(self):
        self.installed = ["qwen2.5-coder:7b", "qwen2.5:7b"]
        self.script = []
        self.requests = []

    def reset(self):
        self.installed = ["qwen2.5-coder:7b", "qwen2.5:7b"]
        self.script = []
        self.requests = []


fake = FakeOllama()
fake_app = FastAPI()


@fake_app.get("/api/tags")
async def tags():
    return {"models": [{"name": n} for n in fake.installed]}


@fake_app.get("/api/ps")
async def ps():
    return {"models": []}


@fake_app.post("/api/chat")
async def chat(request: Request):
    payload = await request.json()
    fake.requests.append(payload)

    if payload["model"] not in fake.installed:
        return JSONResponse({"error": f"model '{payload['model']}' not found"}, status_code=404)

    reply = fake.script.pop(0) if fake.script else "OK"
    if callable(reply):
        reply = reply(payload)

    async def stream():
        for i in range(0, len(reply), 7):
            yield json.dumps({"message": {"role": "assistant", "content": reply[i:i + 7]}, "done": False}) + "\n"
        yield json.dumps({"message": {"content": ""}, "done": True, "eval_count": 40, "eval_duration": 2_000_000_000}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@pytest.fixture(scope="session", autouse=True)
def fake_server():
    server = uvicorn.Server(uvicorn.Config(fake_app, host="127.0.0.1", port=_port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield
    server.should_exit = True


@pytest.fixture(autouse=True)
def reset_fake():
    fake.reset()
    from services import ollama
    ollama._cache.update(at=0.0, models=[])
    yield


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def script():
    return fake


def sse(client, body):
    """POST /chat/stream and return the list of decoded events."""
    events = []
    with client.stream("POST", "/chat/stream", json=body) as response:
        assert response.status_code == 200, response.read()
        for line in response.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return events

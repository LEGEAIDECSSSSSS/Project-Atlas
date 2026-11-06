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
os.environ["LC_WARM_INDEX"] = "0"


class FakeOllama:
    """Scripted replies. Each call to /api/chat pops the next reply from `script`."""

    def __init__(self):
        self.installed = ["qwen2.5-coder:7b", "qwen2.5:7b"]
        self.script = []
        self.requests = []
        self.prompt_eval_count = None

    def reset(self):
        self.installed = ["qwen2.5-coder:7b", "qwen2.5:7b"]
        self.script = []
        self.requests = []
        self.prompt_eval_count = None


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
        yield json.dumps({"message": {"content": ""}, "done": True, "eval_count": 40, "eval_duration": 2_000_000_000,
                          **({"prompt_eval_count": fake.prompt_eval_count} if fake.prompt_eval_count else {})}) + "\n"

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


@pytest.fixture
def project():
    """Create real project folders in the test workspace; removed again afterwards."""
    from tests.test_agent_gate import World
    made = []

    def make(files):
        world = World(files)
        made.append(world)
        return world

    yield make
    for world in made:
        world.close()


SAMPLE_PROJECT = {
    "README.md": "# Shop\n\nA small web shop with a FastAPI backend and a React storefront.\n",
    "package.json": '{"name": "shop", "scripts": {"start": "vite", "test": "vitest run"}, "dependencies": {"react": "18"}}',
    "requirements.txt": "fastapi\nuvicorn\npytest\n",
    "backend/main.py": (
        "from fastapi import FastAPI\n"
        "from backend.api import auth\n\n"
        "app = FastAPI()\n"
        "app.include_router(auth.router)\n\n\n"
        "def main():\n"
        "    import uvicorn\n"
        "    uvicorn.run(app)\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    main()\n"
    ),
    "backend/api/__init__.py": "",
    "backend/api/auth.py": (
        "from fastapi import APIRouter\n"
        "from backend.services.auth_service import authenticate_user\n\n"
        "router = APIRouter()\n\n\n"
        "@router.post(\"/login\")\n"
        "def login(body):\n"
        "    \"\"\"Log a user in.\"\"\"\n"
        "    user = authenticate_user(body.email, body.password)\n"
        "    return {\"token\": user.token}\n\n\n"
        "@router.get(\"/health\")\n"
        "def health():\n"
        "    return {\"ok\": True}\n"
    ),
    "backend/services/__init__.py": "",
    "backend/services/auth_service.py": (
        "from backend.services.user_repo import find_user, verify_password\n\n\n"
        "def authenticate_user(email, password):\n"
        "    # look the user up, then check the password\n"
        "    user = find_user(email)\n"
        "    if not verify_password(user, password):\n"
        "        raise PermissionError(\"bad credentials\")\n"
        "    return user\n"
    ),
    "backend/services/user_repo.py": (
        "USERS = {}\n\n\n"
        "def find_user(email):\n"
        "    return USERS[email]\n\n\n"
        "def verify_password(user, password):\n"
        "    return user.password == password\n"
    ),
    "backend/tests/test_auth.py": (
        "from backend.services.auth_service import authenticate_user\n\n\n"
        "def test_login_ok():\n"
        "    assert authenticate_user('a@b.c', 'pw')\n"
    ),
    "frontend/src/api.js": (
        "const API = 'http://localhost:8000';\n"
        "export async function login(email, password) {\n"
        "  const r = await fetch(`${API}/login`, { method: 'POST', body: JSON.stringify({ email, password }) });\n"
        "  return r.json();\n"
        "}\n"
    ),
    "frontend/src/App.jsx": (
        "import React from 'react';\n"
        "import { login } from './api';\n\n"
        "export function LoginForm() {\n"
        "  const submit = async () => { await login('a', 'b'); };\n"
        "  return <button onClick={submit}>Sign in</button>;\n"
        "}\n"
    ),
    "src/main/java/com/shop/UserController.java": (
        "package com.shop;\n\n"
        "import org.springframework.web.bind.annotation.*;\n\n"
        "@RestController\n"
        "@RequestMapping(\"/api\")\n"
        "public class UserController {\n"
        "    private final UserService service;\n\n"
        "    @GetMapping(\"/users/{id}\")\n"
        "    public User getUser(@PathVariable Long id) {\n"
        "        return service.findById(id);\n"
        "    }\n"
        "}\n"
    ),
}

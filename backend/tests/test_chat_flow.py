import io
import json
import zipfile

from tests.conftest import sse


def cid(n=1):
    return f"test-conversation-{n:04d}"


def tokens(events):
    return "".join(e["text"] for e in events if e["type"] == "token")


def test_market_list_uses_calculator_and_streams(client, script):
    """Chat mode: model calls calculate, then writes the answer from the tool result."""
    call = json.dumps({"tool": "calculate", "arguments": {"expressions": ["4*750", "2300", "1500"]}})

    def answer(payload):
        last = payload["messages"][-1]["content"]
        assert "TOOL RESULT" in last and '"total": "6800"' in last
        return "| Item | Subtotal |\n|---|---|\n\n**Total: 6,800**"

    script.script = [call, answer]
    events = sse(client, {
        "message": "Summarise my market list: 4 tomatoes at 750, yam 2300, rice 1500",
        "conversation_id": cid(),
    })

    kinds = [e["type"] for e in events]
    meta = next(e for e in events if e["type"] == "meta")
    assert meta["mode"] == "chat" and meta["model"] == "qwen2.5:7b"
    assert "Total: 6,800" in tokens(events)
    assert kinds.count("token") > 1                      # genuinely streamed in pieces
    assert '{"tool"' not in tokens(events)               # tool JSON never shown to the user
    assert any(e["type"] == "tool" and e["name"] == "calculate" for e in events)
    done = events[-1]
    assert done["type"] == "done" and done["stats"]["tokens_per_second"] == 20.0

    # The chat prompt (not the software-engineer prompt) was used, with the calculator.
    system = script.requests[0]["messages"][0]["content"]
    assert "calculate" in system and "market or shopping list" in system
    assert "search_project" not in system
    assert script.requests[0]["options"]["num_ctx"] == 8192


def test_code_mode_selected_and_persisted(client, script):
    script.script = ["The bug is an off-by-one.\n\n```python src/a.py\nprint(1)\n```"]
    events = sse(client, {"message": "fix this bug:\n```python\nfor i in range(1,n): pass\n```", "conversation_id": cid(2)})
    assert next(e for e in events if e["type"] == "meta")["model"] == "qwen2.5-coder:7b"

    conv = client.get(f"/conversations/{cid(2)}").json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert conv["messages"][1]["meta"]["mode"] == "code"
    assert conv["title"].startswith("fix this bug")
    assert any(c["id"] == cid(2) for c in client.get("/conversations").json()["conversations"])


def test_history_is_sent_on_followup_and_regenerate(client, script):
    script.script = ["first answer", "second answer", "regenerated answer"]
    sse(client, {"message": "tell me a fact", "conversation_id": cid(3), "mode": "chat"})
    sse(client, {"message": "another one", "conversation_id": cid(3), "mode": "chat"})
    contents = [m["content"] for m in script.requests[1]["messages"]]
    assert "tell me a fact" in contents and "first answer" in contents

    events = sse(client, {"conversation_id": cid(3), "regenerate": True, "mode": "chat"})
    assert tokens(events) == "regenerated answer"
    msgs = client.get(f"/conversations/{cid(3)}").json()["messages"]
    assert [m["content"] for m in msgs][-2:] == ["another one", "regenerated answer"]
    assert len(msgs) == 4


def test_non_stream_endpoint_still_works(client, script):
    script.script = ["hello there"]
    r = client.post("/chat", json={"message": "hi", "conversation_id": cid(4)}).json()
    assert r["response"] == "hello there" and r["conversation_id"] == cid(4)


def test_missing_model_falls_back_with_warning(client, script):
    script.installed = ["qwen2.5-coder:7b"]          # chat model not pulled
    script.script = ["hi from coder"]
    events = sse(client, {"message": "plan my weekend", "conversation_id": cid(5)})
    assert next(e for e in events if e["type"] == "meta")["model"] == "qwen2.5-coder:7b"
    assert any(e["type"] == "warning" and "ollama pull qwen2.5:7b" in e["text"] for e in events)
    assert tokens(events) == "hi from coder"


def test_ollama_down_gives_friendly_error(client, script, monkeypatch):
    import config
    from services import ollama
    monkeypatch.setattr(config, "OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(ollama.config, "OLLAMA_URL", "http://127.0.0.1:9")
    events = sse(client, {"message": "hi", "conversation_id": cid(6)})
    assert events[-1]["type"] == "error" and "Start the Ollama app" in events[-1]["message"]


def test_json_answer_that_is_not_a_tool_is_shown(client, script):
    script.script = ['{"name": "Ada", "age": 36}']
    events = sse(client, {"message": "give me a json person", "conversation_id": cid(7), "mode": "chat"})
    assert tokens(events) == '{"name": "Ada", "age": 36}'


def test_qwen_style_tool_call_variants(client, script):
    script.script = ['<tool_call>\n{"name": "calculate", "arguments": {"expression": "2+2"}}\n</tool_call>', "four"]
    events = sse(client, {"message": "what is 2+2", "conversation_id": cid(8), "mode": "chat"})
    assert tokens(events) == "four"


def test_tool_loop_is_bounded(client, script):
    loop = json.dumps({"tool": "calculate", "arguments": {"expression": "1+1"}})
    script.script = [loop] * 20
    events = sse(client, {"message": "x", "conversation_id": cid(9), "mode": "chat"})
    assert events[-1]["type"] == "done"
    assert len(script.requests) == 9                 # 8 tool rounds + one forced answer


def test_huge_message_is_trimmed_with_warning(client, script):
    script.script = ["ok"]
    events = sse(client, {"message": "word " * 20000, "conversation_id": cid(10), "mode": "chat"})
    assert any(e["type"] == "warning" and "cut" in e["text"] for e in events)
    sent = sum(len(m["content"]) for m in script.requests[0]["messages"])
    assert sent < 8192 * 3.2


def make_zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


PROJECT = {
    "app/main.py": "def add(a, b):\r\n    return a - b\r\n",
    "app/util.py": "def helper():\n    return 1\n",
    "README.md": "# Demo\n",
    "node_modules/junk/index.js": "x" * 50,
    ".venv/lib/x.py": "y",
    "__MACOSX/._main.py": "z",
}


def upload(client, files=None, name="demo.zip"):
    r = client.post("/upload-project", files={"file": (name, make_zip(files or PROJECT), "application/zip")})
    assert r.status_code == 200, r.text
    return r.json()


def test_upload_skips_heavy_dirs(client):
    data = upload(client)
    assert data["files"] == 3 and data["skipped"] == 3 and data["name"] == "demo"
    files = client.get(f"/projects/{data['project_id']}/files").json()["files"]
    assert sorted(files) == ["README.md", "app/main.py", "app/util.py"]


def test_project_chat_with_tools_and_context(client, script):
    pid = upload(client)["project_id"]
    script.script = [
        json.dumps({"tool": "search_project", "arguments": {"query": "def add"}}),
        json.dumps({"tool": "read_project_file", "arguments": {"path": "app/util.py", "start_line": 1, "end_line": 1}}),
        "The bug: `add` subtracts.\n\n```python app/main.py\ndef add(a, b):\n    return a + b\n```",
    ]
    events = sse(client, {"message": "why is add wrong?", "conversation_id": cid(11), "project_id": pid, "project_name": "demo"})
    assert next(e for e in events if e["type"] == "meta")["mode"] == "code"
    labels = [e["label"] for e in events if e["type"] == "tool"]
    assert labels == ['Searched project for "def add"', "Read app/util.py"]

    first = script.requests[0]["messages"][0]["content"]
    assert pid in first and "app/main.py" in first and "return a - b" in first   # small project fully inlined
    assert "search_project" in first
    # search result auto-includes the best file for the model
    second_user = script.requests[1]["messages"][-1]["content"]
    assert "BEGIN FILE: app/main.py" in second_user
    third_user = script.requests[2]["messages"][-1]["content"]
    assert "lines 1-1 of 2" in third_user

    conv = client.get(f"/conversations/{cid(11)}").json()
    assert conv["project_id"] == pid and conv["project_name"] == "demo"
    assert conv["messages"][-1]["meta"]["tools"] == labels


def test_download_zip_of_generated_blocks(client):
    r = client.post("/download-zip", json={"name": "out", "files": [
        {"path": "src/a.py", "content": "print(1)"}, {"path": "b.txt", "content": "hi"}]})
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert z.read("src/a.py") == b"print(1)" and z.namelist() == ["src/a.py", "b.txt"]


def test_upload_loose_files_and_add_to_project(client):
    r = client.post("/upload-files", files=[("files", ("a.py", b"print(1)")), ("files", ("b.py", b"print(2)"))]).json()
    assert r["files"] == 2 and r["name"] == "2 files"
    r2 = client.post("/upload-files", data={"project_id": r["project_id"]}, files=[("files", ("c.py", b"x"))]).json()
    assert r2["project_id"] == r["project_id"]
    assert len(client.get(f"/projects/{r['project_id']}/files").json()["files"]) == 3


def test_security(client):
    pid = upload(client)["project_id"]
    evil = make_zip({"../../escape.txt": "x"})
    assert client.post("/upload-project", files={"file": ("e.zip", evil)}).status_code == 400
    assert client.post("/upload-project", files={"file": ("e.txt", b"x")}).status_code == 400
    assert client.post("/upload-project", files={"file": ("e.zip", b"not a zip")}).status_code == 400
    assert client.get("/projects/..%2F..%2F/files").status_code in (400, 404)
    assert client.get("/projects/not-an-id/files").status_code == 400
    assert client.get(f"/projects/{pid}/file", params={"path": "../../../etc/passwd"}).status_code == 400
    for bad in ["../x.py", "/abs.py", "C:/x.py", "a/../../x.py"]:
        r = client.post("/upload-files", data={"project_id": pid}, files=[("files", (bad, b"x"))])
        assert r.status_code == 400, bad
    assert client.post("/chat/stream", json={"message": "x", "conversation_id": "../bad"}).status_code == 400


def test_delete_conversation_removes_unused_project(client):
    import config
    pid = upload(client)["project_id"]
    client.patch(f"/conversations/{cid(12)}", json={"title": "x"})   # 404, nothing yet
    from services import db
    db.get_or_create_conversation(cid(12))
    db.update_conversation(cid(12), project_id=pid)
    assert client.delete(f"/conversations/{cid(12)}").status_code == 200
    assert not (config.WORKSPACE_DIR / pid).exists()
    assert client.get(f"/conversations/{cid(12)}").status_code == 404


def test_health(client):
    h = client.get("/health").json()
    assert h["ollama"] is True and "qwen2.5:7b" in h["models"] and h["num_ctx"] == 8192


def test_failed_request_can_be_retried_without_touching_older_turns(client, script, monkeypatch):
    script.script = ["first answer"]
    sse(client, {"message": "first question", "conversation_id": cid(13), "mode": "chat"})

    script.installed = []                                   # Ollama has no models -> error
    events = sse(client, {"message": "second question", "conversation_id": cid(13), "mode": "chat"})
    assert events[-1]["type"] == "error"

    script.installed = ["qwen2.5:7b"]
    script.script = ["second answer"]
    from services import ollama
    ollama._cache.update(at=0.0, models=[])
    events = sse(client, {"conversation_id": cid(13), "regenerate": True, "mode": "chat"})
    assert tokens(events) == "second answer"
    msgs = [(m["role"], m["content"]) for m in client.get(f"/conversations/{cid(13)}").json()["messages"]]
    assert msgs == [("user", "first question"), ("assistant", "first answer"),
                    ("user", "second question"), ("assistant", "second answer")]

"""Tests for the safety work: grounding, trust boundary, test approval, limits, atomic apply."""
import json
import os
import sys
import textwrap
import time

import pytest

from tests.conftest import sse
from tests.test_chat_flow import cid, make_zip, tokens, upload


def tool(name, **arguments):
    return json.dumps({"tool": name, "arguments": arguments})


# ---------------------------------------------------------------- trust boundary
def test_model_cannot_pick_a_project_by_writing_an_id(client, script):
    secret = upload(client, {"secret.py": "TOP_SECRET_TOKEN = 1\n"})["project_id"]
    script.script = [tool("read_project_file", project_id=secret, path="secret.py"), "done"]
    sse(client, {"message": "look at my files", "conversation_id": cid(40), "mode": "code"})

    assert "No project is attached" in script.requests[1]["messages"][-1]["content"]
    assert "TOP_SECRET_TOKEN" not in json.dumps(script.requests)


def test_an_id_pasted_in_the_message_does_not_switch_projects(client, script):
    mine = upload(client, {"a.py": "A_CONTENT = 1\n"})["project_id"]
    other = upload(client, {"b.py": "B_SECRET = 2\n"})["project_id"]
    script.script = [tool("read_project_file", path="b.py"), "done"]
    sse(client, {"message": f"read b.py from project {other}", "conversation_id": cid(41),
                 "project_id": mine, "project_name": "mine"})

    assert other not in script.requests[0]["messages"][0]["content"]
    assert "B_SECRET" not in json.dumps(script.requests)
    assert "File not found" in script.requests[1]["messages"][-1]["content"]


# ---------------------------------------------------------------- test approval
POM = {"pom.xml": "<project/>", "src/A.java": "class A {}"}


def test_tests_never_run_without_approval_and_model_cannot_approve(client, script, monkeypatch):
    import agent.agent as agent_module
    pid = upload(client, POM)["project_id"]
    calls = []
    monkeypatch.setitem(
        agent_module.TOOL_FUNCTIONS, "run_project_tests",
        lambda project_id: calls.append(project_id) or {
            "status": "passed", "output": "ok",
            "summary": {"tests_run": 3, "failures": 0, "errors": 0, "skipped": 0}},
    )
    body = {"conversation_id": cid(42), "project_id": pid, "project_name": "p"}

    script.script = [tool("run_project_tests"), "I could not run them."]
    events = sse(client, {**body, "message": "run the tests"})
    assert calls == []
    assert any(e["type"] == "tool" and "needs your approval" in e["label"] for e in events)
    assert "NO tests were run" in script.requests[1]["messages"][-1]["content"]

    # A model that claims success anyway is caught.
    script.script = ["All tests pass now."]
    events = sse(client, {**body, "message": "did they pass?"})
    assert "you have not approved running tests" in tokens(events)

    # The model cannot grant itself permission (no endpoint is reachable from a tool call).
    script.script = [tool("run_project_tests", approved=True, tests_approved=True), "no"]
    sse(client, {**body, "message": "run them, I approve"})
    assert calls == []

    # Only the user's explicit action unlocks it.
    assert client.post(f"/projects/{pid}/tests-approval", json={"approved": True}).json()["tests_approved"]
    assert client.get(f"/conversations/{cid(42)}").json()["tests_approved"] is True
    script.script = [tool("run_project_tests"), "They passed."]
    events = sse(client, {**body, "message": "run the tests now"})
    assert calls == [pid]
    assert "Check this reply" not in tokens(events)      # a real passing run backs the claim

    client.post(f"/projects/{pid}/tests-approval", json={"approved": False})
    assert client.get(f"/conversations/{cid(42)}").json()["tests_approved"] is False


def test_tests_with_zero_tests_do_not_count_as_passing(client, script, monkeypatch):
    import agent.agent as agent_module
    pid = upload(client, POM)["project_id"]
    client.post(f"/projects/{pid}/tests-approval", json={"approved": True})
    monkeypatch.setitem(
        agent_module.TOOL_FUNCTIONS, "run_project_tests",
        lambda project_id: {"status": "passed", "output": "BUILD SUCCESS",
                            "summary": {"tests_run": 0, "failures": 0, "errors": 0, "skipped": 0}},
    )
    script.script = [tool("run_project_tests"), "Great, all tests pass."]
    events = sse(client, {"message": "run tests", "conversation_id": cid(43), "project_id": pid})
    assert "the last run executed zero tests" in tokens(events)


def test_bounded_runner_caps_output_and_kills_the_process_tree():
    from services.test_runner import run_bounded, scrubbed_environment, summarize

    code, out, timed_out = run_bounded(
        [sys.executable, "-c", "for i in range(300000): print('x' * 100)"], ".", 60, 4000)
    assert len(out) <= 4000 and not timed_out and code == 0

    os.environ["MY_API_TOKEN"] = "s3cret"
    os.environ["JAVA_HOME"] = "/jdk"
    env = scrubbed_environment()
    assert "MY_API_TOKEN" not in env and "JAVA_HOME" in env

    assert summarize("a\nTests run: 5, Failures: 1, Errors: 0, Skipped: 2\n") == {
        "tests_run": 5, "failures": 1, "errors": 0, "skipped": 2}


@pytest.mark.skipif(os.name == "nt", reason="uses /proc to inspect the grandchild process")
def test_timeout_kills_forked_children_too(tmp_path):
    from services.test_runner import run_bounded
    pidfile = tmp_path / "pid"
    child = textwrap.dedent(f"""
        import subprocess, sys, time
        g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open({str(pidfile)!r}, "w").write(str(g.pid)); time.sleep(60)
    """)
    started = time.time()
    _, _, timed_out = run_bounded([sys.executable, "-c", child], ".", 2, 1000)
    assert timed_out and time.time() - started < 15
    time.sleep(0.5)
    try:
        state = open(f"/proc/{pidfile.read_text()}/stat").read().split(")")[-1].split()[0]
    except FileNotFoundError:
        state = "gone"
    assert state in ("Z", "gone")          # dead (a zombie just waits to be reaped)


# ---------------------------------------------------------------- grounding / verification
def test_invented_file_is_flagged_in_the_saved_reply(client, script):
    pid = upload(client)["project_id"]
    script.script = ["I reviewed `UserService.java` and it has a null check bug."]
    events = sse(client, {"message": "check my code", "conversation_id": cid(44), "project_id": pid})
    text = tokens(events)
    assert "Check this reply before trusting it" in text and "UserService.java" in text
    saved = client.get(f"/conversations/{cid(44)}").json()["messages"][-1]["content"]
    assert saved == text                                 # the warning survives a reload


def test_reading_a_file_makes_the_claim_legitimate(client, script):
    pid = upload(client)["project_id"]
    script.script = [tool("read_project_file", path="app/util.py"),
                     "I read util.py and `helper` simply returns 1."]
    events = sse(client, {"message": "what does util.py do?", "conversation_id": cid(45), "project_id": pid})
    assert "Check this reply" not in tokens(events)


def test_evidence_is_remembered_across_turns_without_resending_files(client, script):
    pid = upload(client)["project_id"]
    body = {"conversation_id": cid(46), "project_id": pid, "project_name": "demo"}
    script.script = [tool("read_project_file", path="app/main.py"), "It defines add."]
    sse(client, {**body, "message": "read main.py"})

    script.script = ["Anything else?"]
    sse(client, {**body, "message": "thanks"})
    system = script.requests[-1]["messages"][0]["content"]
    assert "app/main.py (lines 1-2 of 2; defines add)" in system
    assert "return a - b" in system                      # small project is inlined anyway ...
    assert "any file NOT listed here has NOT been read" in system

    stored = client.get(f"/conversations/{cid(46)}").json()["messages"][-1]["meta"]
    assert stored.get("evidence") is None or isinstance(stored["evidence"], list)
    first = client.get(f"/conversations/{cid(46)}").json()["messages"][1]["meta"]
    assert first["evidence"][0]["outline"] == ["add"]


def test_evidence_in_a_fresh_conversation_says_nothing_was_read(client, script):
    pid = upload(client)["project_id"]
    script.script = ["hi"]
    sse(client, {"message": "hello", "conversation_id": cid(47), "project_id": pid})
    assert "none. You have not read any file yet." in script.requests[0]["messages"][0]["content"]
    assert "NEVER RUN in this conversation" in script.requests[0]["messages"][0]["content"]


def test_verifier_cases():
    from agent.verify import check_answer
    files = ["src/main/java/com/x/OrderService.java", "README.md"]
    read = [{"tool": "read_project_file", "path": files[0], "lines": "1-9", "of": 9, "complete": True, "outline": []}]
    passed = [{"tool": "run_project_tests", "status": "passed"}]

    assert check_answer("Look at UserService.java for the bug.", files, "fix it", [])
    assert not check_answer("Look at OrderService.java.", files, "fix it", [])             # basename matches
    assert not check_answer("Create Helper.java:\n```java src/Helper.java\nclass H {}\n```", files, "x", [])
    assert not check_answer("Your Foo.java is fine.", files, "check Foo.java", [])         # user named it
    assert check_answer("I reviewed OrderService.java.", files, "x", [])
    assert not check_answer("I reviewed OrderService.java.", files, "x", read)
    assert check_answer("All tests pass.", files, "x", [])
    assert not check_answer("All tests pass.", files, "x", passed)
    assert not check_answer("Run it; the tests should pass afterwards.", files, "x", [])
    assert not check_answer("```py\nopen('config.json')\n```", files, "x", [])             # code is not prose
    assert check_answer("Edit Anything.java.", [], "x", []) == []                          # no project: no checks


def test_test_status_vocabulary():
    from agent.evidence import test_status
    assert test_status({"status": "passed", "summary": {"tests_run": 4}}) == "passed"
    assert test_status({"status": "passed", "summary": {"tests_run": 0}}) == "no_tests"
    assert test_status({"status": "failed"}) == "failed"
    assert test_status({"status": "timeout"}) == "timed_out"
    assert test_status({"status": "not_run"}) == "not_run"
    assert test_status({"status": "error"}) == "error"


# ---------------------------------------------------------------- limits
def project_dirs():
    import config
    return sorted(p.name for p in config.WORKSPACE_DIR.iterdir() if p.is_dir())


def test_batch_upload_limit_and_nothing_left_behind(client, monkeypatch):
    import config
    monkeypatch.setattr(config, "MAX_BATCH_UPLOAD_MB", 1)
    before = project_dirs()
    big = b"x" * 700_000
    r = client.post("/upload-files", files=[("files", ("a.txt", big)), ("files", ("b.txt", big))])
    assert r.status_code == 400 and "more than 1 MB" in r.json()["detail"]
    assert project_dirs() == before


def test_project_file_count_includes_files_added_later(client, monkeypatch):
    import config
    monkeypatch.setattr(config, "MAX_PROJECT_FILES", 3)
    r = client.post("/upload-files", files=[("files", ("a.py", b"1")), ("files", ("b.py", b"2"))]).json()
    more = client.post("/upload-files", data={"project_id": r["project_id"]},
                       files=[("files", ("c.py", b"3")), ("files", ("d.py", b"4"))])
    assert more.status_code == 400 and "at most 3 files" in more.json()["detail"]
    files = client.get(f"/projects/{r['project_id']}/files").json()["files"]
    assert sorted(files) == ["a.py", "b.py"]              # the rejected batch added nothing


def test_project_total_size_includes_existing_files(client, monkeypatch):
    import config
    monkeypatch.setattr(config, "MAX_UNZIPPED_MB", 1)
    r = client.post("/upload-files", files=[("files", ("a.bin", b"x" * 600_000))]).json()
    more = client.post("/upload-files", data={"project_id": r["project_id"]},
                       files=[("files", ("b.bin", b"x" * 600_000))])
    assert more.status_code == 400 and "over 1 MB" in more.json()["detail"]


def test_generated_zip_limits(client, monkeypatch):
    import config
    monkeypatch.setattr(config, "MAX_ZIP_DOWNLOAD_FILES", 2)
    files = [{"path": f"f{i}.txt", "content": "x"} for i in range(3)]
    assert "Too many files" in client.post("/download-zip", json={"files": files}).json()["detail"]
    monkeypatch.setattr(config, "MAX_ZIP_DOWNLOAD_FILES", 500)
    monkeypatch.setattr(config, "MAX_ZIP_DOWNLOAD_MB", 1)
    big = [{"path": "a.txt", "content": "x" * 700_000}, {"path": "b.txt", "content": "x" * 700_000}]
    r = client.post("/download-zip", json={"files": big})
    assert r.status_code == 400 and "more than 1 MB" in r.json()["detail"]


def test_atlas_cannot_modify_a_project(client, script):
    """There is no Apply: code is only ever shown, never written into the uploaded copy."""
    pid = upload(client)["project_id"]
    before = {f: client.get(f"/projects/{pid}/file", params={"path": f}).json()["content"]
              for f in client.get(f"/projects/{pid}/files").json()["files"]}

    for method, path in [("post", f"/projects/{pid}/apply"), ("get", f"/projects/{pid}/download")]:
        assert getattr(client, method)(path, **({"json": {"files": []}} if method == "post" else {})).status_code in (404, 405)

    script.script = ["Here is the fix:\n```python app/main.py\ndef add(a, b):\n    return a + b\n```"]
    sse(client, {"message": "fix add", "conversation_id": cid(60), "project_id": pid})
    after = {f: client.get(f"/projects/{pid}/file", params={"path": f}).json()["content"] for f in before}
    assert after == before
    assert not hasattr(__import__("services.project_service", fromlist=["x"]), "apply_files")
    assert "apply_enabled" not in client.get("/health").json()


# ---------------------------------------------------------------- upload integrity
def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_duplicate_paths_in_one_upload_are_rejected_before_anything_is_written(client):
    import config
    before = project_dirs()
    r = client.post("/upload-files", files=[("files", ("a.py", b"1")), ("files", ("A.PY", b"2"))])
    assert r.status_code == 400 and "same path" in r.json()["detail"]
    assert project_dirs() == before

    pid = upload(client)["project_id"]
    root = config.WORKSPACE_DIR / pid
    state = snapshot(root)
    r = client.post("/upload-files", data={"project_id": pid},
                    files=[("files", ("x/new.py", b"1")), ("files", ("x/new.py", b"2"))])
    assert r.status_code == 400 and snapshot(root) == state

    z = client.post("/download-zip", json={"files": [{"path": "a.py", "content": "1"}, {"path": "a.py", "content": "2"}]})
    assert z.status_code == 400 and "same path" in z.json()["detail"]


@pytest.mark.parametrize("fail_on", [1, 2, 3, 4, 5])
def test_upload_commit_is_all_or_nothing_at_every_step(client, monkeypatch, fail_on):
    """Fail the Nth file operation while moving files into place: the project must be untouched."""
    import config
    from services import project_service
    pid = upload(client)["project_id"]
    root = config.WORKSPACE_DIR / pid
    state = snapshot(root)

    real, calls = os.replace, []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) == fail_on:
            raise OSError("disk full")
        return real(src, dst)

    monkeypatch.setattr(project_service.os, "replace", flaky)
    r = client.post("/upload-files", data={"project_id": pid}, files=[
        ("files", ("app/main.py", b"REPLACED MAIN")),      # existing file (2 operations)
        ("files", ("app/util.py", b"REPLACED UTIL")),      # existing file (2 operations)
        ("files", ("app/brand_new.py", b"NEW")),           # new file (1 operation)
    ])
    monkeypatch.undo()

    assert r.status_code >= 400
    assert snapshot(root) == state                          # originals back, new file gone, no leftovers


def test_failed_upload_of_a_new_project_leaves_no_folder(client, monkeypatch):
    from services import project_service
    before = project_dirs()
    monkeypatch.setattr(project_service.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("boom")))
    r = client.post("/upload-files", files=[("files", ("a.py", b"1"))])
    monkeypatch.undo()
    assert r.status_code >= 400 and project_dirs() == before


# ---------------------------------------------------------------- locking
@pytest.fixture()
def live():
    """The real app on a real port, so requests genuinely overlap."""
    import socket
    import threading
    import uvicorn
    import main
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(5)


def test_a_busy_project_answers_409_and_never_freezes_the_server(live, monkeypatch):
    import threading
    import httpx
    import config
    from services.locks import project_lock
    monkeypatch.setattr(config, "LOCK_TIMEOUT_SECONDS", 1.5)

    pid = httpx.post(live + "/upload-project", files={"file": ("p.zip", make_zip({"a.py": "1"}))}).json()["project_id"]
    result = {}

    def late_upload():
        started = time.time()
        r = httpx.post(live + "/upload-files", data={"project_id": pid}, files=[("files", ("b.py", b"2"))], timeout=20)
        result.update(status=r.status_code, detail=r.json()["detail"], took=time.time() - started)

    with project_lock(pid, timeout=0):                     # e.g. a test run is in progress
        worker = threading.Thread(target=late_upload)
        worker.start()
        time.sleep(0.4)                                    # the upload is now waiting for the lock
        started = time.time()
        assert httpx.get(live + "/health", timeout=10).status_code == 200
        assert time.time() - started < 1.0                 # other requests are not blocked meanwhile
        worker.join(10)

    assert result["status"] == 409 and "already using this project" in result["detail"]
    assert 1.0 < result["took"] < 6.0                      # it waited, then gave up politely
    assert httpx.get(f"{live}/projects/{pid}/files").json()["files"] == ["a.py"]


def test_simultaneous_uploads_to_one_project_do_not_corrupt_each_other(live):
    import threading
    import httpx
    pid = httpx.post(live + "/upload-project", files={"file": ("p.zip", make_zip({"seed.py": "0"}))}).json()["project_id"]
    statuses = []

    def add(i):
        r = httpx.post(live + "/upload-files", data={"project_id": pid},
                       files=[("files", (f"f{i}.py", f"x{i}".encode() * 5000))], timeout=30)
        statuses.append(r.status_code)

    threads = [threading.Thread(target=add, args=(i,)) for i in range(8)]
    [t.start() for t in threads]; [t.join(40) for t in threads]

    assert statuses.count(200) == 8
    files = httpx.get(f"{live}/projects/{pid}/files").json()["files"]
    assert sorted(files) == sorted(["seed.py"] + [f"f{i}.py" for i in range(8)])
    for i in range(8):
        got = httpx.get(f"{live}/projects/{pid}/file", params={"path": f"f{i}.py"}).json()["content"]
        assert got == f"x{i}" * 5000


def test_tests_cannot_start_twice_and_a_busy_project_is_not_deleted(client, monkeypatch):
    import config
    from services import db
    from services.locks import project_lock
    from services.test_runner import run_project_tests
    pid = upload(client, POM)["project_id"]
    db.get_or_create_conversation(cid(61)); db.update_conversation(cid(61), project_id=pid)

    with project_lock(pid, timeout=0):
        result = run_project_tests(pid)
        assert result["status"] == "error" and "already using this project" in result["error"]

        monkeypatch.setattr(config, "LOCK_TIMEOUT_SECONDS", 0.2)
        r = client.delete(f"/conversations/{cid(61)}")
        assert r.status_code == 409
        assert (config.WORKSPACE_DIR / pid).is_dir()                       # folder still there
        assert client.get(f"/conversations/{cid(61)}").status_code == 200  # and so is the chat

    assert client.delete(f"/conversations/{cid(61)}").status_code == 200   # fine once it is free
    assert not (config.WORKSPACE_DIR / pid).exists()


# ---------------------------------------------------------------- big project / long conversation
def make_big_project(folders=30, files_each=10, size=15_000):
    import uuid
    import config
    pid = str(uuid.uuid4())
    root = config.WORKSPACE_DIR / pid
    for d in range(folders):
        (root / f"pkg{d}").mkdir(parents=True)
        for f in range(files_each):
            body = "".join(f"def func_{d}_{f}_{i}(value):\n    return value + {i}\n\n" for i in range(size // 50))
            (root / f"pkg{d}" / f"mod{f}.py").write_text(body)
    return pid


def read_record(path, i):
    return {"tool": "read_project_file", "path": path, "lines": f"1-{100 + i}", "of": 400,
            "complete": False, "outline": [f"func_{i}_{j}" for j in range(12)]}


def seed_conversation(conversation_id, pid, turns=60):
    from services import db
    db.get_or_create_conversation(conversation_id)
    db.update_conversation(conversation_id, project_id=pid, project_name="big", mode="code")
    for i in range(turns):
        db.add_message(conversation_id, "user", f"TURN-{i} " + "please look into this problem again " * 14)
        evidence = []
        if i == 1:
            evidence = [{"tool": "run_project_tests", "status": "passed",
                         "summary": {"tests_run": 12, "failures": 0, "errors": 0, "skipped": 0}}]
        elif i >= 2:
            evidence = [read_record(f"pkg{i % 30}/mod{i % 10}.py", i)]
        db.add_message(conversation_id, "assistant", f"ANSWER-{i} " + "here is what I found out earlier " * 14,
                       {"mode": "code", "model": "m", "evidence": evidence})


def sent(script, index=0):
    return script.requests[index]["messages"]


def test_large_project_and_long_conversation_stay_inside_the_budget(client, script):
    import config
    from agent.context import DIGEST_MARKER, estimate_tokens
    pid = make_big_project()
    seed_conversation(cid(62), pid)
    script.script = ["noted"]

    question = "Now explain pkg29/mod9.py in detail."
    sse(client, {"message": question, "conversation_id": cid(62), "project_id": pid, "project_name": "big"})
    messages = sent(script)
    system = messages[0]["content"]
    total = sum(estimate_tokens(m["content"]) for m in messages)

    assert total <= config.PROMPT_BUDGET_TOKENS                          # fits the model's window
    assert messages[-1]["content"] == question                           # your message is never cut
    assert len(system) <= config.PROMPT_BUDGET_TOKENS * config.CHARS_PER_TOKEN * (config.SYSTEM_SHARE + 0.1)

    # essential evidence survived 58 later file reads and heavy history trimming
    assert "Tests, most recent run: PASSED (12 run, 0 failures, 0 errors)" in system
    assert "pkg9/mod9.py" in system or "pkg8/mod8.py" in system          # the newest reads are listed in detail
    assert "this list is incomplete" in system                            # and it admits older ones may be missing
    assert "NOT been read" not in system                                  # ... rather than falsely claiming they were not read

    # a big project is summarised by its file list, not stuffed into the prompt
    assert "PROJECT FILES (300 files" in system and "more files" in system
    assert "=====" not in system
    assert "def func_0_0_0" not in system

    # old history was replaced by a short, labelled digest of the most recent dropped turns
    assert DIGEST_MARKER in system and "may be incomplete or wrong" in system
    assert "TURN-0 " not in "".join(m["content"] for m in messages)      # the very oldest is gone
    assert "TURN-5" in system or "TURN-4" in system or any("TURN-5" in m["content"] for m in messages)


def test_answer_length_shrinks_to_the_room_left(client, script):
    import config
    pid = make_big_project(folders=2, files_each=2)
    script.script = ["ok", "ok"]
    sse(client, {"message": "hi", "conversation_id": cid(63), "project_id": pid, "project_name": "p"})
    short_prompt_room = script.requests[0]["options"]["num_predict"]
    assert short_prompt_room == config.NUM_PREDICT                        # plenty of room: full length allowed

    seed_conversation(cid(64), make_big_project(folders=2, files_each=2), turns=40)
    sse(client, {"message": "word " * 4000, "conversation_id": cid(64), "project_id": None})
    room = script.requests[1]["options"]["num_predict"]
    assert 256 <= room < config.NUM_PREDICT                               # long prompt: answer is capped


def test_a_huge_pasted_message_is_cut_but_evidence_and_rules_survive(client, script):
    import config
    from agent.context import estimate_tokens
    pid = make_big_project(folders=3, files_each=3)
    seed_conversation(cid(65), pid, turns=12)
    script.script = ["ok"]
    events = sse(client, {"message": "ANALYSE THIS " + "x = 1\n" * 30000, "conversation_id": cid(65),
                          "project_id": pid, "project_name": "p"})
    assert any(e["type"] == "warning" and "cut" in e["text"] for e in events)
    messages = sent(script)
    system = messages[0]["content"]
    assert "Tests, most recent run: PASSED" in system and "GROUNDING RULES" in system
    assert sum(estimate_tokens(m["content"]) for m in messages) <= config.PROMPT_BUDGET_TOKENS


def test_context_pressure_warning_uses_ollamas_own_token_count(client, script):
    import config
    script.script = ["a", "b"]
    script.prompt_eval_count = config.NUM_CTX - 10
    events = sse(client, {"message": "hello", "conversation_id": cid(66), "mode": "chat"})
    assert any(e["type"] == "warning" and "filled the model's memory" in e["text"] for e in events)
    script.prompt_eval_count = 900
    events = sse(client, {"message": "hello again", "conversation_id": cid(66), "mode": "chat"})
    assert not any(e["type"] == "warning" and "filled the model's memory" in e["text"] for e in events)


def test_latest_test_result_is_never_pushed_out_of_the_evidence():
    import config
    from agent import evidence as ev
    long_path = "src/main/java/com/example/service/Service{}Impl.java"
    reads = [[read_record(long_path.format(i), i)] for i in range(80)]
    test = [{"tool": "run_project_tests", "status": "failed"}]
    merged = ev.merge([test] + reads)
    assert ev.last_test_status(merged)[0] == "failed" and len(merged) <= ev.MAX_RECORDS

    cap = int(config.PROMPT_BUDGET_TOKENS * config.CHARS_PER_TOKEN * config.EVIDENCE_SHARE)
    uncapped = ev.render_block(merged, max_chars=10**9)
    assert len(uncapped) > cap * 1.3                     # realistic paths really do overflow the cap ...
    block = ev.render_block(merged)
    assert len(block) <= cap                             # ... and the cap is enforced
    assert "FAILED" in block and "Service79Impl.java" in block


@pytest.mark.parametrize("window,compact", [(8192, False), (6144, True), (4096, True)])
def test_other_window_sizes_still_leave_room_for_the_conversation(window, compact):
    """Run in a fresh interpreter, because the budget is computed from LC_NUM_CTX at start-up."""
    import subprocess
    code = textwrap.dedent("""
        import sys, json
        sys.path.insert(0, '.')
        import config
        from agent.agent import build_system_prompt
        from agent.context import digest_chars
        from agent import evidence as ev
        recs = [{"tool": "read_project_file", "path": f"src/main/java/com/example/Service{i}Impl.java",
                 "lines": "1-300", "of": 900, "complete": False, "outline": [f"m{j}" for j in range(14)]} for i in range(40)]
        base = build_system_prompt("code", "p" * 36, "", ev.render_block(recs))
        cap = int(config.PROMPT_BUDGET_TOKENS * config.CHARS_PER_TOKEN * config.SYSTEM_SHARE)
        floor = int(config.PROMPT_BUDGET_TOKENS * config.CHARS_PER_TOKEN * 0.06)
        ctx = max(floor, min(int(config.MAX_FILE_CHARS * 0.9), cap - len(base)))
        system = (len(base) + ctx + digest_chars()) / config.CHARS_PER_TOKEN
        print(json.dumps({"compact": config.COMPACT_PROMPTS, "budget": config.PROMPT_BUDGET_TOKENS, "system": system}))
    """)
    env = {**os.environ, "LC_NUM_CTX": str(window)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         cwd=os.path.dirname(os.path.dirname(__file__)))
    result = json.loads(out.stdout)
    assert result["compact"] is compact
    assert result["system"] <= result["budget"] * 0.75
    assert result["budget"] - result["system"] >= 700                      # real room left to talk

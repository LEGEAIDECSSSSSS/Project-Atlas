"""Seeded fuzzing: whatever the model says, or the user types, Atlas finishes cleanly."""
import json
import random
import string
import time

from agent import agent as A
from agent import planner, verify
from tests.conftest import SAMPLE_PROJECT
from tests.test_agent_gate import World, run

TOOLS = list(A.TOOL_FUNCTIONS) + ["nope", "", "Search_Project"]
WORDS = ["login", "authenticate_user", "/login", "backend/main.py", "```", "{", "}", "tool", "\u00e9\u4e2d\u6587",
         "\x00", "line 5", "`ghost_fn()`", "tests passed", "../x", "NOT FOUND", "a" * 3000, "\n" * 50]


def random_arguments(rng):
    pool = [{}, {"name": "authenticate_user"}, {"query": "login"}, {"path": "backend/main.py", "start_line": 1, "end_line": 3},
            {"path": "../../etc/passwd"}, {"name": 5}, {"query": None}, {"limit": -3}, {"line": "x"},
            {"name": "login", "path": "backend/api/auth.py", "line": 8}, {"query": "/login", "depth": 99},
            {"expression": "1+1"}, {"expressions": []}, {"ecosystem": "python"}, {"junk": [1, 2, 3]}]
    return rng.choice(pool)


def random_reply(rng):
    kind = rng.randrange(9)
    if kind == 0:
        return json.dumps({"tool": rng.choice(TOOLS), "arguments": random_arguments(rng)})
    if kind == 1:
        return json.dumps({"tool": rng.choice(TOOLS), "arguments": random_arguments(rng)})[:rng.randrange(1, 60)]
    if kind == 2:
        return "Sure. " + json.dumps({"name": rng.choice(TOOLS), "parameters": random_arguments(rng)}) + " done"
    if kind == 3:
        return "```json\n" + json.dumps({"tool": rng.choice(TOOLS), "arguments": random_arguments(rng)}) + "\n```"
    if kind == 4:
        return "".join(rng.choice(WORDS) for _ in range(rng.randrange(1, 8)))
    if kind == 5:
        return json.dumps({"tool": "find_symbol", "arguments": {"name": "x"}}) * rng.randrange(2, 4)
    if kind == 6:
        return ""
    if kind == 7:
        return json.dumps({"answer": rng.choice(WORDS)})
    return "".join(rng.choice(string.printable) for _ in range(rng.randrange(0, 400)))


MESSAGES = ["How does the /login endpoint work?", "who calls authenticate_user", "read backend/main.py lines 1-5",
            "list the files", "check my code", "Explain the zzz_missing_function method",
            "find where login is defined", "hello", "x", "", "a b c " * 400, "show ../../etc/passwd.py",
            "What tests cover find_user?", "How does the app start up?", "```py\nx=1\n``` explain this",
            "Search for \x00 \u202e weird", "compare backend/main.py and backend/api/auth.py"]


def test_agent_always_terminates_cleanly_whatever_the_model_says():
    rng = random.Random(20240601)
    world = World(SAMPLE_PROJECT)
    try:
        for i in range(120):
            message = rng.choice(MESSAGES)
            replies = [random_reply(rng) for _ in range(30)]
            mode = rng.choice(["code", "chat"])
            out = run(message, replies, world=world, mode=mode, close=False,
                      project_id=rng.choice([world.project_id, None]), max_calls=rng.choice([1, 2, 4]))
            done = [e for e in out.events if e["type"] == "done"]
            assert len(done) == 1 and out.events[-1]["type"] == "done", (i, message)
            assert out.model_calls <= 14, (i, message, out.model_calls)
            text = done[0]["response"]
            assert isinstance(text, str)
            # a tool request is never presented as the answer
            assert not A.looks_like_tool_call(text), (i, text[:80])
    finally:
        world.close()


def test_user_input_never_crashes_classification_planning_or_verification():
    rng = random.Random(7)
    started = time.monotonic()
    for _ in range(400):
        text = "".join(rng.choice(WORDS + list(string.printable)) for _ in range(rng.randrange(0, 40)))
        A.classify_request(text, True, ["a.py", "Dockerfile"], rng.choice(["code", "chat", None]))
        A.classify_request(text, False, None)
        planner.make_plan(text)
        verify.check_answer(text, ["a.py"], text, [{"tool": "read_project_file", "path": "a.py", "source_text": text}],
                            facts={"symbols": {"x"}, "files": {"a.py"}, "call_edges": set(), "line_counts": {"a.py": 3}})
        A.parse_tool_calls(text)
        A.looks_like_tool_attempt(text)
        A.diagnose_bad_tool_call(text)
        A.validate_tool_arguments(rng.choice(TOOLS), rng.choice([random_arguments(rng), text, None, [1]]))
    assert time.monotonic() - started < 20


def test_pathological_inputs_stay_fast():
    nasty = ["(" * 20000, "{" * 20000, "`" * 20000, "a_" * 20000, "line " + "1," * 20000, "/" * 20000,
             "`" + "a" * 50000 + "`", "x.py:" + "9" * 5000, "def " * 20000, " " * 100000,
             "a." * 30000 + "py", "lines " + "1-2, " * 5000, "a_" * 20000]
    started = time.monotonic()
    for text in nasty:
        A.classify_request(text, True, None, "code")
        planner.make_plan(text)
        verify.check_answer(text, ["a.py"], "q", [], facts={"symbols": {"a"}, "files": set(), "call_edges": set(), "line_counts": {}})
        A.parse_tool_calls(text)
        A.looks_like_tool_attempt(text)
    assert time.monotonic() - started < 6

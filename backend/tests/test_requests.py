"""Request classification on a wide set of phrasings, identifier matching, large-file reads."""
import importlib

import pytest

import config
from agent import agent as A
from tests.test_agent_gate import World, run

KNOWN = ["backend/agent/agent.py", "backend/api/chat.py", "docs/my notes.md", "Dockerfile", "README.md"]


def kind(text, mode="code", has_project=True):
    r = A.classify_request(text, has_project, KNOWN, mode)
    return r.kind if r else None


@pytest.mark.parametrize("text,expected", [
    # named files and ranges
    ("Show me the first 30 lines of backend/api/chat.py", "read"),
    ("can you open README.md", "read"),
    ("print lines 10 to 20 of agent.py", "read"),
    ("what's in the Dockerfile?", "read"),
    ("Please read `backend/agent/agent.py` and tell me what is wrong", "read"),
    ("show \"docs/my notes.md\"", "read"),
    # listings
    ("list the files", "list"), ("Show me all the files in the project", "list"),
    ("what files are in this project?", "list"), ("give me the project structure", "list"),
    # searches
    ("find where run_agent_stream is defined", "search"), ("search the project for TODO comments", "search"),
    ("where is the login code?", "search"), ("grep for usages of fetch_user", "search"),
    # analysis of named things without a file
    ("Explain the build_project_context function", "analyze"),
    ("why does my code throw an error on startup?", "analyze"),
    # everything else about the project is investigated
    ("What happens after a user signs in?", "explore"),
    ("Add a password reset endpoint", "explore"),
    ("how is authentication implemented", "explore"),
    ("Summarize the architecture of this app", "explore"),
    ("Where do we validate emails", "search"),
    ("refactor the payment module to use async", "explore"),
])
def test_project_requests(text, expected):
    assert kind(text) == expected


@pytest.mark.parametrize("text", [
    "hello", "hi there!", "thanks a lot", "ok", "good morning", "yes", "What are the pros and cons of microservices?",
    "List the pros and cons of file systems", "What is the difference between a list and a tuple?",
    "How do I write a function in Python?", "Explain how recursion works",
    "Show me how to read files in Python", "I renamed agent.py to main.py yesterday",
    "see https://example.com/readme.md for details", "version 3.10 came out, e.g. today",
    "Explain this code:\n```python\ndef f(): pass\n```", "mail me at bob@corp.py please",
])
def test_conversation_and_concepts_are_left_alone(text):
    assert kind(text) is None


def test_chat_mode_only_reacts_to_clearly_project_specific_questions():
    assert kind("Explain how recursion works", mode="chat") is None
    assert kind("plan my weekend", mode="chat") is None
    assert kind("What does `build_context` do?", mode="chat") == "analyze"
    assert kind("why does my code fail", mode="chat") == "analyze"


def test_paths_with_spaces_are_found_when_quoted():
    r = A.classify_request('read "docs/my notes.md"', True, KNOWN, "code")
    assert r.paths == ["docs/my notes.md"]
    assert A.resolve_path("docs/my notes.md", KNOWN) == ("docs/my notes.md", None)
    assert A.resolve_path("my notes.md", KNOWN)[0] == "docs/my notes.md"


def test_without_a_project_only_clear_file_requests_are_refused():
    assert kind("read agent.py", has_project=False) == "read"
    assert kind("find where login is defined", has_project=False) is None
    assert kind("find where login is defined in my project", has_project=False) == "search"


@pytest.mark.parametrize("term,text,expected", [
    ("user", "def user(): pass", True), ("user", "class User:", True), ("user", "user_service = 1", False),
    ("user", "UserService()", False), ("user", "get_user(1)", False), ("user", "def get_user_profile():", False),
    ("user_service", "from x import user_service", True), ("user_service", "user_service_v2 = 1", False),
    ("UserService", "new UserService()", True), ("UserService", "MyUserService", False),
    ("get_user", "x = get_user()", True), ("get_user", "get_user_profile()", False),
    ("get_user_profile", "get_user_profile()", True), ("user", "the user logged in", True),
])
def test_identifiers_match_only_as_whole_names(term, text, expected):
    assert A._mentions([term], text) is expected


def test_a_similar_name_is_not_the_requested_name(project):
    w = project({"a.py": "def get_user_profile():\n    return 1\n"})
    out = run("Explain the get_user function", ["There is no get_user, only get_user_profile."], mode="code", world=w)
    prompt = "\n".join(m["content"] for m in out.model.calls[0])
    assert "did not appear in the code that was read" in prompt
    assert out.text == "There is no get_user, only get_user_profile."


def test_ignored_folders_are_configurable(project, monkeypatch):
    from services import project_service as ps
    w = project({"src/a.py": "needle\n", "build/gen.py": "needle\n", "vendor_x/b.py": "needle\n"})
    assert "build/gen.py" not in ps.list_project_files(w.project_id)       # built-in default
    monkeypatch.setattr(config, "EXTRA_IGNORED_DIRS", ("vendor_x",))
    monkeypatch.setattr(config, "INCLUDE_DIRS", ("build",))
    import services.project_service as reloaded
    importlib.reload(reloaded)
    try:
        listed = reloaded.list_project_files(w.project_id)
        assert "build/gen.py" in listed and "vendor_x/b.py" not in listed
    finally:
        monkeypatch.setattr(config, "EXTRA_IGNORED_DIRS", ())
        monkeypatch.setattr(config, "INCLUDE_DIRS", ())
        importlib.reload(reloaded)


def test_several_named_files_that_cannot_fit_are_not_sent_as_partial_evidence(project):
    body = "\n".join(f"def f{i}():\n    return {i}  # " + "x" * 60 for i in range(150))
    w = project({"a.py": body, "b.py": body, "c.py": body})
    out = run("Compare a.py and b.py and c.py", ["They are similar."], world=w, mode="code")
    assert out.model_calls == 0
    assert "too large to fit" in out.text


def test_a_cut_off_read_is_not_sent_to_the_model_as_complete_evidence(project, monkeypatch):
    monkeypatch.setattr(config, "PROMPT_BUDGET_TOKENS", 14000)
    monkeypatch.setattr(config, "NUM_CTX", 16384)
    monkeypatch.setattr(A, "MAX_FILE_CHARS", 3000)
    body = "\n".join(f"line {i} " + "y" * 70 for i in range(1, 400))
    w = project({"long.py": body})
    out = run("Explain long.py", ["It has many lines."], world=w, mode="code")
    assert out.model_calls == 0
    assert "cut at the size limit" in out.text


def test_the_model_cannot_set_the_internal_read_budget(project):
    w = project({"a.py": "x = 1\n" * 50})
    call = '{"tool": "read_project_file", "arguments": {"path": "a.py", "max_chars": 5}}'
    out = run("what does the app do overall?", [call, "Done."], world=w, mode="chat")
    seen = "\n".join(m["content"] for m in out.model.calls[1])
    assert "Ignored argument(s) max_chars" in seen and "x = 1" in seen


def test_explicit_function_request_uses_symbol_prefetch():
    r = A.classify_request("Inspect only the `login` function in auth_controller.py", True,
                           ["app/controllers/auth_controller.py"], "code")
    assert r.kind == "read"
    assert r.symbol == "login"
    assert r.paths == ["auth_controller.py"]


def test_named_function_prefetch_reads_only_the_requested_symbol(project):
    w = project({"auth_controller.py": (
        "def helper():\n    return 0\n\n"
        "async def login(form_data):\n    return authenticate(form_data)\n\n"
        "def logout():\n    return None\n"
    )})
    out = run("Inspect only the `login` function in auth_controller.py",
              ["The function delegates to authenticate."], world=w, mode="code")
    assert out.model_calls == 1
    tool_results = [m["content"] for m in out.model.calls[0]
                    if m["content"].startswith("TOOL RESULT")]
    assert len(tool_results) == 1
    assert "async def login" in tool_results[0]
    assert "def helper" not in tool_results[0]
    assert "def logout" not in tool_results[0]

"""Agent protocol and investigation: parsing, duplicates, route/startup tracing, pagination notes."""
import json

from agent import agent as A
from tests.conftest import SAMPLE_PROJECT
from tests.test_agent_gate import World, run


def sent(out, index=0):
    """Everything the model was shown on request `index`, as plain text."""
    return "\n".join(m["content"] for m in out.model.calls[index])


def tool_json(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


# ---------------------------------------------------------------------------- parsing
def test_json_documents_are_answers_not_tool_calls():
    out = run("give me a json person", ['{"name": "Ada", "age": 36}'], project_id=None)
    assert out.text == '{"name": "Ada", "age": 36}' and out.streamed == out.text


def test_truncated_tool_calls_are_still_recognised_as_attempts():
    assert A.looks_like_tool_attempt('{"tool": "search_proj')
    assert not A.looks_like_tool_attempt('{"name": "Ada", "age": 36}')
    out = run("hello there", ['{"tool": "search_project", "argum', "fixed: hello"], project_id=None)
    assert out.text == "fixed: hello" and "INVALID TOOL CALL" in sent(out, 1)


def test_several_tool_calls_in_one_message_run_one_and_say_so(project):
    w = project(SAMPLE_PROJECT)
    two = tool_json("find_symbol", name="find_user") + "\n" + tool_json("find_symbol", name="health")
    out = run("what does the app do overall?", [two, "Done."], world=w)
    assert out.text == "Done."
    assert "only the first was run" in sent(out, 1)


def test_prose_around_a_tool_call_does_not_hide_it(project):
    w = project(SAMPLE_PROJECT)
    out = run("what does the app do overall?",
              ["I will look.\n```json\n" + tool_json("find_entry_points") + "\n```\nthen wait", "Done."], world=w)
    assert out.text == "Done." and any(t["name"] == "find_entry_points" for t in out.tools)


# ---------------------------------------------------------------------------- protocol
def test_unknown_arguments_are_reported_back_not_dropped_silently(project):
    w = project(SAMPLE_PROJECT)
    call = tool_json("find_symbol", name="find_user", colour="red", project_id="x")
    out = run("what does the app do overall?", [call, "Done."], world=w)
    seen = sent(out, 1)
    assert "Ignored argument(s) colour" in seen and "project_id" not in seen.split("Ignored argument(s)")[1][:40]


def test_an_identical_repeat_is_refused_and_not_counted_as_progress(project):
    w = project(SAMPLE_PROJECT)
    call = tool_json("find_symbol", name="find_user")
    out = run("what does the app do overall?", [call, call, "Done."], world=w)
    oks = [t["ok"] for t in out.tools][-2:]
    assert oks == [True, False] and "Duplicate call" in sent(out, 2)


def test_circling_searches_get_a_stop_searching_hint(project):
    w = project(SAMPLE_PROJECT)
    calls = [tool_json("find_symbol", name="find_user"), tool_json("find_symbol", name="find_use"),
             tool_json("find_symbol", name="find_us"), tool_json("find_symbol", name="find_u"), "Done."]
    out = run("what does the app do overall?", calls, world=w)
    assert any("showed nothing new" in "".join(m["content"] for m in c) for c in out.model.calls)


def test_bad_arguments_get_a_specific_message(project):
    w = project(SAMPLE_PROJECT)
    out = run("what does the app do overall?", [tool_json("read_symbol"), "Done."], world=w)
    assert "Invalid arguments for read_symbol" in sent(out, 1) and "or both" in sent(out, 1)


def test_index_tools_resolve_short_paths_and_reject_traversal(project):
    w = project(SAMPLE_PROJECT)
    out = run("what does the app do overall?",
              [tool_json("find_imports", path="auth_service.py"), tool_json("find_imports", path="../../etc/passwd"), "Done."],
              world=w)
    ok = [t["ok"] for t in out.tools][-2:]
    assert ok == [True, False] and "which is not allowed" in sent(out, 2)


# ---------------------------------------------------------------------------- investigation
def test_endpoint_questions_trace_the_route_and_read_the_chain(project):
    w = project(SAMPLE_PROJECT)
    out = run("How does the /login endpoint work?", ["It calls authenticate_user."], mode="code", world=w)
    labels = [t["label"] for t in out.tools]
    assert "Traced route /login" in labels
    reads = [l for l in labels if l.startswith("Read ")]
    assert any("authenticate_user" in l or "login" in l for l in reads)
    prompt = sent(out)
    assert "def authenticate_user" in prompt and "frontend" not in prompt.split("ATLAS INVESTIGATION")[0][:0]
    assert out.text == "It calls authenticate_user."


def test_startup_questions_find_entry_points(project):
    w = project(SAMPLE_PROJECT)
    out = run("How does the app start up?", ["It runs main()."], mode="code", world=w)
    assert any(t["name"] == "find_entry_points" for t in out.tools)
    assert "backend/main.py" in sent(out)


def test_caller_and_test_questions_use_the_graph_tools(project):
    w = project(SAMPLE_PROJECT)
    try:
        out = run("who calls authenticate_user?", ["login and a test."], mode="code", world=w,
                  close=False)
        assert any(t["name"] == "find_callers" for t in out.tools)
        out = run("which tests cover authenticate_user?", ["test_login_ok."], mode="code", world=w,
                  close=False)
        assert any(t["name"] == "find_tests_for_symbol" for t in out.tools)
    finally:
        w.close()


def test_vague_questions_start_from_the_entry_points(project):
    w = project(SAMPLE_PROJECT)
    out = run("check my code", ["Looks fine."], mode="code", world=w)
    assert any(t["name"] == "find_entry_points" for t in out.tools)
    assert any(t["label"].startswith("Read backend/main.py") for t in out.tools)


def test_small_talk_and_conceptual_questions_trigger_no_investigation(project):
    w = project(SAMPLE_PROJECT)
    for text in ["hello there", "thanks!", "What are the pros and cons of microservices?"]:
        out = run(text, ["ok"], mode="code", world=w)
        assert out.tools == [] and out.model_calls == 1


def test_the_investigation_stays_inside_its_budget(project, monkeypatch):
    import config
    big = {f"m{i}.py": "def target_fn():\n" + "    x = 1\n" * 300 for i in range(8)}
    w = project(big)
    out = run("Explain target_fn", ["ok"], mode="code", world=w)
    used = sum(len(m["content"]) for m in out.model.calls[0] if m["role"] == "user")
    assert used < config.PROMPT_BUDGET_TOKENS * config.CHARS_PER_TOKEN


def test_one_search_shows_several_ranked_files_with_complete_definitions(project):
    files = {f"mod{i}.py": f"import os\n\ndef compute_total():\n    return {i}\n" + "# pad\n" * 600 for i in range(3)}
    w = project(files)
    out = run("find where compute_total is defined", ["It is in the modules."], mode="code", world=w)
    prompt = sent(out)
    assert prompt.count("BEGIN FILE: mod") >= 2
    assert "def compute_total" in prompt and "complete" in prompt


def test_search_results_say_when_more_pages_exist(project):
    files = {f"f{i:02d}.py": "needle = 1\n" for i in range(30)}
    w = project(files)
    out = run("find where needle is used", ["Many places."], mode="code", world=w)
    prompt = sent(out)
    assert "Showing 8 of 30 matching files" in prompt and "offset=8" in prompt


def test_the_model_can_page_forward(project):
    files = {f"f{i:02d}.py": "needle = 1\n" for i in range(30)}
    w = project(files)
    page2 = tool_json("search_project", query="needle", offset=8, limit=8)
    out = run("find where needle is used", [page2, "Done."], mode="code", world=w)
    assert out.text == "Done." and "starting at 8" in sent(out, 1)


def test_negative_results_are_honest_after_two_empty_searches(project):
    w = project(SAMPLE_PROJECT)
    out = run("Where is the refund_payment function?", ["I could not find refund_payment."], mode="code", world=w)
    assert out.model_calls == 1 and "Nothing matching was found" in sent(out)


def test_follow_up_messages_still_work_with_history(project):
    w = project(SAMPLE_PROJECT)
    history = [{"role": "user", "content": "How does login work?"}, {"role": "assistant", "content": "It checks credentials."}]
    out = run("and what about find_user?", ["It reads USERS."], mode="code", world=w, history=history)
    assert out.text == "It reads USERS." and "How does login work?" in sent(out)


def test_investigation_survives_a_broken_index(project, monkeypatch):
    w = project(SAMPLE_PROJECT)
    from services import code_index
    monkeypatch.setattr(code_index, "get_index", lambda pid: (_ for _ in ()).throw(RuntimeError("boom")))
    out = run("How does the /login endpoint work?", ["I could not trace it."], mode="code", world=w)
    assert out.model_calls >= 1 and out.done["type"] == "done"

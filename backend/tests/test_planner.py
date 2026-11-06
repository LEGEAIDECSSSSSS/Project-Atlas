"""The deterministic request planner."""
import pytest

from agent import planner


def plan(text):
    return planner.make_plan(text)


def test_identifiers_become_symbols_and_never_duplicate_words():
    p = plan("Explain the run_agent_stream method")
    assert p.symbols == ["run_agent_stream"]
    assert p.queries[0] == "run_agent_stream"            # not "run_agent_stream|run_agent_stream"
    assert p.queries[1] == "run agent stream"            # a second spelling, for negative evidence


def test_camel_case_dotted_and_call_syntax():
    assert plan("what does getUserProfile() return").symbols == ["getUserProfile"]
    assert plan("see UserService.findById").symbols == ["findById"]
    assert plan("how does `build_context` work").symbols == ["build_context"]


def test_file_names_are_not_symbols():
    assert plan("why does agent.py fail").symbols == []


def test_plain_words_become_the_search_terms_without_question_noise():
    p = plan("where is the user login handled?")
    assert p.queries[0].split("|") == ["user", "login"]
    assert "handled" not in p.queries[0] and "where" not in p.queries[0]


def test_intents_select_extra_tools():
    p = plan("how does the app start up")
    assert p.entry_points and "startup" in p.intents
    r = plan("what happens when I call the /api/login endpoint")
    assert r.route_queries == ["/api/login"]
    c = plan("who calls authenticate_user")
    assert c.callers_of == ["authenticate_user"]
    t = plan("which tests cover authenticate_user")
    assert t.tests_for == ["authenticate_user"] and t.wants_tests


def test_vague_questions_produce_an_empty_plan():
    assert plan("check my code").is_empty
    assert plan("what is this").is_empty


def test_limits_hold_for_very_long_questions():
    p = plan(" ".join(f"word{i}x_y" for i in range(40)))
    assert len(p.symbols) <= planner.MAX_IDENTIFIERS and len(p.queries) <= planner.MAX_QUERIES


@pytest.mark.parametrize("text", ["", "   ", "???", "a b", "the and of"])
def test_degenerate_input_never_crashes(text):
    assert planner.make_plan(text).is_empty

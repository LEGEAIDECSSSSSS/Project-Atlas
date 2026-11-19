"""Regression tests for the latest real Qwen run (Atlas_exact_response_regression).

That run showed no corrections at all. Root causes, each pinned here:
  1. The prompt said "the difference between ..."; a general-knowledge guard classified it as a conceptual
     question, so Atlas never investigated: no tools, no packet, no finalize step. The model then copied its
     previous answer from the conversation history, wrong claims included.
  2. Once investigated, the depth of the trace depended on whether the prompt repeated each helper's wording.
  3. The verifier's connection checks keyed on the label `Connection:`, so `Connections:` skipped them.

All agent-driving tests use a scripted stand-in for the model (NOT Qwen); the real answer is an INPUT fixture.
"""
from pathlib import Path

import pytest

from agent import agent as A
from agent import finalize as fz
from tests.test_call_analysis import CONFIRMED, set_window  # noqa: F401
from tests.test_flow_reliability import (  # noqa: F401
    ROOT, assert_packet_honest, entry, labels, oracle, run_agent, taskboard, uploader, verdict, withheld)

FIX = Path(__file__).parent / "fixtures"
LATEST_PROMPT = (FIX / "prompt_latest_user.txt").read_text().strip()
LATEST_ANSWER = (FIX / "qwen_login_trace_answer_latest_user.txt").read_text()


def short(text):
    return text.replace(ROOT, "")


def packet_of(calls):
    return calls[0][-1]["content"]


# ----------------------------------------------------------------------------- 1. classification
def test_the_latest_prompt_is_investigated_not_treated_as_a_conceptual_question(taskboard):
    paths = A.listing_paths(A.project_service.list_project_files(taskboard)) if hasattr(A, "project_service") else []
    from services import project_service as ps
    paths = A.listing_paths(ps.list_project_files(taskboard))
    requirement = A.classify_request(LATEST_PROMPT, True, paths, "code")
    assert requirement is not None and requirement.kind == "explore"


@pytest.mark.parametrize("message", [
    "What is the difference between files and folders?",
    "Explain the difference between a function and a method in Python",
    "What are the pros and cons of microservices?",
    "Explain the difference between TCP and UDP",
])
def test_general_knowledge_questions_still_skip_the_investigation(taskboard, message):
    from services import project_service as ps
    paths = A.listing_paths(ps.list_project_files(taskboard))
    assert A.classify_request(message, True, paths, "code") is None


@pytest.mark.parametrize("message", [
    "Explain the difference between the service return and the route return, with file paths and line numbers.",
    "What is the difference between login and logout in this project?",
    "Explain the difference between what the frontend function and the backend route handler return, for every relevant function.",
])
def test_project_evidence_cues_override_the_general_knowledge_guard(taskboard, message):
    from services import project_service as ps
    paths = A.listing_paths(ps.list_project_files(taskboard))
    assert A.classify_request(message, True, paths, "code") is not None


# ----------------------------------------------------------------------------- 2. depth independent of wording
def _defs_and_gaps(packet):
    defs, gaps = assert_packet_honest(packet)
    return {(n, p.replace(ROOT, "")): (int(a), int(b)) for n, p, a, b in defs}, {g[0] for g in gaps}


def test_the_latest_prompt_traces_the_same_functions_as_the_original_prompt(monkeypatch, taskboard):
    from tests.test_flow_reliability import FLOW_PROMPT
    _, calls_old = run_agent(monkeypatch, taskboard, FLOW_PROMPT, oracle)
    events, calls_new = run_agent(monkeypatch, taskboard, LATEST_PROMPT, oracle)
    old, old_gaps = _defs_and_gaps(packet_of(calls_old))
    new, new_gaps = _defs_and_gaps(packet_of(calls_new))
    assert new == old == CONFIRMED                       # all 16, frontend API client included
    assert new_gaps == old_gaps                          # the same plumbing is left out, whatever the wording
    assert any(label.startswith("Traced route") for label in labels(events))


def test_the_api_client_is_read_even_when_the_prompt_never_mentions_a_request(monkeypatch, taskboard):
    _, calls = run_agent(monkeypatch, taskboard, LATEST_PROMPT, oracle)
    defs, _ = _defs_and_gaps(packet_of(calls))
    assert ("request", "frontend/src/api.js") in defs and ("api.post", "frontend/src/api.js") in defs
    assert ("find_by_email", "app/repositories/user_repository.py") not in defs      # (qualified name is used)
    assert ("UserRepository.find_by_email", "app/repositories/user_repository.py") in defs


def _node(index, symbol, path, start, end, called_from=None):
    return {"kind": "function", "symbol": symbol, "path": path, "start_line": start, "end_line": end,
            "called_from": called_from}


def _route():
    chain = [_node(0, "login", "c.py", 10, 20), _node(1, "local_helper", "c.py", 30, 35, "c.py:12"),
             _node(2, "Service.login", "s.py", 1, 30, "c.py:15"), _node(3, "Repo.find", "r.py", 1, 5, "s.py:5"),
             _node(4, "Repo.make", "r.py", 7, 9, "s.py:20"), _node(5, "token", "t.py", 1, 3, "r.py:8"),
             _node(6, "utcnow", "time.py", 1, 2, "s.py:22"), _node(7, "Limiter.hit", "rl.py", 1, 4, "c.py:11"),
             _node(8, "Limiter.reset", "rl.py", 6, 8, "c.py:18"), _node(9, "deep", "d.py", 1, 2, "t.py:2")]
    return {"chain": chain}


def _ranked(route, relevant):
    return [(1 if i in relevant else 0, i, n) for i, n in enumerate(route["chain"])]


def test_flow_node_selection_is_structural_not_lexical():
    route = _route()
    chosen = {i for _r, i, _n in A._select_flow_nodes(route, _ranked(route, relevant={0, 2}))}
    assert chosen == {0, 1, 2, 3, 4, 5}      # handler, its module-local helper, service callees to depth 2
    # excluded: a single-function leaf module (utcnow), the handler's cross-module guards, a depth-3 callee
    assert not chosen & {6, 7, 8, 9}


def test_wording_that_names_a_helper_includes_it():
    route = _route()
    chosen = {i for _r, i, _n in A._select_flow_nodes(route, _ranked(route, relevant={0, 2, 7, 8}))}
    assert {7, 8} <= chosen                  # the question named the rate limiter: it is read


def test_a_question_that_matches_nothing_still_reads_the_call_graph_minus_plumbing():
    route = _route()
    chosen = {i for _r, i, _n in A._select_flow_nodes(route, _ranked(route, relevant=set()))}
    assert 6 not in chosen and {0, 2, 3} <= chosen


# ----------------------------------------------------------------------------- 3. the latest real answer
@pytest.fixture
def replay(monkeypatch, taskboard):
    events, calls = run_agent(monkeypatch, taskboard, LATEST_PROMPT, LATEST_ANSWER)
    out = verdict(events)
    return out, short(out[len(LATEST_ANSWER):])


def test_the_latest_real_answer_is_accepted_kept_verbatim_and_challenged(replay):
    out, add = replay
    assert not out.startswith("Atlas could not") and out.startswith(LATEST_ANSWER)
    assert fz.CORRECTIONS_HEADING in add, "the corrections section must appear"


def test_each_error_in_the_review_is_challenged_with_the_source_value(replay):
    _, add = replay
    corrections = add[add.index(fz.CORRECTIONS_HEADING):add.index(fz.HEADING)]
    assert "the source calls `login(normaliseEmail(email), password)` at line 36" in corrections
    assert "`service.login(...)` (line 89)" in corrections and "is called at line 69 of controllers/auth_controller.py" in corrections
    assert "`login` returns `user_out(user)` (line 72)" in corrections
    assert "`access` is passed as argument 2 of `set_auth_cookies(...)` (line 71)" in corrections
    assert "line 44 is where `SessionRepository.create` is defined" in corrections
    assert "the call `self.sessions.create(...)` is at line 125" in corrections
    assert "receiver `self.sessions` is a `SessionRepository`" in corrections
    assert "utcnow() + timedelta(days=settings.REFRESH_TOKEN_DAYS)" in corrections          # the real arguments


def test_correct_claims_in_the_latest_answer_are_not_flagged(replay):
    _, add = replay
    corrections = add[add.index(fz.CORRECTIONS_HEADING):add.index(fz.HEADING)]
    for symbol in ("_client_ip", "create_access_token", "generate_token", "verify_password",
                   "UserRepository.find_by_email", "request", "api.post", "AuthService.login"):
        assert f"- **`{symbol}`**" not in corrections or f"- **`{symbol}`**" in corrections and "contradicted" not in \
            "".join(l for l in corrections.splitlines() if l.startswith(f"- **`{symbol}`**")), symbol


def test_the_missing_connections_in_the_review_are_supplied(replay):
    _, add = replay
    verified = add[add.index(fz.HEADING):]
    login_backend = verified[verified.index("- **`AuthService.login`**"):]
    login_backend = login_backend[:login_backend.index("\n- **", 5)]
    assert "`self._start_session(user, user_agent)` at line 92" in login_backend
    assert "`create_access_token(user.id)` at line 94" in login_backend and "`verify_password(...)` at line 86" in login_backend
    start = verified[verified.index("- **`AuthService._start_session`**"):]
    assert "`generate_token()` at line 124" in start[:start.index("\n- **", 5)]


def test_the_frontend_login_return_value_and_context_reference_are_stated(replay):
    _, add = replay
    verified = add[add.index(fz.HEADING):]
    block = verified[verified.index("- **`login`** (`frontend/src/auth.jsx:38-44`)"):]
    block = block[:block.index("\n- **", 5)]
    assert "Returns `me` (line 43), the result of `api.post(...)` (line 40)" in block
    assert "`me` is also passed to `queryClient.setQueryData(...)` (line 42)" in block
    assert "Also referenced (not called) at line 57" in block and "login, logout" in block


def test_the_returned_variable_fact_is_not_added_when_the_answer_states_it_and_skips_error_classes():
    flow = ["line 40 `me = api.post(...)` receives its value from `api.post`",
            "`me` is passed to `queryClient.setQueryData(...)` (line 42)", "`me` is passed to `new ApiError(...)` (line 61)"]
    item = {"symbol": "login", "path": "a.jsx", "start": 38, "end": 44, "calls": [], "called_by": [], "flow": flow,
            "params": [], "returns": [(43, ["me"])], "references": []}
    got = fz._facts_for(item, "data flow: returns the response.", set())
    assert [s for s, _ in got] == ["Returns `me` (line 43), the result of `api.post(...)` (line 40); "
                                   "`me` is also passed to `queryClient.setQueryData(...)` (line 42)."]
    assert fz._facts_for(item, "data flow: returns the result of api.post to the caller.", set()) == []


# ----------------------------------------------------------------------------- 4. history echo
def test_atlas_sections_are_not_fed_back_to_the_model_as_conversation():
    previous = "Function: x ...model text...\n\n---\n### Atlas corrections to the answer above\n- **`x`** wrong\n\n---\n### Atlas-verified connections\n- more"
    cleaned = A.sanitize_history([{"role": "user", "content": "q"}, {"role": "assistant", "content": previous}])
    assert cleaned[1]["content"] == "Function: x ...model text..."
    assert A.strip_atlas_sections("plain answer") == "plain answer"


def test_a_previous_finalized_answer_does_not_reach_the_model_prompt(monkeypatch, taskboard):
    events, _ = run_agent(monkeypatch, taskboard, LATEST_PROMPT, LATEST_ANSWER)
    finalized = verdict(events)
    history = [{"role": "user", "content": LATEST_PROMPT}, {"role": "assistant", "content": finalized}]
    _, calls = run_agent(monkeypatch, taskboard, LATEST_PROMPT, oracle, history=history)
    sent = "\n".join(m["content"] for call in calls for m in call)
    assert "Atlas-verified connections" not in sent and "Atlas corrections" not in sent
    assert "Not covered by the model's answer" not in sent


# ----------------------------------------------------------------------------- 5. verifier labels
@pytest.mark.parametrize("label", ["Connection", "Connections", "**Connection:**", "**Connections:**"])
def test_a_fabricated_call_is_rejected_whatever_the_connection_label(monkeypatch, taskboard, label):
    sep = "" if label.endswith(":**") else ":"
    answer = (f"Function: create_access_token File Path: {ROOT}app/core/security.py Line Numbers: 37-45 Purpose: x "
              f"{label}{sep} Calls generate_token which is a project function. Data Flow: y\n")
    events, _ = run_agent(monkeypatch, taskboard, LATEST_PROMPT, answer)
    assert withheld(events) and "does not call `generate_token`" in verdict(events)


def test_a_called_by_statement_quoting_the_callers_expression_is_not_an_outgoing_call(monkeypatch, taskboard):
    answer = (f"Function: set_auth_cookies File Path: {ROOT}app/controllers/auth_controller.py Line Numbers: 23-28 "
              "Purpose: x Connections: Called by `login` (controllers/auth_controller.py:65-72) at line 71 as "
              "`set_auth_cookies(response, access, refresh)`. Data Flow: y\n")
    events, _ = run_agent(monkeypatch, taskboard, LATEST_PROMPT, answer)
    assert not withheld(events), verdict(events)

"""Completing a verified flow answer, and checking connection claims written as plain prose.

The fixture tests/fixtures/qwen_login_trace_answer.txt is a REAL output of Qwen2.5-Coder 7B on the Mini
Trello login trace. The tests that feed it through the pipeline use a scripted stand-in that returns that
text, so they exercise what Atlas does with a real answer, not what Qwen would do.
"""
from pathlib import Path

import pytest

from agent import finalize as fz
from agent import verify as vf
from tests.test_flow_reliability import FLOW_PROMPT, ROOT, run_agent, taskboard, uploader, verdict, withheld  # noqa: F401

REAL = (Path(__file__).parent / "fixtures" / "qwen_login_trace_answer.txt").read_text()
FABRICATED = ("Connection: Calls generate_token which is a project function. Data Flow: Receives user_id, "
              "calls generate_token, and returns the access token.")
CORRECTED = ("Connection: Calls utcnow(), str(user_id), timedelta(...) and jwt.encode(...), which are a project "
             "function and external libraries. Data Flow: Receives user_id and returns the signed access token.")


def short(text):
    return text.replace(ROOT, "")


# ----------------------------------------------------------------------------- prose connection claims
@pytest.mark.parametrize("text,expected", [
    ("Calls generate_token, SessionRepository.create, and hash_token which are project functions.",
     ["generate_token", "SessionRepository.create", "hash_token"]),
    ("Calls login(normaliseEmail(email), password) which is a project function.", ["login"]),
    ("Receives request, calls find_by_email and verify_password, and returns the user, access token.",
     ["find_by_email", "verify_password"]),
    ("Calls `self.users.find_by_email(request.email)` and verify_password(request.password, x if y else None).",
     ["self.users.find_by_email", "verify_password"]),
    ("It does not call hash_password. Never calls other_thing.", []),
    ("Calls the AuthService.login method.", ["AuthService.login"]),
])
def test_prose_calls_are_extracted_with_arguments_and_trailing_clauses_removed(text, expected):
    assert vf._prose_called_names(text) == expected


def test_a_real_answer_that_invents_a_call_is_rejected_for_that_claim_only(monkeypatch, taskboard):
    events, _ = run_agent(monkeypatch, taskboard, FLOW_PROMPT, REAL)
    assert withheld(events)
    notes = [l for l in verdict(events).splitlines() if l.startswith("- ")]
    assert len(notes) == 1, notes                       # the other 13 entries raise no false alarms
    assert "create_access_token" in notes[0] and "does not call `generate_token`" in notes[0]


def test_plain_labels_without_bold_are_checked_like_bold_ones(monkeypatch, taskboard):
    answer = ("Function: verify_password File Path: " + ROOT + "app/core/security.py Line Numbers: 21-25 "
              "Purpose: x Connection: Calls hash_password which is a project function.\n")
    events, _ = run_agent(monkeypatch, taskboard, FLOW_PROMPT, answer)
    assert withheld(events) and "does not call `hash_password`" in verdict(events)


def test_a_downstream_call_attributed_to_its_caller_is_rejected(monkeypatch, taskboard):
    wrong = REAL.replace("Calls service.login(body, request.headers.get(\"user-agent\")) which is a project function.",
                         "Calls service.login(body) which then calls verify_password.").replace(FABRICATED, CORRECTED)
    events, _ = run_agent(monkeypatch, taskboard, FLOW_PROMPT, wrong)
    assert withheld(events) and "does not call `verify_password`" in verdict(events)


# ----------------------------------------------------------------------------- completing the answer
@pytest.fixture
def completed(monkeypatch, taskboard):
    answer = REAL.replace(FABRICATED, CORRECTED)
    assert answer != REAL
    events, _ = run_agent(monkeypatch, taskboard, FLOW_PROMPT, answer)
    out = verdict(events)
    return answer, out, short(out[len(answer):])


def test_the_models_text_is_never_altered_and_still_passes_verification(completed):
    answer, out, addendum = completed
    assert not out.startswith("Atlas could not") and out.startswith(answer) and addendum.strip()


def test_leaf_functions_get_their_callers(completed):
    _, _, add = completed
    assert "Called by `login` (controllers/auth_controller.py:65-72) at line 67 as `_client_ip(request)`" in add
    assert ("Called by `AuthService.login` (services/auth_service.py:84-94) at line 85 as "
            "`self.users.find_by_email(request.email)`; receiver `self.users` is a `UserRepository`") in add
    assert "at line 86 as `verify_password(...)`" in add
    for line, expr in ((124, "generate_token()"), (127, "hash_token(token)"), (125, "self.sessions.create(...)")):
        assert f"Called by `AuthService._start_session` (services/auth_service.py:123-131) at line {line} as `{expr}`" in add


def test_known_dependencies_are_reported_as_such_not_as_unknown(completed):
    _, _, add = completed
    assert "`_hasher.verify(...)` at line 23 -> external library `argon2`, not a project function" in add
    assert "`secrets.token_urlsafe(32)`" in add and "`hashlib.sha256(...).hexdigest()`" in add
    assert "`self.session.exec(...).first()`" in add and "sqlmodel" in add


def test_the_written_call_and_its_receiver_type_are_shown(completed):
    _, _, add = completed
    assert ("Call as written: `self.sessions.create(...)` at line 125 -> project function `SessionRepository.create` "
            "(repositories/user_repository.py:44-47); receiver `self.sessions` is a `SessionRepository`") in add


def test_recursion_and_conditions_are_added_when_the_answer_missed_them(completed):
    _, _, add = completed
    assert "`refreshSession()` at line 49 -> project function `refreshSession`" in add
    assert "conditional (in `if` block from line 48)" in add
    assert "`request(method, path, body, false)` at line 49: RECURSIVE call to itself" in add


def test_controller_data_flow_explains_the_tokens(completed):
    _, _, add = completed
    assert "line 69 `(user, access, refresh) = service.login(...)` receives 3 values from `AuthService.login`" in add
    assert "`access` = `create_access_token(user.id)`" in add
    assert "`access` is passed as argument 2 of `set_auth_cookies(...)` (line 71)" in add
    assert "`refresh` is passed as argument 3 of `set_auth_cookies(...)` (line 71)" in add
    assert "`user` is passed as argument 1 of `user_out(...)` (line 72)" in add


def test_skipped_definitions_are_restored_with_accurate_connections(completed):
    _, _, add = completed
    section = add[add.index("### Not covered"):]
    assert "**Function:** set_auth_cookies" in section and "**Line Numbers:** 23-28" in section
    assert "**Function:** user_out" in section and "**Line Numbers:** 36-43" in section
    assert "Called by `login`" in section and "`response.set_cookie(...)` at line 25 -> external library `fastapi`" in section
    assert "Calls `UserOut(...)` at line 37 -> project class `UserOut`" in section
    assert "No calls" not in section                    # never claims "no calls" when calls exist


def test_the_completion_stays_focused(completed):
    _, _, add = completed
    assert len(add) < 6000
    for noise in ("setEmailError", "setBusy", "navigate(", "ApiError", "validateEmail", "utcnow()", "str(user_id)",
                  "timedelta"):
        assert noise not in add, noise                  # helpers outside the trace, hook setters, error classes


def test_the_frontend_chain_is_untouched(completed):
    answer, out, _ = completed
    for name, lines in (("submit", "27-44"), ("api.post", "72-72")):
        assert f"Function: {name} File Path: {ROOT}frontend/src/" in out and lines in out


# ----------------------------------------------------------------------------- what is NOT added
def _item(**over):
    base = {"number": 1, "symbol": "f", "path": "a.py", "start": 1, "end": 5, "called_by": [], "calls": [],
            "flow": [], "params": [], "signature": "def f():"}
    base.update(over)
    return base


def _call(**over):
    base = {"line": 3, "expr": "g(x)", "callee": "g", "kind": "project", "targets": [
        {"path": "a.py", "symbol": "g", "start": 10, "end": 12, "kind": "function"}], "note": "",
        "conditional": None, "receiver": "", "receiver_note": "", "hook_value": False}
    base.update(over)
    return base


KEYS = {("a.py", 10, 12)}


def test_nothing_is_added_for_facts_the_answer_already_states():
    cb = {"caller": "main", "caller_no": 2, "line": 9, "expr": "f()", "receiver_note": "", "path": "a.py", "start": 20, "end": 30}
    item = _item(called_by=[cb], calls=[_call()])
    text = "connection: called by main; calls g(x) which is a project function."
    assert fz._facts_for(item, text, KEYS) == []


def test_callers_are_added_only_for_entries_the_model_could_not_connect():
    cb = {"caller": "main", "caller_no": 2, "line": 9, "expr": "f()", "receiver_note": "", "path": "a.py", "start": 20, "end": 30}
    item = _item(called_by=[cb])
    assert fz._facts_for(item, "connection: calls g.", KEYS) == []
    got = fz._facts_for(item, "connection: cannot be established.", KEYS)
    assert got and "Called by `main`" in got[0][0]


def test_hook_setters_helpers_outside_the_trace_and_constructors_are_not_added():
    calls = [_call(callee="setBusy", expr="setBusy(true)", kind="external", targets=[], note="react", hook_value=True),
             _call(callee="util", expr="util()", targets=[{"path": "u.py", "symbol": "util", "start": 1, "end": 2, "kind": "function"}]),
             _call(callee="Err", expr="new Err()", targets=[{"path": "u.py", "symbol": "Err", "start": 5, "end": 6, "kind": "class"}],
                   conditional="in `if` block from line 2")]
    assert fz._facts_for(_item(calls=calls), "connection: cannot be established.", KEYS) == []
    again = fz._facts_for(_item(calls=calls), "", KEYS, nothing_said=True)         # a skipped definition: say it all
    assert any("project class `Err`" in s for s, _ in again)


def test_a_conditional_project_function_call_is_added_unless_already_described():
    call = _call(callee="refresh", expr="refresh()", conditional="in `if` block from line 2",
                 targets=[{"path": "a.py", "symbol": "refresh", "start": 40, "end": 41, "kind": "function"}])
    assert fz._facts_for(_item(calls=[call]), "connection: calls fetch.", KEYS)
    assert not fz._facts_for(_item(calls=[call]), "connection: conditionally calls refresh when the token expired.", KEYS)


def test_answers_that_are_not_per_function_entries_are_left_alone(taskboard):
    assert fz.build_addendum("The login starts in the form and goes to the server.", [{"symbol": "x"}], taskboard, {}) == ""
    assert fz.build_addendum("", [], taskboard, {}) == ""


# ============================================================================ challenging the model's claims
REAL2 = (Path(__file__).parent / "fixtures" / "qwen_login_trace_answer_2.txt").read_text()


@pytest.fixture
def challenged(monkeypatch, taskboard):
    events, _ = run_agent(monkeypatch, taskboard, FLOW_PROMPT, REAL2)
    out = verdict(events)
    text = short(out[len(REAL2):])
    section = text[text.index(fz.CORRECTIONS_HEADING):text.index(fz.HEADING)]
    items = [l for l in section.splitlines() if l.startswith("- **")]
    return out, text, items


def _item_for(items, symbol, needle):
    hits = [i for i in items if i.startswith(f"- **`{symbol}`**") and needle in i]
    assert len(hits) == 1, (symbol, needle, hits)
    return hits[0]


def test_the_second_real_answer_is_kept_verbatim_and_not_withheld(challenged):
    out, text, items = challenged
    assert out.startswith(REAL2) and not out.startswith("Atlas could not")


def test_a_wrong_call_site_line_is_contradicted_with_the_real_one(challenged):
    _, _, items = challenged
    hit = _item_for(items, "login", "(line 89)")
    assert "**contradicted:**" in hit and "is called at line 69 of controllers/auth_controller.py" in hit
    assert "line 89 is inside `AuthService.login` (services/auth_service.py:84-94)" in hit
    hit = _item_for(items, "submit", "(line 40)")
    assert "is called at line 36 of pages/Login.jsx" in hit and "line 40 is inside `login` (src/auth.jsx:38-44)" in hit


def test_invocation_and_definition_locations_are_kept_distinct(challenged):
    _, _, items = challenged
    hit = _item_for(items, "AuthService._start_session", "(line 44)")
    assert "**clarified:**" in hit
    assert "line 44 is where `SessionRepository.create` is defined" in hit
    assert "the call `self.sessions.create(...)` is at line 125 of services/auth_service.py" in hit
    assert "receiver `self.sessions` is a `SessionRepository`" in hit
    hit = _item_for(items, "AuthService.login", "(line 16)")
    assert "line 16 is where `UserRepository.find_by_email` is defined" in hit and "is at line 85" in hit


def test_wrong_call_arguments_are_contradicted_with_the_source_arguments(challenged):
    _, _, items = challenged
    hit = _item_for(items, "submit", "`login(email, password)`")
    assert "the source calls `login(normaliseEmail(email), password)` at line 36" in hit
    hit = _item_for(items, "AuthService._start_session", "`self.sessions.create(user, user_agent)`")
    assert ("self.sessions.create(user.id, hash_token(token), utcnow() + timedelta(days=settings.REFRESH_TOKEN_DAYS), "
            "(user_agent or \"\")[:255] or None)") in hit and "at line 125" in hit


def test_a_service_return_is_not_mistaken_for_the_route_response(challenged):
    _, _, items = challenged
    hit = _item_for(items, "login", "returns `access`, `refresh`")
    assert "**contradicted:**" in hit
    assert "`login` returns `user_out(user)` (line 72)" in hit
    assert "`access` is passed as argument 2 of `set_auth_cookies(...)` (line 71)" in hit
    assert hit.endswith(".") and not hit.endswith("..") and ").." not in hit      # no doubled punctuation


def test_correct_claims_are_never_flagged(challenged):
    _, _, items = challenged
    flagged = {(i.split("`")[1], i.split("answer says ")[1][:40]) for i in items}
    assert len(items) == 9, "\n".join(items)
    untouched = ("_client_ip", "create_access_token", "generate_token", "verify_password", "UserRepository.find_by_email",
                 "request", "api.post")
    for symbol in untouched:
        bad = [i for i in items if i.startswith(f"- **`{symbol}`**") and "**contradicted:**" in i]
        assert not bad, bad                                          # `fetch(...)`'s abbreviated object is not an error
    assert not [i for i in items if i.startswith("- **`AuthService.login`**") and "contradicted" in i]


def test_incomplete_connections_are_completed_without_listing_every_helper(challenged):
    _, text, _ = challenged
    text = text[text.index(fz.HEADING):]                       # the verified-connections section, not the corrections
    block = text[text.index("- **`AuthService.login`** (`app/services/auth_service.py:84-94`)"):]
    block = block[:block.index("\n- **", 10) if "\n- **" in block[10:] else len(block)]
    assert "`verify_password(...)` at line 86" in block and "`self._start_session(user, user_agent)` at line 92" in block
    assert "`create_access_token(user.id)` at line 94" in block
    assert "utcnow" not in block and "ForbiddenError" not in block and "self.session.add" not in block


def test_a_wrong_line_in_prose_is_flagged_but_does_not_discard_the_answer(monkeypatch, taskboard):
    events, _ = run_agent(monkeypatch, taskboard, FLOW_PROMPT, REAL2)
    assert not withheld(events)                                     # corrections are not blocking
    events, _ = run_agent(monkeypatch, taskboard, FLOW_PROMPT, REAL.replace(FABRICATED, CORRECTED))
    assert not withheld(events) and fz.CORRECTIONS_HEADING not in verdict(events)    # nothing wrong -> no section


def test_the_first_real_answer_gets_no_false_corrections(monkeypatch, taskboard):
    events, _ = run_agent(monkeypatch, taskboard, FLOW_PROMPT, REAL.replace(FABRICATED, CORRECTED))
    assert fz.CORRECTIONS_HEADING not in verdict(events)


# ----------------------------------------------------------------------------- unit: how claims are judged
@pytest.mark.parametrize("claimed,actual,conflict", [
    ("email, password", "normaliseEmail(email), password", True),                       # an argument was simplified away
    ("user, user_agent", "user.id, hash_token(token), x, y", True),                     # wrong number of arguments
    ("'POST', path, body ?? {}", "'POST', path, body ?? {}", False),
    ('"POST", path', "'POST', path", False),                                            # quote style is irrelevant
    ("a, b", "a,\n b,", False),                                                         # trailing comma / whitespace
    ("BASE + path, { method, credentials: 'include', headers, body }",
     "BASE + path, { method, credentials: 'include', headers: cond ? {x: 1} : undefined, body: cond ? f(body) : undefined, }", False),
    ("BASE + path, { method, headers }", "BASE + path, { method, credentials: 'include', headers }", True),   # a key is missing
])
def test_call_arguments_are_compared_structurally(claimed, actual, conflict):
    assert fz._args_conflict(claimed, actual) is conflict


def _call_item(**over):
    call = {"line": 36, "expr": "login(normaliseEmail(email), password)", "callee": "login", "kind": "project",
            "targets": [{"path": "web/auth.jsx", "symbol": "login", "start": 38, "end": 44, "kind": "function"}],
            "note": "", "conditional": None, "receiver": "", "receiver_note": "", "hook_value": False,
            "args": "normaliseEmail(email), password"}
    call.update(over)
    return {"symbol": "submit", "path": "web/Login.jsx", "start": 27, "end": 44, "calls": [call], "called_by": [],
            "returns": [], "locals": [], "flow": []}


def test_a_cited_line_is_ok_at_the_call_clarified_at_the_definition_and_contradicted_elsewhere():
    ok = fz._challenge(_call_item(), "Connection: Calls login(...) from web/auth.jsx (line 36).")
    assert ok == []
    clarified = fz._challenge(_call_item(), "Connection: Calls login(...) from web/auth.jsx (line 38).")
    assert [k for k, *_ in clarified] == ["clarified"]
    wrong = fz._challenge(_call_item(), "Connection: Calls login(...) at line 99.")
    assert [k for k, *_ in wrong] == ["contradicted"] and "neither where it is called nor where it is defined" in wrong[0][2]


def test_claims_about_calls_the_analysis_does_not_know_are_left_alone():
    assert fz._challenge(_call_item(), "Connection: Calls fetchSomething(x) at line 99.") == []
    assert fz._challenge(_call_item(), "Purpose: x. Data Flow: Receives email.") == []


def test_called_by_is_checked_only_when_callers_are_known():
    item = _call_item()
    item["called_by"] = [{"caller": "main", "caller_no": 2, "line": 9, "expr": "submit()", "receiver_note": "",
                          "path": "web/App.jsx", "start": 1, "end": 20}]
    assert fz._challenge(item, "Connection: Called by main to start it.") == []
    wrong = fz._challenge(item, "Connection: Called by render to start it.")
    assert wrong and "the callers found in the source are `main` (line 9)" in wrong[0][2]
    item["called_by"] = []
    assert fz._challenge(item, "Connection: Called by render.") == []                 # unknown callers: no claim made


def test_a_returned_value_claim_is_checked_only_against_local_names():
    item = _call_item()
    item.update(returns=[(72, ["user_out(user)"])], locals=["body", "user", "access", "refresh"],
                flow=["`access` is passed as argument 2 of `set_auth_cookies(...)` (line 71)"])
    bad = fz._challenge(item, "Connection: x. Data Flow: returns user, access token, and refresh token.")
    assert bad and bad[0][1] == "returns `access`, `refresh`"
    fine = fz._challenge(item, "Connection: x. Data Flow: returns the response from the server and the user.")
    assert fine == []                                                                  # descriptive wording is not a claim about a local


def test_wrappers_the_answer_describes_only_by_their_callers_get_their_external_dependency(challenged):
    _, text, _ = challenged
    verified = text[text.index(fz.HEADING):]
    def bullet(symbol):
        start = verified.index(f"- **`{symbol}`**")
        end = verified.find("\n- **", start + 5)
        return verified[start:end if end > 0 else len(verified)]
    assert "`_hasher.verify(...)` at line 23 -> external library `argon2`, not a project function" in bullet("verify_password")
    assert "`secrets.token_urlsafe(32)`" in bullet("generate_token")
    assert "hashlib" in verified[verified.index("**Function:** hash_token"):]    # restored entry, same rule
    assert "jwt" in bullet("create_access_token")


def test_an_entry_that_already_lists_its_calls_gets_no_extra_dependencies():
    item = _item(calls=[_call(callee="jwt", expr="jwt.encode(x)", kind="external", targets=[], note="jwt", receiver="jwt")])
    assert fz._facts_for(item, "connection: calls jwt.encode to sign the token.", KEYS) == []
    assert fz._facts_for(item, "connection: called by login to build the token.", KEYS)



def test_markdown_backticked_call_expressions_are_checked_for_wrong_call_sites(monkeypatch, taskboard):
    """Match the formatting used by the real Qwen response, not only plain-text fixtures."""
    answer = REAL2
    for label in ("Function", "File Path", "Line Numbers", "Purpose", "Connection", "Data Flow"):
        answer = answer.replace(f"{label}:", f"**{label}:**")
    answer = answer.replace("**Connection:**", "**Connections:**")
    answer = answer.replace("login(email, password)", "`login(email, password)`")
    answer = answer.replace('service.login(body, request.headers.get("user-agent"))',
                            '`service.login(body, request.headers.get("user-agent"))`')
    answer = answer.replace("self.sessions.create(user, user_agent)", "`self.sessions.create(user, user_agent)`")

    events, _ = run_agent(monkeypatch, taskboard, FLOW_PROMPT, answer)
    out = verdict(events)
    assert not withheld(events)
    assert out.startswith(answer)
    assert fz.CORRECTIONS_HEADING in out
    corrections = out.split(fz.CORRECTIONS_HEADING, 1)[1].split(fz.HEADING, 1)[0]
    assert "`submit`" in corrections and "(line 40)" in corrections
    assert "the source calls `login(normaliseEmail(email), password)` at line 36" in corrections
    assert "`login`" in corrections and "(line 89)" in corrections
    assert "is called at line 69 of controllers/auth_controller.py" in corrections
    assert "`AuthService._start_session`" in corrections and "(line 44)" in corrections
    assert "the call `self.sessions.create(...)` is at line 125" in corrections
    assert "`login` returns `user_out(user)` (line 72)" in corrections
    assert len([line for line in corrections.splitlines() if line.startswith("- **`")]) == 9

def test_plural_bold_connections_label_still_challenges_the_second_real_answer(monkeypatch, taskboard):
    """Regression for Qwen's **Connections:** formatting in the latest real run."""
    answer = REAL2
    for label in ("Function", "File Path", "Line Numbers", "Purpose", "Connection", "Data Flow"):
        answer = answer.replace(f"{label}:", f"**{label}:**")
    answer = answer.replace("**Connection:**", "**Connections:**")

    events, _ = run_agent(monkeypatch, taskboard, FLOW_PROMPT, answer)
    out = verdict(events)
    assert not withheld(events), "corrections must remain non-blocking"
    assert out.startswith(answer), "the model's original answer must remain unchanged"
    assert fz.CORRECTIONS_HEADING in out
    corrections = out.split(fz.CORRECTIONS_HEADING, 1)[1].split(fz.HEADING, 1)[0]

    expected = (
        ("login", "(line 89)", "is called at line 69"),
        ("submit", "(line 40)", "is called at line 36"),
        ("AuthService._start_session", "(line 44)", "is at line 125"),
        ("submit", "`login(email, password)`", "login(normaliseEmail(email), password)"),
        ("AuthService._start_session", "`self.sessions.create(user, user_agent)`", "hash_token(token)"),
    )
    for symbol, claim, correction in expected:
        assert f"**`{symbol}`**" in corrections and claim in corrections and correction in corrections, corrections
    assert "`login` returns `user_out(user)` (line 72)" in corrections
    correction_items = [line for line in corrections.splitlines() if line.startswith("- **`")]
    assert len(correction_items) == 9, corrections
    for symbol in ("_client_ip", "create_access_token", "generate_token", "verify_password", "request"):
        assert not any(line.startswith(f"- **`{symbol}`**") and "**contradicted:**" in line
                       for line in correction_items), correction_items


def test_exact_latest_user_qwen_answer_gets_corrections_without_false_flags(monkeypatch, taskboard):
    """Replay the complete latest Qwen response, including bold labels and all helper entries."""
    latest = (Path(__file__).parent / "fixtures" / "qwen_login_trace_answer_latest_user.txt").read_text()
    events, _ = run_agent(monkeypatch, taskboard, FLOW_PROMPT, latest)
    out = verdict(events)
    assert not withheld(events), "corrections must remain non-blocking"
    assert out.startswith(latest), "the original answer must remain unchanged"
    assert fz.CORRECTIONS_HEADING in out
    corrections = out.split(fz.CORRECTIONS_HEADING, 1)[1].split(fz.HEADING, 1)[0]
    required = (
        ("submit", "login(email, password)", "login(normaliseEmail(email), password)"),
        ("login", "(line 89)", "is called at line 69"),
        ("AuthService.login", "line 16 is where", "is at line 85"),
        ("AuthService._start_session", "(line 44)", "is at line 125"),
        ("AuthService._start_session", "self.sessions.create(user, user_agent)", "self.sessions.create(...)"),
    )
    for symbol, claim, correction in required:
        assert f"**`{symbol}`**" in corrections and claim in corrections and correction in corrections, corrections
    assert "`login` returns `user_out(user)` (line 72)" in corrections
    # Correct entries must not be falsely marked as contradicted.
    items = [line for line in corrections.splitlines() if line.startswith("- **` ".strip())]
    for symbol in ("_client_ip", "verify_password", "create_access_token", "generate_token",
                   "set_auth_cookies", "user_out", "SessionRepository.create", "hash_token"):
        assert not any(line.startswith(f"- **`{symbol}`**") and "**contradicted:**" in line for line in items), items

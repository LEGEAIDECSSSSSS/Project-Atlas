"""Call classification, conditional/recursive detection, call expressions and data flow.

Generic projects prove nothing is tied to Mini Trello. The Mini Trello tests pin facts verified by
hand against the real source. As in test_flow_reliability.py, every agent-driving test uses a
recording stand-in for the model (NOT Qwen).
"""
import re

import pytest

import config
from agent import callflow as cf
from agent import packet as pk
from services import code_index

# helpers and fixtures shared with the main reliability tests
from tests.test_flow_reliability import (  # noqa: F401  (fixtures are used by name)
    FLOW_PROMPT, ROOT, assert_packet_honest, entry, labels, oracle, run_agent, taskboard, uploader,
    verdict, withheld)

PY = '''import hashlib
import requests
from sqlalchemy.orm import Session
from shop.util import normalise

_client = requests.Session()


class Repo:
    def __init__(self, session: Session):
        self.session = session

    def save(self, item):
        self.session.add(item)
        return item


def fetch_remote(url, retries=2):
    resp = _client.get(url)
    if resp.status_code == 200:
        return normalise(resp.text)
    return fetch_remote(url + "/", retries - 1)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def split_pair(body):
    return body[:2], body[2:]


def pipeline(url, repo, callback):
    body = fetch_remote(url)
    head, tail = split_pair(body)
    saved = repo.save(tail)
    verified = digest(body) if saved else None
    flag = saved and digest(tail)
    try:
        callback(saved)
    except ValueError:
        print("failed")
    return head, verified
'''
JS = '''import React, { useState } from 'react'
import { formatPrice } from './format'
import axios from 'axios'

export default function Cart({ items }) {
  const [busy, setBusy] = useState(false)
  async function pay(total) {
    setBusy(true)
    const res = await axios.post('/api/pay', { total })
    if (!res.ok) {
      console.error('failed')
      return pay(total)
    }
    const label = formatPrice(total)
    items.forEach((fn) => fn(label))
    return res.data
  }
  return null
}
'''
GENERIC = {"shop/store.py": PY, "shop/util.py": "def normalise(text):\n    return text.strip().lower()\n",
           "web/Cart.jsx": JS, "web/format.js": "export function formatPrice(n) {\n  return '$' + n.toFixed(2)\n}\n"}


@pytest.fixture
def generic(uploader):
    pid = uploader.files(GENERIC)
    facts = code_index.project_facts(pid)

    def classify(path, symbol):
        loc = next(i for i in facts["symbol_locations"][symbol.split(".")[-1]]
                   if i["path"] == path and i["qualname"] == symbol)
        definition = {"symbol": symbol, "path": path, "start": loc["start_line"], "end": loc["end_line"]}
        analyzer = cf.Analyzer(pid)
        calls = analyzer.classify(path, definition, {})
        return {(c["line"], c["callee"]): c for c in calls}, analyzer, definition, calls

    classify.pid = pid
    return classify


def set_window(monkeypatch, ctx):
    """Set the model window coherently, as LC_NUM_CTX would."""
    reserve = min(config.NUM_PREDICT, 2048, max(512, ctx // 3))
    monkeypatch.setattr(config, "NUM_CTX", ctx)
    monkeypatch.setattr(config, "OUTPUT_RESERVE_TOKENS", reserve)
    monkeypatch.setattr(config, "PROMPT_BUDGET_TOKENS", max(1024, ctx - reserve))
    monkeypatch.setattr(config, "MAX_FILE_CHARS", int(max(1024, ctx - reserve) * config.CHARS_PER_TOKEN * 0.30))


# ----------------------------------------------------------------------------- classification
def test_known_external_dependencies_are_never_reported_unknown(generic):
    calls, *_ = generic("shop/store.py", "fetch_remote")
    assert calls[(19, "get")]["kind"] == "external" and calls[(19, "get")]["note"] == "requests"
    digest, *_ = generic("shop/store.py", "digest")
    assert {c["note"] for c in digest.values() if c["callee"] in ("sha256", "hexdigest")} == {"hashlib stdlib"}
    save, *_ = generic("shop/store.py", "Repo.save")
    assert save[(14, "add")]["kind"] == "external" and "sqlalchemy.orm" in save[(14, "add")]["note"]
    pay, *_ = generic("web/Cart.jsx", "pay")
    assert pay[(9, "post")]["note"] == "axios" and pay[(8, "setBusy")]["note"] == "react"   # hook-returned setter


def test_platform_and_builtin_calls_are_not_project_functions(generic):
    pay, *_ = generic("web/Cart.jsx", "pay")
    assert pay[(11, "error")]["kind"] == "platform" and "console" in pay[(11, "error")]["note"]
    assert pay[(15, "forEach")]["kind"] == "platform"
    pipeline, *_ = generic("shop/store.py", "pipeline")
    assert pipeline[(42, "print")]["kind"] == "platform" and "built-in" in pipeline[(42, "print")]["note"]


def test_project_calls_keep_the_written_expression_apart_from_the_definition(generic):
    pipeline, *_ = generic("shop/store.py", "pipeline")
    call = pipeline[(36, "save")]
    assert call["expr"] == "repo.save(tail)" and call["kind"] == "project"
    assert [t["symbol"] for t in call["targets"]] == ["Repo.save"]


def test_recursive_calls_are_labelled_recursive(generic):
    py, *_ = generic("shop/store.py", "fetch_remote")
    assert py[(22, "fetch_remote")]["kind"] == "recursive"
    js, *_ = generic("web/Cart.jsx", "pay")
    assert js[(12, "pay")]["kind"] == "recursive" and "if" in js[(12, "pay")]["conditional"]


def test_conditional_calls_state_why(generic):
    py, *_ = generic("shop/store.py", "fetch_remote")
    assert py[(21, "normalise")]["conditional"] == "in `if` block from line 20"
    pipeline, *_ = generic("shop/store.py", "pipeline")
    assert "ternary" in pipeline[(37, "digest")]["conditional"]
    assert "short-circuit" in pipeline[(38, "digest")]["conditional"]
    assert "except" in pipeline[(42, "print")]["conditional"]
    assert pipeline[(34, "fetch_remote")]["conditional"] is None                # unconditional stays unflagged
    assert pipeline[(40, "callback")]["conditional"] is None                    # inside `try`, not conditional


def test_only_genuinely_unestablishable_calls_are_unknown_and_say_why(generic):
    pipeline, *_ = generic("shop/store.py", "pipeline")
    assert pipeline[(40, "callback")]["kind"] == "unknown"
    pay, *_ = generic("web/Cart.jsx", "pay")
    assert pay[(15, "fn")]["kind"] == "unknown" and "callback parameter" in pay[(15, "fn")]["note"]
    unknown = [c for table in (generic("shop/store.py", "pipeline")[0], pay) for c in table.values()
               if c["kind"] == "unknown"]
    assert {c["callee"] for c in unknown} == {"callback", "fn"}                 # nothing else is unknown


def test_describe_lists_a_chained_call_once_and_exposes_written_expressions(generic):
    _, analyzer, definition, _ = generic("shop/store.py", "digest")
    rows, expressions = cf.describe(generic.pid, [definition], [])
    external = next((t[0] if isinstance(t, list) else t) for rank, t in rows[1] if rank == cf.RANK_EXTERNAL)
    assert external.count("hashlib.sha256") == 1 and "hashlib stdlib" in external        # listed once, not twice
    pipeline = generic("shop/store.py", "pipeline")[2]
    _rows, expressions = cf.describe(generic.pid, [pipeline], [])
    assert expressions[("shop/store.py", 36, "save")] == "repo.save(tail)"


def test_data_flow_maps_values_by_position_and_follows_them(generic):
    _, analyzer, definition, calls = generic("shop/store.py", "pipeline")
    flow = analyzer.data_flow("shop/store.py", definition, calls)
    joined = "\n".join(flow)
    assert "`(head, tail) = split_pair(...)`" in joined.replace("split_pair(body)", "split_pair(...)") or \
        "receives 2 values from `split_pair`" in joined
    assert "`head` = `body[:2]`; `tail` = `body[2:]`" in joined
    assert "`tail` is passed as argument 1 of `repo.save(...)` (line 36)" in joined
    assert "(line 21) or `fetch_remote(url + '/', retries - 1)` (line 22)" in joined     # every return is named


def test_a_value_count_mismatch_is_not_mapped_by_position(uploader):
    pid = uploader.files({"m.py": "def two():\n    return 1, 2\n\n\ndef use():\n    a, b, c = two()\n    return a\n"})
    facts = code_index.project_facts(pid)
    loc = facts["symbol_locations"]["use"][0]
    definition = {"symbol": "use", "path": "m.py", "start": loc["start_line"], "end": loc["end_line"]}
    analyzer = cf.Analyzer(pid)
    flow = "\n".join(analyzer.data_flow("m.py", definition, analyzer.classify("m.py", definition, {})))
    assert "receives 3 values" in flow and "maps by position" not in flow


# ----------------------------------------------------------------------------- packet budget
def _toy_world():
    records = [{"tool": "read_project_file", "path": "a.py", "lines": "1-3", "complete": True,
                "source_text": "def one():\n    return two()\n"},
               {"tool": "read_project_file", "path": "a.py", "lines": "5-6", "complete": True,
                "source_text": "def two():\n    return 2"}]
    records[0]["lines"] = "1-2"
    route = {"chain": [{"kind": "function", "symbol": "one", "path": "a.py", "start_line": 1, "end_line": 2},
                       {"kind": "function", "symbol": "two", "path": "a.py", "start_line": 5, "end_line": 6,
                        "called_from": "a.py:2"}],
             "edges": [{"from": "one", "to": "two", "path": "a.py", "line": 2}],
             "frontend_chains": [], "frontend_ui_callers": []}
    return records, route


def test_analysis_never_pushes_required_source_out_of_the_packet():
    records, route = _toy_world()
    bulky = {1: [(cf.RANK_KEY, "x" * 400), (cf.RANK_EXTERNAL, "y" * 400)],
             2: [(cf.RANK_EXTERNAL, "z" * 400)]}
    for budget in (1200, 1600, 2000, 3000, 6000):
        text, report = pk.build_packet(records, route, budget, lambda v, e: (bulky, {}))
        if report["included"]:
            assert len(text) <= budget
            defs, _ = assert_packet_honest(text)
            assert len(defs) == 2                                # both required definitions still have source


def test_fit_drops_low_value_tiers_first_and_degrades_a_long_line_before_dropping_it():
    definitions = [{"symbol": "f"}, {"symbol": "g"}]
    rows = {1: [(cf.RANK_EXTERNAL, ["LONG " * 40, "short"])], 2: [(cf.RANK_UNKNOWN, "u" * 120)]}
    lines, omitted, _ = cf.fit(rows, definitions, 100)
    joined = "\n".join(lines)
    assert "short" in joined and "LONG" not in joined          # degraded, not dropped
    assert "u" * 120 not in joined and omitted == 1            # the least valuable tier went first


def test_fit_reports_when_an_edge_line_was_lost_so_call_sites_are_kept():
    definitions = [{"symbol": "f"}, {"symbol": "g"}]
    rows = {1: [(cf.RANK_VERIFIED, "calls definitions above: line 2 `g()` -> #2"),
                (cf.RANK_EXTERNAL, "NOT project functions: line 3 `os.getcwd()` [os stdlib]")]}
    _, _, lost = cf.fit(rows, definitions, 25)
    assert lost is True                                          # edge line did not survive
    _, _, lost = cf.fit(rows, definitions, 25, skip_ranks=(cf.RANK_VERIFIED,))
    assert lost is False                                         # edges are carried by the call-site list instead
    _, _, lost = cf.fit(rows, definitions, 500)
    assert lost is False


# ----------------------------------------------------------------------------- Mini Trello (real pipeline)
CONFIRMED = {                                    # verified by hand against the source archive
    ("submit", "frontend/src/pages/Login.jsx"): (27, 44), ("login", "frontend/src/auth.jsx"): (38, 44),
    ("api.post", "frontend/src/api.js"): (72, 72), ("request", "frontend/src/api.js"): (35, 68),
    ("login", "app/controllers/auth_controller.py"): (65, 72), ("_client_ip", "app/controllers/auth_controller.py"): (19, 20),
    ("set_auth_cookies", "app/controllers/auth_controller.py"): (23, 28), ("user_out", "app/controllers/auth_controller.py"): (36, 43),
    ("AuthService.login", "app/services/auth_service.py"): (84, 94), ("AuthService._start_session", "app/services/auth_service.py"): (123, 131),
    ("UserRepository.find_by_email", "app/repositories/user_repository.py"): (16, 17),
    ("SessionRepository.create", "app/repositories/user_repository.py"): (44, 47),
    ("verify_password", "app/core/security.py"): (21, 25), ("generate_token", "app/core/security.py"): (28, 29),
    ("hash_token", "app/core/security.py"): (32, 34), ("create_access_token", "app/core/security.py"): (37, 45),
}


def _analysis(packet):
    return packet[packet.index("CALL ANALYSIS"):packet.index("ANSWER FORMAT")]


def test_the_confirmed_definitions_are_unchanged_and_nothing_extra_is_added(monkeypatch, taskboard):
    for ctx in (8192, 16384):
        set_window(monkeypatch, ctx)
        events, calls = run_agent(monkeypatch, taskboard, FLOW_PROMPT, oracle)
        packet = calls[0][-1]["content"]
        defs, _ = assert_packet_honest(packet)
        got = {(n, p.replace(ROOT, "")): (int(a), int(b)) for n, p, a, b in defs}
        assert got == CONFIRMED, f"ctx {ctx}: {set(got) ^ set(CONFIRMED)}"
        assert not withheld(events), verdict(events)


def test_mini_trello_connections_are_classified_with_a_large_window(monkeypatch, taskboard):
    set_window(monkeypatch, 16384)
    events, calls = run_agent(monkeypatch, taskboard, FLOW_PROMPT, oracle)
    analysis = _analysis(calls[0][-1]["content"].replace(ROOT, ""))
    # verify_password: an external Argon2 call, not an unknown connection
    assert "`_hasher.verify(...)` [argon2]" in analysis
    # request: browser fetch, a conditional refresh and a conditional recursive call
    assert "`fetch(...)` [browser/runtime fetch]" in analysis
    assert "`refreshSession()`" in analysis and "conditional (in `if` block from line 48)" in analysis
    assert re.search(r"line 49 `request\(.*\)` -> RECURSIVE call to itself; conditional", analysis)
    # call expression kept apart from the definition it resolves to
    assert "`self.sessions.create(...)`" in analysis or "written `self.sessions.create(...)`" in \
        calls[0][-1]["content"]
    # controller data flow
    assert "receives 3 values from `AuthService.login`" in analysis
    assert "`user` = `user`; `access` = `create_access_token(user.id)`; `refresh` = `refresh`" in analysis
    assert "`access` is passed as argument 2 of `set_auth_cookies(...)` (line 71)" in analysis
    assert "`refresh` is passed as argument 3 of `set_auth_cookies(...)` (line 71)" in analysis
    assert "`user` is passed as argument 1 of `user_out(...)` (line 72)" in analysis
    # known dependencies are not described as unknown connections
    for known in ("fetch", "_hasher.verify", "set_cookie", "token_urlsafe", "jwt.encode"):
        assert not re.search(r"`[^`]*" + re.escape(known) + r"[^`]*` -> cannot be established", analysis)
    assert not withheld(events), verdict(events)


def test_at_the_default_window_the_requested_connections_survive_trimming(monkeypatch, taskboard):
    set_window(monkeypatch, 8192)
    events, calls = run_agent(monkeypatch, taskboard, FLOW_PROMPT, oracle)
    packet = calls[0][-1]["content"].replace(ROOT, "")
    analysis = _analysis(packet)
    assert "[argon2]" in analysis and "RECURSIVE" in analysis and "`fetch(...)`" in analysis
    assert "[browser/runtime fetch]" in analysis
    assert "`access` = `create_access_token(user.id)`" in analysis
    # the written expression is preserved in the call-site list at ANY window size
    assert "#12 `AuthService._start_session` -> #15 `SessionRepository.create` at line 125, written `self.sessions.create(...)`" in packet
    assert "lower-priority analysis lines omitted" in analysis            # omissions are declared, never silent
    assert_packet_honest(packet)


@pytest.mark.parametrize("ctx", [4096, 6144, 8192, 12288, 16384])
def test_analysis_never_costs_required_source_at_any_window(monkeypatch, taskboard, ctx):
    set_window(monkeypatch, ctx)
    events, calls = run_agent(monkeypatch, taskboard, FLOW_PROMPT, oracle)
    packet = calls[0][-1]["content"]
    if "SOURCE " in packet:
        defs, missing = assert_packet_honest(packet)
        if ctx >= 8192:
            assert len(defs) == 16 and not [m for m in missing if m[0] in {n for n, _p in CONFIRMED}]


def _good_connections():
    return "".join((
        entry("submit", ROOT + "frontend/src/pages/Login.jsx", 27, 44)
        .replace("**Connection:** y", "**Connection:** calls `login(...)` at line 36 (project function)."),
        entry("request", ROOT + "frontend/src/api.js", 35, 68).replace(
            "**Connection:** y", "**Connection:** calls `fetch(...)` at line 38 (platform API, not a project function); "
            "conditionally calls `refreshSession()` at line 49 and recursively `request(method, path, body, false)` at line 49."),
        entry("login", ROOT + "app/controllers/auth_controller.py", 65, 72).replace(
            "**Connection:** y", "**Connection:** calls `service.login(...)` at line 69, `set_auth_cookies(response, access, refresh)` "
            "at line 71 and `user_out(user)` at line 72.\n**Data Flow:** `access` and `refresh` go to `set_auth_cookies`."),
        entry("verify_password", ROOT + "app/core/security.py", 21, 25).replace(
            "**Connection:** y", "**Connection:** calls `_hasher.verify(...)` at line 23, an external Argon2 method."),
        entry("_start_session", ROOT + "app/services/auth_service.py", 123, 131).replace(
            "**Connection:** y", "**Connection:** calls `self.sessions.create(...)` at line 125."),
    ))


def test_correct_connection_text_is_accepted(monkeypatch, taskboard):
    events, _ = run_agent(monkeypatch, taskboard, FLOW_PROMPT, _good_connections())
    assert not withheld(events), verdict(events)


@pytest.mark.parametrize("name,old,new", [
    ("fabricated call to a project function", "conditionally calls `refreshSession()`",
     "calls `hash_password(...)` at line 40 and conditionally calls `refreshSession()`"),
    ("fabricated call in the controller", "`set_auth_cookies(response, access, refresh)` at line 71",
     "`verify_password(...)` at line 70 and `set_auth_cookies(response, access, refresh)` at line 71"),
    ("fabricated external call", "calls `fetch(...)` at line 38", "calls `axios.get(...)` at line 38"),
    ("call on the wrong line", "`refreshSession()` at line 49", "`refreshSession()` at line 44"),
    ("recursion in a function that is not recursive", "calls `login(...)` at line 36",
     "calls `login(...)` at line 36 and recursively `submit(event)` at line 40"),
    ("downstream call attributed to the caller", "`user_out(user)` at line 72",
     "`user_out(user)` at line 72, which calls `UserOut(...)` at line 37"),
])
def test_fabricated_or_misattributed_connections_are_rejected(monkeypatch, taskboard, name, old, new):
    answer = _good_connections()
    assert old in answer, name
    events, _ = run_agent(monkeypatch, taskboard, FLOW_PROMPT, answer.replace(old, new))
    assert withheld(events), f"{name} was wrongly accepted"
    assert "It says" in verdict(events) or "It cites" in verdict(events)


def test_single_file_retrieval_is_untouched_by_call_analysis(monkeypatch, taskboard):
    events, calls = run_agent(monkeypatch, taskboard, "Inspect the login function in app/controllers/auth_controller.py. "
                              "Return the complete function signature, complete function body, and starting and "
                              "ending line numbers. Do not read or return the entire file.")
    assert not calls and "CALL ANALYSIS" not in verdict(events) and "lines 65\u201372" in verdict(events)


# ----------------------------------------------------------------------------- another project structure
def test_analysis_is_derived_for_a_project_with_no_frontend_backend_or_api_files(monkeypatch, uploader):
    pid = uploader.files({
        "web/static/checkout.js": "async function startCheckout(cart) {\n  const res = await fetch('/api/pay', {method: 'POST', body: JSON.stringify(cart)});\n  return res.json();\n}\ndocument.querySelector('#buy').addEventListener('click', () => startCheckout(window.cart));\n",
        "server/pay_routes.py": "from flask import Flask, request\nfrom server.billing import process_payment\n\napp = Flask(__name__)\n\n@app.route('/api/pay', methods=['POST'])\ndef pay():\n    data = request.get_json()\n    return process_payment(data)\n",
        "server/billing.py": "def charge_card(total):\n    return {'ok': True, 'total': total}\n\n\ndef process_payment(data):\n    total = sum(i['price'] for i in data['items'])\n    return charge_card(total)\n"})
    set_window(monkeypatch, 16384)
    events, calls = run_agent(monkeypatch, pid, "Explain the complete payment process: where does it start and how does it reach the server?", oracle)
    analysis = _analysis(calls[0][-1]["content"])
    assert "`request.get_json()` [flask]" in analysis                 # imported external, receiver of a call
    assert "`fetch(...)` [browser/runtime fetch]" in analysis
    assert re.search(r"line 7 `charge_card\(total\)` -> #\d+ `charge_card`", analysis) or \
        re.search(r"`charge_card\(total\)` -> #\d+", analysis)
    assert "`sum(" not in analysis.replace("built-in", "") or "Python built-in sum" in analysis
    assert not withheld(events), verdict(events)

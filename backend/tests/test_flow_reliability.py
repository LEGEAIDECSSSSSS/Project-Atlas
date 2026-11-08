"""Regression tests for the login-flow failures reproduced on the real Mini Trello project.

Every test that drives the agent uses the REAL pipeline (classification, planning, search,
code index, retrieval tools, evidence recording, context fitting, packet building, verification).
The only stand-in is the model: a scripted function that records the exact prompt it was sent.
NONE of these tests exercises Qwen itself; they prove what Qwen is given and how its answer is
judged, not how Qwen behaves.
"""
import asyncio
import io
import re
import shutil
import zipfile
from pathlib import Path

import pytest

import config
from agent import agent as A
from agent import context as ctx
from agent import evidence as ev
from agent import packet as pk
from agent import verify as vf
from agent.planner import make_plan
from services import code_index, code_parsers
from services import project_service as ps
from services.textlines import source_lines

FIXTURE = Path(__file__).parent / "fixtures" / "taskboard.zip"
ROOT = "Team-Task-Board---Mini-Trello-main/"

FLOW_PROMPT = (
    "Trace the complete login process in this project, starting from the frontend login page and "
    "following the request through the backend until the user is authenticated and the response "
    "is returned.\n\nIdentify the relevant functions and files involved. Explain the sequence of calls "
    "and the responsibility of each function.\n\nInclude:\n1. The frontend function that initiates the "
    "login request.\n2. The API client function that sends the request.\n3. The backend route handler, "
    "including its complete signature.\n4. The authentication service method called by the route.\n"
    "5. How the user is located and how the password is verified.\n6. How authentication tokens and "
    "session cookies are created and returned.\n\nFor every function, provide its file path and line "
    "numbers.\n\nUse the actual project source as evidence. Do not invent missing implementation "
    "details. Do not return entire files. If a detail cannot be verified from the available source, "
    "explicitly say so."
)


# ----------------------------------------------------------------------------- helpers
class _Upload:
    def __init__(self, filename, data):
        self.filename, self.file = filename, io.BytesIO(data)


@pytest.fixture
def uploader():
    made = []

    def upload(data, name="proj.zip"):
        pid = ps.extract_project_with_stats(_Upload(name, data))["project_id"]
        made.append(pid)
        return pid

    def upload_files(files):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for path, content in files.items():
                z.writestr(path, content)
        return upload(buf.getvalue())

    upload.files = upload_files
    yield upload
    for pid in made:
        code_index.clear_cache(pid)
        shutil.rmtree(config.WORKSPACE_DIR / pid, ignore_errors=True)


@pytest.fixture
def taskboard(uploader):
    if not FIXTURE.exists():
        pytest.skip("taskboard fixture missing")
    return uploader(FIXTURE.read_bytes(), "taskboard.zip")


def run_agent(monkeypatch, pid, message, reply="ok", history=None, prior=None):
    """Run the real agent loop with a recording stand-in for the model (NOT Qwen)."""
    ctx.reset_calibration()
    calls = []

    async def fake_model(messages, model, temperature, num_predict=None, **kwargs):
        calls.append([dict(m) for m in messages])
        text = reply(messages) if callable(reply) else reply
        yield {"text": text}
        yield {"done": True, "stats": {"tokens": 10, "tokens_per_second": 10.0, "prompt_tokens": None}}

    monkeypatch.setattr(A, "stream_chat", fake_model)

    async def go():
        return [e async for e in A.run_agent_stream(
            message=message, history=history or [], project_id=pid, mode="code",
            model="qwen2.5-coder:7b", project_context="", evidence=prior or [], tests_approved=False)]

    return asyncio.run(go()), calls


def verdict(events):
    return next(e for e in events if e["type"] == "done")["response"]


def withheld(events):
    return verdict(events).startswith("Atlas could not provide a verified source-evidence answer")


def labels(events):
    return [e.get("label", "") for e in events if e["type"] == "tool"]


def entry(function, path, start, end):
    return (f"**Function:** {function}\n**File Path:** {path}\n**Line Numbers:** {start}-{end}\n"
            "**Purpose:** x\n**Connection:** y\n\n")


def oracle(messages):
    """A perfect model: writes the answer straight from the packet's VERIFIED DEFINITIONS."""
    text = messages[-1]["content"]
    defs = re.findall(r"^\d+\. \w+ `([^`]+)` \| (\S+) \| (\d+)-(\d+)$", text, re.M)
    return "\n".join(entry(n, p, a, b) for n, p, a, b in defs)


def assert_packet_honest(packet):
    """Every VERIFIED DEFINITION is fully present as numbered source, or is named NOT VERIFIED."""
    defs = re.findall(r"^\d+\. \w+ `([^`]+)` \| (\S+) \| (\d+)-(\d+)$", packet, re.M)
    not_verified = set(re.findall(r"^- `([^`]+)` \| (\S+) \| (\d+)-(\d+)", packet, re.M))
    covered = {}
    for block in re.finditer(r"^SOURCE (\S+) \| lines (\d+)-(\d+) \| COMPLETE\n((?:\s*\d+ \| .*\n?)+)",
                             packet, re.M):
        path, start, end = block.group(1), int(block.group(2)), int(block.group(3))
        numbers = [int(n) for n in re.findall(r"^\s*(\d+) \| ", block.group(4), re.M)]
        assert numbers == list(range(start, end + 1)), f"block header {start}-{end} != lines {numbers[:3]}.."
        covered.setdefault(path, set()).update(numbers)
    for name, path, start, end in defs:
        full = all(n in covered.get(path, set()) for n in range(int(start), int(end) + 1))
        assert full or (name, path, start, end) in not_verified, f"{name} {path}:{start}-{end} silently omitted"
    return defs, not_verified


# ----------------------------------------------------------------------------- shared line splitting
def test_source_lines_count_only_real_line_breaks():
    text = "a\n\x0cb\u2028c\x1d\r\nd\re\n"
    assert source_lines(text) == ["a", "\x0cb\u2028c\x1d", "d", "e"]
    assert source_lines("") == [] and source_lines("x") == ["x"] and source_lines("a\n\n") == ["a", ""]


def test_index_and_reader_agree_on_files_containing_form_feeds(uploader):
    pid = uploader.files({"m/ff.py": "def a():\n    return 1\n\x0c\ndef b():\n    return 2\n"})
    location = code_index.project_facts(pid)["symbol_locations"]["b"][0]
    result = A.read_project_file_tool(pid, "m/ff.py", location["start_line"], location["end_line"])
    assert (location["start_line"], location["end_line"]) == (4, 5)
    assert result["content"] == "def b():\n    return 2"


# ----------------------------------------------------------------------------- evidence records
def _record_for(content, lines, complete=True):
    return ev._read_record("read_project_file", "x.py", lines, 500, complete, content)


def test_record_is_incomplete_when_its_text_was_shortened():
    content = "\n".join(f"line {i:04d} " + "x" * 80 for i in range(200))        # ~18 KB
    record = _record_for(content, "1-200")
    assert len(content) > ev.MAX_SOURCE_CHARS
    assert record["complete"] is False and record["source_truncated"] is True
    assert record["retained_lines"] == len(record["source_text"].splitlines()) < 200


def test_record_is_incomplete_when_declared_range_disagrees_with_retained_lines():
    assert _record_for("a\nb\nc", "1-10")["complete"] is False
    assert _record_for("a\nb\nc", "5-7")["complete"] is True
    assert _record_for("a\nb\nc", "5-7", complete=False)["complete"] is False


# ----------------------------------------------------------------------------- packet builder
def _source(name, count):
    return "\n".join(f"{name}_{i} = {i}" for i in range(count))


def _fixture_world():
    """Two files; four functions; two of them share the name `login`."""
    a_lines = ["def login(body):", "    return svc.login(body)", "", "def helper():", "    return 1"]
    b_lines = ["class Svc:", "    def login(self, body):", "        return find(body)", "", "def find(x):", "    return x"]
    records = [
        {"tool": "read_project_file", "path": "api/a.py", "lines": "1-5", "complete": True,
         "source_text": "\n".join(a_lines)},
        {"tool": "read_project_file", "path": "svc/b.py", "lines": "1-6", "complete": True,
         "source_text": "\n".join(b_lines)},
    ]
    route = {"method": "POST", "route": "/login", "handler": "login", "declared_at": "api/a.py:1",
             "chain": [
                 {"kind": "function", "symbol": "login", "path": "api/a.py", "start_line": 1, "end_line": 2},
                 {"kind": "method", "symbol": "Svc.login", "path": "svc/b.py", "start_line": 2, "end_line": 3,
                  "called_from": "api/a.py:2"},
                 {"kind": "function", "symbol": "find", "path": "svc/b.py", "start_line": 5, "end_line": 6,
                  "called_from": "svc/b.py:3"}],
             "edges": [{"from": "login", "to": "Svc.login", "path": "api/a.py", "line": 2},
                       {"from": "login", "to": "find", "path": "svc/b.py", "line": 3}],
             "frontend_chains": [], "frontend_ui_callers": []}
    return records, route


def test_packet_numbers_every_source_line_with_its_real_line_number():
    records, route = _fixture_world()
    text, report = pk.build_packet(records, route, 20000)
    assert "   2 |     def login(self, body):" in text
    assert_packet_honest(text)
    assert not report["missing"]


def test_packet_call_sites_distinguish_same_named_functions():
    records, route = _fixture_world()
    text, _ = pk.build_packet(records, route, 20000)
    assert "#1 `login` -> #2 `Svc.login` at line 2" in text
    assert "#2 `Svc.login` -> #3 `find` at line 3" in text


@pytest.mark.parametrize("budget", [300, 600, 900, 1200, 1500, 2000, 3000, 20000])
def test_packet_never_exceeds_budget_and_never_silently_omits(budget):
    records, route = _fixture_world()
    text, report = pk.build_packet(records, route, budget)
    if report["included"]:
        assert len(text) <= budget or len(report["included"]) == 0
        assert_packet_honest(text)
    else:
        assert "NOT VERIFIED" in text or "No complete, verified source" in text


def test_packet_names_definitions_that_do_not_fit_instead_of_dropping_everything():
    big = "\n".join(f"    value_{i} = compute({i})  # padding to make this long" for i in range(60))
    records = [{"tool": "read_project_file", "path": "m.py", "lines": "1-62", "complete": True,
                "source_text": "def one():\n" + big + "\n    return 1"},
               {"tool": "read_project_file", "path": "n.py", "lines": "1-2", "complete": True,
                "source_text": "def two():\n    return 2"}]
    route = {"chain": [
        {"kind": "function", "symbol": "one", "path": "m.py", "start_line": 1, "end_line": 62},
        {"kind": "function", "symbol": "two", "path": "n.py", "start_line": 1, "end_line": 2}],
        "edges": [], "frontend_chains": [], "frontend_ui_callers": []}
    text, report = pk.build_packet(records, route, 1900)
    assert "   1 | def two():" in text                              # the small essential one still fits
    assert "- `one` | m.py | 1-62 (its source did not fit in the context budget)" in text
    assert [m["symbol"] for m in report["missing"]] == ["one"]
    assert_packet_honest(text)


def test_packet_ignores_incomplete_records_and_reports_the_gap():
    records, route = _fixture_world()
    records[1]["complete"] = False
    text, report = pk.build_packet(records, route, 20000)
    assert "- `Svc.login` | svc/b.py | 2-3 (its complete source was not retrieved)" in text
    assert "SOURCE svc/b.py" not in text
    assert_packet_honest(text)


def test_packet_does_not_verify_a_definition_whose_declaration_is_absent():
    records, route = _fixture_world()
    route["chain"][1]["symbol"] = "Svc.logout"                         # range is covered, name is not there
    text, report = pk.build_packet(records, route, 20000)
    assert [u["symbol"] for u in report["unverified"]] == ["Svc.logout"]
    assert "declaration not found at the stated lines" in text


def test_contained_records_are_not_duplicated():
    records, route = _fixture_world()
    records.append({"tool": "read_project_file", "path": "api/a.py", "lines": "1-2", "complete": True,
                    "source_text": "def login(body):\n    return svc.login(body)"})
    assert len(pk.usable_records(records)) == 2


# ----------------------------------------------------------------------------- verifier
def test_function_entries_are_paired_inside_their_own_block():
    answer = entry("submit", "a/Login.jsx", 27, 44) + entry("login", "a/auth.jsx", 38, 44)
    pairs = [(e["function"], e["path"], e["start"], e["end"]) for e in vf.parse_function_entries(answer)]
    assert pairs == [("submit", "a/Login.jsx", 27, 44), ("login", "a/auth.jsx", 38, 44)]


def test_function_entries_accept_path_first_numbered_and_table_layouts():
    numbered = ("1. **File Path:** app/c.py\n2. **Line Numbers:** 65-72\n3. **Function:** login\n\n"
                "1. **File Path:** app/s.py\n2. **Line Numbers:** 84-94\n3. **Function:** AuthService.login\n")
    assert [(e["function"], e["path"], e["start"]) for e in vf.parse_function_entries(numbered)] == [
        ("login", "app/c.py", 65), ("AuthService.login", "app/s.py", 84)]
    table = "| Function | File Path | Line Numbers |\n|---|---|---|\n| `auth_check` | app/auth_controller.py | 5-9 |\n"
    got = vf.parse_function_entries(table)
    assert got == [{"function": "auth_check", "path": "app/auth_controller.py", "start": 5, "end": 9}]


def test_paths_keep_their_full_extension_and_underscores():
    got = vf.parse_function_entries(entry("x", "a/b.jsx", 1, 2) + entry("y", "a/c.tsx", 3, 4))
    assert [e["path"] for e in got] == ["a/b.jsx", "a/c.tsx"]


def test_intervals_ignore_incomplete_records_and_clamp_to_retained_text():
    complete = {"tool": "read_project_file", "path": "a.py", "lines": "1-10", "complete": True,
                "source_text": "\n".join(["x"] * 10)}
    shortened = {"tool": "read_project_file", "path": "b.py", "lines": "1-10", "complete": True,
                 "source_text": "\n".join(["x"] * 4)}
    partial = {"tool": "read_project_file", "path": "c.py", "lines": "1-10", "complete": False,
               "source_text": "x"}
    intervals = vf._read_intervals([complete, shortened, partial])
    assert intervals == {"a.py": [(1, 10)], "b.py": [(1, 4)]}


def test_stale_incomplete_history_cannot_veto_a_fresh_answer():
    stale = {"tool": "read_project_file", "path": "app/Login.jsx", "lines": "12-27", "complete": False}
    fresh = {"tool": "read_project_file", "path": "app/auth.py", "lines": "1-5", "complete": True,
             "source_text": "a\nb\nc\nd\ne"}
    merged = [stale, fresh]
    assert vf._relevant_incomplete(merged, [fresh], ["Login.jsx"], merged) is False   # not this turn
    assert vf._relevant_incomplete(merged, None, ["Login.jsx"], merged) is True       # legacy behaviour


def test_current_incomplete_read_matters_only_for_files_discussed_and_uncovered():
    partial = {"tool": "read_project_file", "path": "app/Login.jsx", "lines": "12-27", "complete": False}
    covering = {"tool": "read_project_file", "path": "app/Login.jsx", "lines": "1-60", "complete": True,
                "source_text": "\n".join(["x"] * 60)}
    assert vf._relevant_incomplete([partial], [partial], ["Login.jsx"], [partial]) is True
    assert vf._relevant_incomplete([partial], [partial], ["other.py"], [partial]) is False
    both = [partial, covering]
    assert vf._relevant_incomplete(both, both, ["Login.jsx"], both) is False


def test_withheld_notices_are_not_replayed_as_conversation():
    history = [{"role": "user", "content": "trace login"},
               {"role": "assistant", "content": A.WITHHELD_MARKER + " The unverified draft was withheld."},
               {"role": "user", "content": "thanks"},
               {"role": "assistant", "content": "welcome"}]
    assert A.sanitize_history(history) == history[2:]


# ----------------------------------------------------------------------------- planner
@pytest.mark.parametrize("question", [
    "Explain the complete login process.",
    "Where does login start, and how does it reach the backend?",
    "Find the functions involved in login and explain how they connect.",
    "How does registration work from the signup form to the database?",
    "Walk me through what happens when a user uploads a file.",
])
def test_process_questions_are_planned_as_flows(question):
    plan = make_plan(question)
    assert "flow" in plan.intents and plan.flow_candidates


def test_ordinary_questions_are_not_planned_as_flows():
    assert "flow" not in make_plan("Fix the typo in README.md").intents
    assert "flow" not in make_plan("What does the add function return?").intents


def test_route_candidates_prefer_specific_words_and_related_project_words():
    plan = make_plan("Explain the complete payment process from the form to the server")
    paths = ["server/pay_routes.py", "web/checkout.js", "server/billing.py"]
    assert A._route_query_candidates(plan, paths, False, ["pay", "process_payment"])[0] == "pay"
    registration = make_plan("How does registration work from the signup form to the database?")
    got = A._route_query_candidates(registration, ["pages/Register.jsx", "app/auth_controller.py"], False, [])
    assert "register" in got


# ----------------------------------------------------------------------------- parsers
def test_javascript_parser_edge_cases():
    source = (
        'import { useState } from "react";\nimport axios from "axios";\n'
        'export async function fetchUser(id) {\n  const s = "}{ not a brace";   // } comment brace\n'
        '  /* { block } */\n  return axios.get(`/u/${id}`);\n}\nexport const quick = (x) => x * 2;\n'
        'const multi = async (a, b) => {\n  const inner = () => { return 1; };\n'
        '  items.map((i) => { return i; });\n  return inner() + a + b;\n};\n'
        'export default function Page() {\n  const [v, setV] = useState(0);\n'
        '  const handle = useCallback(() => { setV(1); }, []);\n  return <div onClick={handle}>{v}</div>;\n}\n'
        'const api = {\n  get(url) { return 1; },\n  post: async (url, body) => { return 2; },\n};\n')
    got = {(s.qualname, s.start, s.end) for s in code_parsers.parse_js(source).symbols}
    assert got == {("fetchUser", 3, 7), ("quick", 8, 8), ("multi", 9, 13), ("inner", 10, 10),
                   ("Page", 14, 18), ("handle", 16, 16), ("api.get", 20, 20), ("api.post", 21, 21)}
    names = [s.qualname for s in code_parsers.parse_js(source).symbols]
    assert len(names) == len(set(names))                       # no duplicate definitions
    assert "useState" not in names and "axios" not in names     # imported functions are not definitions


def test_python_parser_edge_cases():
    source = (
        'import os\nfrom fastapi import APIRouter\n\n@router.get("/a")\n@limiter.limit("5/min")\n'
        'async def multi_deco(\n    a: int,\n    b: str = "x",\n) -> dict:\n'
        '    """Doc with def fake(): and class Fake:"""\n    def nested():\n        return 1\n'
        '    return {"a": nested()}\n\nclass Service:\n    """class doc"""\n    def login(self, email):\n'
        '        return email\n\n    @staticmethod\n    def build():\n        return 1\n\n'
        'def login(x):\n    return x\n')
    got = {(s.qualname, s.start, s.end) for s in code_parsers.parse_python(source).symbols}
    assert got == {("multi_deco", 4, 13), ("multi_deco.nested", 11, 12), ("Service", 15, 22),
                   ("Service.login", 17, 18), ("Service.build", 20, 22), ("login", 24, 25)}


# ----------------------------------------------------------------------------- real pipeline: Mini Trello
def test_single_file_symbol_retrieval_still_works(monkeypatch, taskboard):
    """Regression guard (Test 1): the earlier fix must be preserved."""
    prompt = ("Inspect the login function in app/controllers/auth_controller.py. Return the complete function "
              "signature, complete function body, and starting and ending line numbers. "
              "Do not read or return the entire file.")
    events, calls = run_agent(monkeypatch, taskboard, prompt)
    got = labels(events)
    assert len(got) == 1 and got[0] == "Read login in " + ROOT + "app/controllers/auth_controller.py"
    text = verdict(events)
    with zipfile.ZipFile(FIXTURE) as z:
        source = z.read(ROOT + "app/controllers/auth_controller.py").decode()
    expected = "\n".join(source_lines(source)[64:72])          # lines 65-72, inclusive
    assert "lines 65\u201372" in text
    assert expected in text and expected.startswith("@router.post") and expected.endswith("user_out(user)")
    assert "def logout" not in text and "def _client_ip" not in text    # nothing beyond the function


def test_exact_failing_prompt_gives_the_model_a_complete_numbered_packet(monkeypatch, taskboard):
    events, calls = run_agent(monkeypatch, taskboard, FLOW_PROMPT, oracle)
    packet = calls[0][-1]["content"]
    assert "did not fit" not in packet and "too small" not in packet
    defs, missing = assert_packet_honest(packet)
    by_name = {(n, p.replace(ROOT, "")): (a, b) for n, p, a, b in defs}
    assert by_name[("login", "app/controllers/auth_controller.py")] == ("65", "72")
    assert by_name[("submit", "frontend/src/pages/Login.jsx")] == ("27", "44")
    assert by_name[("request", "frontend/src/api.js")] == ("35", "68")
    assert len(defs) >= 12
    assert "  65 | @router.post(\"/login\", response_model=UserOut)" in packet.replace(ROOT, "")
    assert not withheld(events), verdict(events)


def test_correct_answer_is_accepted_even_with_stale_incomplete_history(monkeypatch, taskboard):
    stale = [{"tool": "read_project_file", "path": ROOT + "frontend/src/pages/Login.jsx", "lines": "12-27",
              "of": 108, "complete": False, "source_text": "x"},
             {"tool": "read_project_file", "path": ROOT + "frontend/src/auth.jsx", "lines": "1-13",
              "of": 61, "complete": False, "source_text": "x"}]
    history = [{"role": "user", "content": FLOW_PROMPT},
               {"role": "assistant", "content": A.WITHHELD_MARKER + " The unverified draft was withheld.\n" + "x" * 8000}] * 4
    events, calls = run_agent(monkeypatch, taskboard, FLOW_PROMPT, oracle, history=history, prior=stale)
    assert not withheld(events), verdict(events)
    assert "NOT VERIFIED" in calls[0][-1]["content"] or "VERIFIED DEFINITIONS" in calls[0][-1]["content"]


def _bad_answers():
    base = (entry("submit", ROOT + "frontend/src/pages/Login.jsx", 27, 44)
            + entry("login", ROOT + "frontend/src/auth.jsx", 38, 44)
            + entry("login", ROOT + "app/controllers/auth_controller.py", 65, 72))
    return {
        "wrong range": base + entry("_client_ip", ROOT + "app/controllers/auth_controller.py", 16, 23),
        "invented function": base + entry("hash_password", ROOT + "app/core/security.py", 10, 20),
        "right name, wrong file": base + entry("login", ROOT + "frontend/src/pages/Login.jsx", 38, 44),
        "range never read": base + entry("verify_password", ROOT + "app/core/security.py", 21, 90),
        "no labels": "The login starts in the form and goes to the server.",
    }, base


def test_bad_answers_are_rejected_against_real_evidence(monkeypatch, taskboard):
    bad, _ = _bad_answers()
    for name, answer in bad.items():
        events, _calls = run_agent(monkeypatch, taskboard, FLOW_PROMPT, answer)
        assert withheld(events), f"{name} was wrongly accepted"


def test_good_answers_are_accepted_in_every_reasonable_layout(monkeypatch, taskboard):
    _, base = _bad_answers()
    numbered = ("1. **File Path:** " + ROOT + "app/controllers/auth_controller.py\n2. **Line Numbers:** 65-72\n"
                "3. **Function:** login\n\n1. **File Path:** " + ROOT + "frontend/src/pages/Login.jsx\n"
                "2. **Line Numbers:** 27-44\n3. **Function:** submit\n")
    table = ("| Function | File Path | Line Numbers |\n|---|---|---|\n| `submit` | " + ROOT +
             "frontend/src/pages/Login.jsx | 27-44 |\n| `login` | " + ROOT + "app/controllers/auth_controller.py | 65-72 |\n")
    for name, answer in {"full paths": base, "suffix paths": base.replace(ROOT, ""),
                         "path first": numbered, "table": table}.items():
        events, _calls = run_agent(monkeypatch, taskboard, FLOW_PROMPT, answer)
        assert not withheld(events), f"{name}: {verdict(events)[:300]}"


@pytest.mark.parametrize("question", [
    "Explain the complete login process.",
    "Where does login start, and how does it reach the backend?",
    "Find the functions involved in login and explain how they connect.",
])
def test_login_phrasings_investigate_the_real_code_instead_of_stopping_at_a_search(monkeypatch, taskboard, question):
    events, calls = run_agent(monkeypatch, taskboard, question)
    got = labels(events)
    assert any(label.startswith("Traced route") for label in got)
    assert sum(label.startswith("Read") for label in got) >= 3
    assert not any(label.startswith("Searched") for label in got)


def test_wording_differences_still_find_the_route(monkeypatch, taskboard):
    events, _ = run_agent(monkeypatch, taskboard, "How does registration work from the signup form to the database?")
    assert any(label.startswith("Traced route") for label in labels(events))


def test_unknown_concepts_fall_back_to_search_and_still_reach_the_model(monkeypatch, taskboard):
    events, calls = run_agent(monkeypatch, taskboard, "Walk me through what happens when a user creates a card.")
    assert any(label.startswith("Searched") for label in labels(events))
    sent = "\n".join(m["content"] for m in calls[0])
    assert "ATLAS INVESTIGATION" in sent                       # search results were NOT held back


@pytest.mark.parametrize("prompt_tokens", [2731, 3300, 4096, 6144, 12288, 24000])
def test_context_stress_never_loses_source_silently(monkeypatch, taskboard, prompt_tokens):
    monkeypatch.setattr(config, "PROMPT_BUDGET_TOKENS", prompt_tokens)
    events, calls = run_agent(monkeypatch, taskboard, FLOW_PROMPT, oracle)
    packet = calls[0][-1]["content"]
    if "ATLAS VERIFIED FLOW EVIDENCE" in packet and "SOURCE " in packet:
        assert_packet_honest(packet)
    else:
        assert any(phrase in packet for phrase in ("too small", "did not fit", "No complete, verified"))


def test_a_long_conversation_yields_room_to_the_evidence(monkeypatch, taskboard):
    history = []
    for i in range(20):
        history += [{"role": "user", "content": f"question {i} " + "q" * 400},
                    {"role": "assistant", "content": f"answer {i} " + "a" * 1500}]
    events, calls = run_agent(monkeypatch, taskboard, FLOW_PROMPT, oracle, history=history)
    defs, _ = assert_packet_honest(calls[0][-1]["content"])
    assert len(defs) >= 12 and not withheld(events)


# ----------------------------------------------------------------------------- real pipeline: another structure
def test_a_project_with_no_frontend_backend_or_api_files_is_traced_the_same_way(monkeypatch, uploader):
    pid = uploader.files({
        "web/static/checkout.js": "async function startCheckout(cart) {\n  const res = await fetch('/api/pay', {method: 'POST', body: JSON.stringify(cart)});\n  return res.json();\n}\ndocument.querySelector('#buy').addEventListener('click', () => startCheckout(window.cart));\n",
        "server/pay_routes.py": "from flask import Flask, request\nfrom server.billing import process_payment\n\napp = Flask(__name__)\n\n@app.route('/api/pay', methods=['POST'])\ndef pay():\n    data = request.get_json()\n    return process_payment(data)\n",
        "server/billing.py": "def charge_card(total):\n    return {'ok': True, 'total': total}\n\n\ndef process_payment(data):\n    total = sum(i['price'] for i in data['items'])\n    return charge_card(total)\n"})
    events, calls = run_agent(monkeypatch, pid, "Explain the complete payment process: where does it start and how does it reach the server?", oracle)
    packet = calls[0][-1]["content"]
    defs, missing = assert_packet_honest(packet)
    assert {(n, p, a, b) for n, p, a, b in defs} == {
        ("startCheckout", "web/static/checkout.js", "1", "4"), ("pay", "server/pay_routes.py", "6", "9"),
        ("process_payment", "server/billing.py", "5", "7"), ("charge_card", "server/billing.py", "1", "2")}
    assert "ROUTE: POST /api/pay -> pay" in packet
    assert not withheld(events), verdict(events)


def test_long_function_is_never_reported_complete(monkeypatch, uploader):
    body = "\n".join(f"    value_{i} = compute({i})  # padding comment to make the function long" for i in range(220))
    pid = uploader.files({"svc/big.py": "def short_one():\n    return 1\n\n\ndef giant(x):\n" + body + "\n    return x\n"})
    result = A.read_project_file_tool(pid, "svc/big.py", 5, 226)
    records = ev.records_from_tool("read_project_file", {"path": "svc/big.py"}, result)
    assert result.get("truncated") and records and all(not r["complete"] for r in records)
    assert pk.usable_records(records) == []

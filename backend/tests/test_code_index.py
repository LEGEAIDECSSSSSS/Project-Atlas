"""The structural index: parsers, symbol/graph tools, route tracing, project map."""
import pytest

from services import code_index as ci
from services import code_parsers as cp
from tests.conftest import SAMPLE_PROJECT


@pytest.fixture
def shop(project):
    return project(SAMPLE_PROJECT)


# ----------------------------------------------------------------------------- parsers
def test_python_symbols_have_exact_ranges_and_decorators_included():
    text = "import os\n\n@deco\ndef a(x):\n    return b(x)\n\n\nclass C:\n    def m(self):\n        return a(1)\n"
    parsed = cp.parse_source("m.py", text)
    names = {s.qualname: (s.start, s.end, s.kind) for s in parsed.symbols}
    assert names["a"] == (3, 5, "function")
    assert names["C"] == (8, 10, "class") and names["C.m"] == (9, 10, "method")
    assert ("b", "a") in {(c.callee, c.caller) for c in parsed.calls}
    assert ("a", "C.m") in {(c.callee, c.caller) for c in parsed.calls}


def test_python_syntax_errors_fall_back_instead_of_failing():
    parsed = cp.parse_source("broken.py", "def ok():\n    pass\n\ndef bad(:\n  x\n")
    assert parsed.parser.startswith("generic")
    assert "ok" in [s.name for s in parsed.symbols]


def test_python_routes_and_entry_points():
    parsed = cp.parse_source("a.py", SAMPLE_PROJECT["backend/api/auth.py"])
    assert {(r.method, r.path, r.handler) for r in parsed.routes} == {
        ("POST", "/login", "login"), ("GET", "/health", "health")}
    main = cp.parse_source("main.py", SAMPLE_PROJECT["backend/main.py"])
    assert {e.kind for e in main.entry_points} >= {"app-object", "main-guard", "main()"}


def test_flask_route_with_methods_and_django_urls():
    flask = cp.parse_source("a.py", "@app.route('/x', methods=['GET', 'POST'])\ndef x():\n    pass\n")
    assert {(r.method, r.path) for r in flask.routes} == {("GET", "/x"), ("POST", "/x")}
    django = cp.parse_source("proj/urls.py", "urlpatterns = [path('a/', views.index), path('b/', other)]\n")
    assert {(r.path, r.handler) for r in django.routes} == {("/a/", "index"), ("/b/", "other")}


def test_js_symbols_imports_routes_and_api_calls():
    parsed = cp.parse_source("a.jsx", SAMPLE_PROJECT["frontend/src/App.jsx"])
    assert {s.name for s in parsed.symbols} >= {"LoginForm", "submit"}
    assert [i.module for i in parsed.imports] == ["react", "./api"]
    api = cp.parse_source("api.js", SAMPLE_PROJECT["frontend/src/api.js"])
    assert [(a.method, a.path) for a in api.api_calls] == [("POST", "/login")]
    express = cp.parse_source("s.js", "app.get('/users/:id', getUser);\nrouter.post('/x', async (req, res) => { res.send(1) });\n")
    assert {(r.method, r.path, r.handler) for r in express.routes} == {
        ("GET", "/users/:id", "getUser"), ("POST", "/x", "<inline>")}


def test_js_braces_in_strings_and_comments_do_not_break_block_ends():
    text = "function f() {\n  const s = '}';\n  // }\n  return 1;\n}\nfunction g() {}\n"
    parsed = cp.parse_source("a.js", text)
    ends = {s.name: s.end for s in parsed.symbols}
    assert ends == {"f": 5, "g": 6}


def test_java_class_methods_and_spring_routes():
    parsed = cp.parse_source("UserController.java", SAMPLE_PROJECT["src/main/java/com/shop/UserController.java"])
    assert {s.qualname for s in parsed.symbols} >= {"UserController", "UserController.getUser"}
    assert [(r.method, r.path, r.handler) for r in parsed.routes] == [("GET", "/api/users/{id}", "getUser")]
    method = next(s for s in parsed.symbols if s.name == "getUser")
    assert method.start == 10 and method.end == 13          # annotation line included


def test_a_parser_bug_never_escapes(monkeypatch):
    monkeypatch.setattr(cp, "parse_js", lambda text: 1 / 0)
    assert cp.parse_source("a.js", "function f(){}").parser.startswith("generic")


# ----------------------------------------------------------------------------- tools
def test_find_symbol_exact_partial_and_missing(shop):
    r = ci.find_symbol(shop.project_id, "authenticate_user")
    assert [(s["path"], s["start_line"]) for s in r["symbols"] if s["kind"] == "function"] == [
        ("backend/services/auth_service.py", 4)]
    partial = ci.find_symbol(shop.project_id, "authenticate")
    assert partial["symbols"] and all(s["match_type"] == "partial" for s in partial["symbols"])
    none = ci.find_symbol(shop.project_id, "no_such_thing")
    assert none["symbols"] == [] and "No symbol named" in none["note"]
    with pytest.raises(ValueError):
        ci.find_symbol(shop.project_id, "  ")


def test_read_symbol_returns_a_complete_function_with_its_imports(shop):
    r = ci.read_symbol(shop.project_id, name="authenticate_user")
    assert r["complete"] and r["start_line"] == 4 and r["end_line"] == 9
    assert r["content"].startswith("def authenticate_user") and r["content"].rstrip().endswith("return user")
    assert "from backend.services.user_repo import" in r["imports"]


def test_read_symbol_by_line_truncates_visibly_and_offers_a_continuation(project):
    body = "def big():\n" + "".join(f"    x{i} = {i}\n" for i in range(200)) + "    return 1\n"
    w = project({"b.py": body})
    r = ci.read_symbol(w.project_id, path="b.py", line=50, max_chars=300)
    assert r["truncated"] and not r["complete"] and r["next_start_line"] == r["end_line"] + 1
    assert r["symbol_end_line"] > r["end_line"]


def test_read_symbol_ambiguity_and_errors(project):
    w = project({"a.py": "def dup():\n    pass\n", "b.py": "def dup():\n    pass\n"})
    with pytest.raises(ValueError, match="several files"):
        ci.read_symbol(w.project_id, name="dup")
    assert ci.read_symbol(w.project_id, name="dup", path="b.py")["path"] == "b.py"
    with pytest.raises(ValueError):
        ci.read_symbol(w.project_id)
    with pytest.raises(ValueError, match="No definition"):
        ci.read_symbol(w.project_id, name="nope")
    outside = ci.read_symbol(w.project_id, path="a.py", line=2)
    assert outside["name"] == "dup"


def test_find_references_classifies_each_use(shop):
    r = ci.find_references(shop.project_id, "authenticate_user")
    kinds = {(x["path"], x["kind"]) for x in r["references"]}
    assert ("backend/services/auth_service.py", "definition") in kinds
    assert ("backend/api/auth.py", "import") in kinds and ("backend/api/auth.py", "call") in kinds
    assert r["by_kind"]["definition"] == 1


def test_find_callers_and_callees(shop):
    callers = ci.find_callers(shop.project_id, "authenticate_user")
    assert {c["caller"] for c in callers["callers"]} == {"login", "test_login_ok"}
    assert callers["callers"][-1]["caller_is_test"] is True      # tests listed after real callers
    callees = ci.find_callees(shop.project_id, "authenticate_user")
    resolved = {c["callee"]: c["resolved"][0]["path"] for c in callees["callees"] if c["resolved"]}
    assert resolved == {"find_user": "backend/services/user_repo.py",
                        "verify_password": "backend/services/user_repo.py"}
    assert any(c["external"] for c in callees["callees"])        # PermissionError


def test_ambiguous_names_are_reported_not_guessed(project):
    w = project({"a.py": "def helper():\n    pass\n", "b.py": "def helper():\n    pass\n\ndef use():\n    helper()\n"})
    r = ci.find_callers(w.project_id, "helper")
    assert "defined 2 times" in r["note"]
    with pytest.raises(ValueError, match="several files"):
        ci.find_callees(w.project_id, "helper")


def test_imports_are_resolved_both_ways(shop):
    r = ci.find_imports(shop.project_id, "backend/api/auth.py")
    assert any(i["resolved_paths"] == ["backend/services/auth_service.py"] for i in r["imports"])
    assert any(i["external"] and i["module"] == "fastapi" for i in r["imports"])
    back = ci.find_imports(shop.project_id, "backend/services/auth_service.py")
    assert set(back["imported_by"]) == {"backend/api/auth.py", "backend/tests/test_auth.py"}
    js = ci.find_imports(shop.project_id, "frontend/src/App.jsx")
    assert any(i["resolved_paths"] == ["frontend/src/api.js"] for i in js["imports"])
    with pytest.raises(ValueError, match="not found"):
        ci.find_imports(shop.project_id, "nope.py")


def test_trace_route_follows_handler_services_and_frontend(shop):
    r = ci.trace_route(shop.project_id, "/login")
    route = r["routes"][0]
    assert (route["method"], route["route"]) == ("POST", "/login")
    assert [n["symbol"] for n in route["chain"][:2]] == ["login", "authenticate_user"]
    assert {n["path"] for n in route["chain"]} >= {"backend/services/user_repo.py"}
    assert route["frontend_callers"][0]["path"] == "frontend/src/api.js"
    assert {"path": "backend/api/auth.py", "start_line": 7, "end_line": 11} in route["files_to_read"]
    assert any(e["from"] == "authenticate_user" and e["to"] == "find_user" for e in route["edges"])


def test_trace_route_by_keyword_and_fallback(shop):
    assert ci.trace_route(shop.project_id, "health check")["routes"][0]["route"] == "/health"
    none = ci.trace_route(shop.project_id, "zzz unknown")
    assert none["routes"] == [] and "No route" in none["note"]
    sym = ci.trace_route(shop.project_id, "verify password")
    assert sym["routes"] == [] and any(s["name"] == "verify_password" for s in sym["related_symbols"])


def test_tests_for_symbol(shop):
    r = ci.find_tests_for_symbol(shop.project_id, "authenticate_user")
    assert r["test_files"][0]["path"] == "backend/tests/test_auth.py"
    assert r["test_files"][0]["test_functions"] == ["test_login_ok"]
    assert ci.find_tests_for_symbol(shop.project_id, "find_user")["note"]    # no direct test: says so


def test_entry_points_cover_code_manifests_and_docker(project):
    files = dict(SAMPLE_PROJECT)
    files["Dockerfile"] = "FROM python\nCMD [\"python\", \"-m\", \"backend.main\"]\n"
    w = project(files)
    kinds = {(e["kind"], e["path"]) for e in ci.find_entry_points(w.project_id)["entry_points"]}
    assert ("main-guard", "backend/main.py") in kinds
    assert ("app-object", "backend/main.py") in kinds
    assert ("CMD", "Dockerfile") in kinds
    assert any(k.startswith("npm script 'start'") for k, _ in kinds)


def test_project_map_is_compact_and_factual(shop):
    text = ci.project_map_text(shop.project_id, 3000)
    assert "PROJECT MAP" in text and "small web shop" in text
    assert "python" in text and "javascript" in text
    assert "POST /login -> login" in text and "FastAPI" in text and "React" in text
    assert len(ci.project_map_text(shop.project_id, 500)) <= 500


def test_facts_include_symbols_edges_and_line_counts(shop):
    facts = ci.project_facts(shop.project_id)
    assert "authenticate_user" in facts["symbols"]
    assert ("authenticate_user", "find_user") in facts["call_edges"]
    assert facts["line_counts"]["backend/services/user_repo.py"] == 9


def test_index_rebuilds_when_files_change(project):
    w = project({"a.py": "def one():\n    pass\n"})
    assert ci.find_symbol(w.project_id, "two")["symbols"] == []
    (w.root / "a.py").write_text("def one():\n    pass\n\ndef two():\n    pass\n")
    assert ci.find_symbol(w.project_id, "two")["symbols"][0]["name"] == "two"


def test_unparseable_and_huge_files_do_not_break_the_index(project, monkeypatch):
    monkeypatch.setattr(ci, "MAX_INDEX_FILE_BYTES", 50)
    w = project({"small.py": "def s():\n    pass\n", "huge.py": "def h():\n    pass\n" + "# pad\n" * 40,
                 "bin.py": "\x00\x01\x02"})
    assert [s["name"] for s in ci.find_symbol(w.project_id, "s")["symbols"] if s["match_type"] == "exact"] == ["s"]
    index = ci.get_index(w.project_id)
    assert index.skipped_large == 1 and index.unparsed == 1
    assert "Not indexed" in ci.project_map_text(w.project_id, 2000)

"""The multi-ecosystem test runner: detection, summaries, honest statuses."""
import shutil
import sys

import pytest

from services import test_runner as tr
from agent import evidence as ev


def run(w, **kw):
    return tr.run_project_tests(w.project_id, **kw)


def test_python_project_runs_pytest_and_reports_counts(project):
    w = project({"tests/test_a.py": "def test_ok():\n    assert 1\n\ndef test_bad():\n    assert 0\n",
                 "pytest.ini": "[pytest]\n"})
    r = run(w)
    assert r["ecosystem"] == "python" and r["status"] == "failed"
    assert r["summary"]["tests_run"] == 2 and r["summary"]["failures"] == 1
    assert ev.test_status(r) == "failed"


def test_passing_python_project(project):
    w = project({"test_x.py": "def test_ok():\n    assert True\n"})
    r = run(w)
    assert r["status"] == "passed" and r["summary"]["tests_run"] == 1
    assert ev.test_status(r) == "passed"


def test_collecting_no_tests_is_zero_tests_not_a_pass(project):
    w = project({"pytest.ini": "[pytest]\n", "tests/test_empty.py": "x = 1\n"})
    r = run(w)
    assert r["summary"]["tests_run"] == 0
    assert ev.test_status(r) == "no_tests"


def test_unittest_is_used_when_pytest_is_unavailable(project, monkeypatch):
    monkeypatch.setattr(tr.importlib.util, "find_spec", lambda name: None)
    w = project({"tests/test_u.py": "import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertTrue(True)\n",
                 "tests/__init__.py": ""})
    r = run(w)
    assert r["status"] == "passed" and r["summary"]["tests_run"] == 1


def test_a_clean_exit_without_test_counts_is_unverified(project, monkeypatch):
    w = project({"test_x.py": "def test_ok():\n    pass\n"})
    monkeypatch.setattr(tr, "run_bounded", lambda *a, **k: (0, "everything is fine", False))
    r = run(w)
    assert r["status"] == "unverified" and "no test counts" in r["note"]
    assert ev.test_status(r) == "unverified"


def test_timeouts_are_reported(project, monkeypatch):
    w = project({"test_x.py": "def test_ok():\n    pass\n"})
    monkeypatch.setattr(tr, "run_bounded", lambda *a, **k: (None, "partial", True))
    assert run(w)["status"] == "timeout"


def test_no_tests_at_all_is_an_error_listing_what_is_supported(project):
    r = run(project({"README.md": "hi\n", "app.py": "x = 1\n"}))
    assert r["status"] == "error" and "No test setup" in r["error"]


def test_several_kinds_of_tests_require_an_explicit_choice(project):
    w = project({"pom.xml": "<project/>", "test_x.py": "def test_ok():\n    assert True\n"})
    r = run(w)
    assert r["status"] == "error" and "maven" in r["error"] and "python" in r["error"]
    assert run(w, ecosystem="python")["status"] == "passed"
    assert run(w, ecosystem="cobol")["status"] == "error"
    assert "No go test setup" in run(w, ecosystem="go")["error"]


def test_node_needs_installed_dependencies_and_never_installs_them(project):
    w = project({"package.json": '{"scripts": {"test": "jest"}}', "a.test.js": "test('x', () => {})\n"})
    r = run(w)
    assert r["ecosystem"] == "node" and r["status"] == "error"
    assert "NOT run" in r["error"] and "npm install" in r["error"]


def test_the_default_npm_placeholder_is_not_a_test_script(project):
    w = project({"package.json": '{"scripts": {"test": "echo \\"Error: no test specified\\" && exit 1"}}'})
    assert "node" not in tr.detect_ecosystems(w.root)


def test_missing_tools_are_reported_not_crashed(project, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert "Go could not be found" in run(project({"go.mod": "module x\n"}))["error"]
    assert "Cargo could not be found" in run(project({"Cargo.toml": "[package]\n"}))["error"]
    assert "Gradle could not be found" in run(project({"build.gradle": "x"}))["error"]


def test_nested_modules_of_one_kind_are_one_project(project):
    w = project({"pom.xml": "<project/>", "mod/pom.xml": "<project/>"})
    assert len(tr.detect_ecosystems(w.root)["maven"]) == 1
    w2 = project({"a/pom.xml": "<project/>", "b/pom.xml": "<project/>"})
    r = run(w2)
    assert r["status"] == "error" and "Several maven projects" in r["error"]


@pytest.mark.parametrize("fn,text,expected", [
    (tr.summarize_pytest, "..F\n1 failed, 2 passed in 0.12s", (3, 1, 0, 0)),
    (tr.summarize_pytest, "========= 4 passed, 2 skipped in 1.00s =========", (6, 0, 0, 2)),
    (tr.summarize_pytest, "no tests ran in 0.01s", (0, 0, 0, 0)),
    (tr.summarize_pytest, "1 error in 0.02s", (1, 0, 1, 0)),
    (tr.summarize_unittest, "Ran 5 tests in 0.1s\n\nOK (skipped=1)", (5, 0, 0, 1)),
    (tr.summarize_unittest, "Ran 3 tests in 0.1s\n\nFAILED (failures=1, errors=1)", (3, 1, 1, 0)),
    (tr.summarize_node, "Tests:       2 failed, 1 skipped, 7 passed, 10 total", (10, 2, 0, 1)),
    (tr.summarize_node, "  3 passing (5ms)\n  1 failing\n  2 pending", (6, 1, 0, 2)),
    (tr.summarize_node, " Tests  1 failed | 4 passed (5)", (5, 1, 0, 0)),
    (tr.summarize_node, "# tests 5\n# pass 4\n# fail 1\n", (5, 1, 0, 0)),
    (tr.summarize_go, "--- PASS: TestA (0s)\n--- FAIL: TestB (0s)\n--- SKIP: TestC", (3, 1, 0, 1)),
    (tr.summarize_go, "?   \tx\t[no test files]", (0, 0, 0, 0)),
    (tr.summarize_cargo, "test result: ok. 3 passed; 1 failed; 2 ignored;\ntest result: ok. 1 passed; 0 failed; 0 ignored;", (7, 1, 0, 2)),
])
def test_output_summaries(fn, text, expected):
    s = fn(text)
    assert (s["tests_run"], s["failures"], s["errors"], s["skipped"]) == expected


def test_summaries_return_none_for_unrecognised_output():
    for fn in (tr.summarize_pytest, tr.summarize_unittest, tr.summarize_node, tr.summarize_go, tr.summarize_cargo):
        assert fn("hello world") is None and fn("") is None and fn(None) is None


def test_junit_xml_reports_are_read(tmp_path):
    folder = tmp_path / "build" / "test-results" / "test"
    folder.mkdir(parents=True)
    (folder / "TEST-a.xml").write_text('<testsuite tests="3" failures="1" errors="0" skipped="1"/>')
    (folder / "TEST-b.xml").write_text('<testsuites><testsuite tests="2" failures="0" errors="1"/></testsuites>')
    (folder / "TEST-broken.xml").write_text("<nope")
    assert tr.junit_xml_summary([tmp_path / "build" / "test-results", tmp_path / "missing"]) == {
        "tests_run": 5, "failures": 1, "errors": 1, "skipped": 1}
    assert tr.junit_xml_summary([tmp_path / "empty"]) is None


def test_the_agent_tool_accepts_an_ecosystem_argument():
    from agent import agent as A
    assert A.validate_tool_arguments("run_project_tests", {"ecosystem": "Python"}) == ({"ecosystem": "python"}, None)
    assert A.validate_tool_arguments("run_project_tests", {}) == ({}, None)

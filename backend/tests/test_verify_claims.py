"""Claim-level verification: symbols, line citations, call edges, strict-mode triggers."""
from agent import evidence as ev
from agent import verify as V

FACTS = {
    "symbols": {"run_agent_stream", "execute_tool", "UserService", "find_user"},
    "files": {"agent/agent.py", "services/user_service.py"},
    "call_edges": {("run_agent_stream", "execute_tool")},
    "line_counts": {"agent/agent.py": 100},
}
FILES = ["agent/agent.py", "services/user_service.py"]


def notes(answer, evidence=(), facts=FACTS, user="explain"):
    return V.check_answer(answer, FILES, user, list(evidence), facts=facts)


def test_invented_symbols_are_flagged_real_ones_are_not():
    out = notes("`run_agent_stream` uses `ghost_helper` and `find_user`.")
    assert len(out) == 1 and "ghost_helper" in out[0] and "run_agent_stream" not in out[0]


def test_symbol_seen_in_read_source_is_supported_even_without_an_index():
    source = [{"tool": "read_project_file", "path": "a.py", "source_text": "def local_helper(): pass"}]
    assert notes("It calls `local_helper`.", source, facts=None) == []
    assert notes("It calls `other_helper`.", source, facts=None)


def test_names_the_user_typed_or_the_answer_defines_are_not_flagged():
    assert notes("Try `brand_new_function`.", user="add brand_new_function") == []
    answer = "Add this:\n```python\ndef proposed_helper():\n    pass\n```\nThen call `proposed_helper`."
    assert notes(answer) == []


def test_ordinary_words_and_stdlib_names_are_ignored():
    assert notes("Use `len`, `print`, `None`, `self` and `user`.") == []


def test_nothing_is_judged_without_anything_to_judge_against():
    assert notes("It calls `ghost_helper`.", facts=None) == []


def test_line_citations_are_checked_per_file():
    assert any("only 100 lines" in n for n in notes("See agent/agent.py:250 for details."))
    assert notes("See agent/agent.py:42 and lines 10-20 of agent/agent.py.") == []
    assert any("agent/agent.py" in n for n in notes("The bug is in lines 90-300 of agent/agent.py."))
    # a file the checker cannot place is never given a made-up line count
    assert not any("cites" in n for n in notes("See elsewhere/other.py:9999."))


def test_line_totals_come_from_evidence_when_there_are_no_facts():
    record = [{"tool": "read_project_file", "path": "agent/agent.py", "of": 20, "lines": "1-20"}]
    assert any("only 20 lines" in n for n in notes("See agent/agent.py:50.", record, facts=None))


def test_call_edges_are_checked_when_both_ends_are_known():
    assert notes("`run_agent_stream` calls `execute_tool`.") == []
    out = notes("`execute_tool` calls `run_agent_stream`.")
    assert out and "no such call" in out[0]
    assert notes("`execute_tool` calls `unknown_fn`.") == [] or "unknown_fn" in notes("`execute_tool` calls `unknown_fn`.")[0]
    assert V.claim_items("`a_b_c` -> `d_e_f`", facts={"symbols": set(), "call_edges": set()}) == []


def test_grounding_summary_counts_supported_claims():
    summary = V.grounding_summary(
        "`run_agent_stream` and `ghost_one` and `ghost_two` (agent/agent.py:5)", [], FACTS)
    assert summary["checked"] == 4 and summary["supported"] == 2
    assert set(summary["unsupported"]) == {"ghost_one", "ghost_two"}
    assert V.grounding_summary("plain text", [], FACTS) == {"checked": 0, "supported": 0, "unsupported": []}


def test_low_coverage_adds_a_summary_warning():
    out = notes("`aa_one` `bb_two` `cc_three` `dd_four` `run_agent_stream`")
    assert any("Only 1 of 5" in n for n in out)


def test_library_names_are_not_missing_files():
    assert notes("This is built with Next.js and Node.js, not a project file.") == []
    assert any("ghost.js" in n for n in notes("See ghost.js for the bug."))


def test_everyday_phrases_do_not_switch_on_strict_mode():
    assert not V.requires_source_evidence("why is the file not found when I open it?")
    assert not V.requires_source_evidence("trace the error that happens when the page loads a request")
    assert V.requires_source_evidence("give me an evidence table of the routes")
    assert V.requires_source_evidence("write NOT FOUND for anything missing")
    assert V.requires_source_evidence("trace the login request through the code")


def test_test_claims_follow_the_recorded_status():
    passed = [{"tool": "run_project_tests", "status": "passed"}]
    failed = [{"tool": "run_project_tests", "status": "failed"}]
    assert notes("All tests pass now.", passed) == []
    assert any("failed" in n for n in notes("All tests pass now.", failed))
    assert any("no tests have been run" in n for n in notes("All tests pass now."))
    assert notes("Once the tests pass, merge it.") == []


def test_evidence_handles_unverified_runs_and_malformed_results():
    assert ev.test_status({"status": "passed", "summary": {"tests_run": 3}}) == "passed"
    assert ev.test_status({"status": "passed"}) == "unverified"
    assert ev.test_status({"status": "passed", "summary": {"tests_run": 0}}) == "no_tests"
    assert ev.test_status({"status": "unverified"}) == "unverified"
    # results with missing keys or odd shapes never raise
    assert ev.records_from_tool("search_project", {"query": "x"}, {"results": [{"nope": 1}, "str"]})[0]["files"] == []
    assert ev.records_from_tool("search_project", {}, {"results": None, "automatically_read_file": {"path": 5}})
    assert ev.records_from_tool("read_project_file", {}, {"path": "a.py", "content": None}) != []
    assert ev.merge([[None, "x", {"tool": "list_project_files"}], None]) == [{"tool": "list_project_files"}]
    assert V.check_answer("x", ["a.py"], "q", None) == []


def test_new_tools_leave_compact_records_and_show_in_the_evidence_block():
    rec = ev.records_from_tool("find_callers", {"name": "run_agent_stream"},
                               {"callers": [{"path": "api/chat.py"}], "total": 1})
    assert rec == [{"tool": "find_callers", "query": "run_agent_stream", "total": 1, "files": ["api/chat.py"]}]
    block = ev.render_block(rec)
    assert "find_callers run_agent_stream" in block
    sym = ev.records_from_tool("read_symbol", {}, {"path": "a.py", "qualname": "A.go", "content": "x = 1",
                                                     "start_line": 3, "end_line": 4, "total_lines": 9})
    assert sym[0]["symbol"] == "A.go" and sym[0]["lines"] == "3-4" and "go" in sym[0]["outline"]


def test_outline_uses_the_real_parser_when_it_can():
    text = "class A:\n    def m(self):\n        pass\n\n\ndef f():\n    return 1\n"
    assert ev.outline(text, path="a.py") == ["A", "m", "f"]
    assert ev.outline("def partial(:", path="a.py") == ["partial"] or ev.outline("def partial(:", path="a.py") == []


def test_source_request_does_not_require_a_table_unless_table_was_requested():
    source = [{"tool": "read_project_file", "path": "agent/agent.py",
              "lines": "1-3", "source_text": "def login():\n    return 1"}]
    assert V.check_answer("The function is `login`.", FILES, "Inspect the login function in agent.py", source) == []
    assert any("evidence table" in n for n in V.check_answer(
        "The function is `login`.", FILES, "Give me an evidence table for login", source
    ))


def test_read_sources_merges_chunks_in_line_order_instead_of_overwriting():
    records = [
        {"tool": "read_project_file", "path": "agent/agent.py", "lines": "1-2",
         "source_text": "def first():\n    return 1"},
        {"tool": "read_project_file", "path": "agent/agent.py", "lines": "3-4",
         "source_text": "def second():\n    return 2"},
    ]
    assert V._read_sources(records)["agent/agent.py"] == (
        "def first():\n    return 1\ndef second():\n    return 2"
    )


def test_read_symbol_counts_as_a_real_source_read():
    source = [{"tool": "read_symbol", "path": "agent/agent.py", "lines": "10-12",
               "source_text": "def login():\n    return True"}]
    assert V.check_answer("The `login` function exists.", FILES,
                           "Inspect the login function in agent.py", source) == []


def test_structured_function_line_ranges_must_match_index_and_read_source():
    evidence = [{
        "tool": "read_project_file", "path": "app/auth.py", "lines": "10-20", "of": 40,
        "complete": True, "source_text": "\n" * 9 + "def login():\n    return True\n" + "\n" * 9,
    }]
    facts = {
        "symbols": {"login"}, "files": {"app/auth.py"}, "call_edges": set(),
        "line_counts": {"app/auth.py": 40},
        "symbol_locations": {"login": [{"path": "app/auth.py", "start_line": 10, "end_line": 11}]},
    }
    prompt = "Trace the complete login process. Include file paths and line numbers."
    good = "**Function:** `login`\n**File Path:** `app/auth.py`\n**Line Numbers:** 10-11"
    bad_function_range = "**Function:** `login`\n**File Path:** `app/auth.py`\n**Line Numbers:** 12-13"
    unread_range = "**Function:** `login`\n**File Path:** `app/auth.py`\n**Line Numbers:** 30-31"
    assert V.check_answer(good, FILES + ["app/auth.py"], prompt, evidence, facts) == []
    assert any("different line range" in note for note in
               V.check_answer(bad_function_range, FILES + ["app/auth.py"], prompt, evidence, facts))
    assert any("not included in the source" in note for note in
               V.check_answer(unread_range, FILES + ["app/auth.py"], prompt, evidence, facts))


def test_bolded_invented_symbol_names_are_checked():
    evidence = [{"tool": "read_project_file", "path": "a.py", "lines": "1-2", "of": 2,
                 "source_text": "def real_helper():\n    pass\n"}]
    out = V.check_answer("It calls **_invented_helper**.", ["a.py"], "explain", evidence, facts=None)
    assert out and "_invented_helper" in out[0]


def test_each_function_label_requires_its_own_path_and_line_numbers():
    answer = (
        "**Function:** `login`\n**File Path:** `app/auth.py`\n**Line Numbers:** 10-11\n\n"
        "**Function:** `verify_password`\nNo source location was provided."
    )
    out = V.check_answer(
        answer, FILES + ["app/auth.py"],
        "Trace the complete process. For every function, provide its file path and line numbers.",
        [{"tool": "read_project_file", "path": "app/auth.py", "lines": "10-20", "of": 20,
          "complete": True}],
        {"symbols": {"login", "verify_password"}, "files": set(FILES + ["app/auth.py"]),
         "call_edges": set(), "line_counts": {"app/auth.py": 20},
         "symbol_locations": {"login": [{"path": "app/auth.py", "start_line": 10, "end_line": 11}]}}
    )
    assert any("without both its project file path and line numbers" in note for note in out)


def test_unresolved_imported_api_is_not_misreported_as_a_project_definition():
    evidence = [{
        "tool": "read_project_file", "path": "frontend/auth.jsx", "lines": "1-12",
        "of": 12, "complete": True,
        "source_text": "import { useCallback } from 'react'\nconst login = useCallback(() => true, [])\n",
    }]
    facts = {
        "symbols": {"login"}, "files": {"frontend/auth.jsx"}, "call_edges": set(),
        "line_counts": {"frontend/auth.jsx": 12},
        "symbol_locations": {"login": [{"path": "frontend/auth.jsx", "start_line": 2, "end_line": 2}]},
    }
    answer = (
        "**Function:** `useCallback`\n**File Path:** `frontend/auth.jsx`\n**Line Numbers:** 1-2"
    )
    notes = V.check_answer(answer, ["frontend/auth.jsx"], "Explain the source", evidence, facts)
    assert any("not an indexed project definition" in note for note in notes)

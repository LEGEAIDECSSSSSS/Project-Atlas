"""Project search: identifier-aware matching, ranking, pagination, synonyms and file policy."""
import json

import pytest

from services import project_service as ps
from services import search_engine, vocab


def paths(result):
    return [r["path"] for r in result["results"]]


def test_split_identifier_variants():
    assert vocab.split_identifier("getUserProfile") == ["get", "user", "profile"]
    assert vocab.split_identifier("HTTPServerError") == ["http", "server", "error"]
    assert vocab.split_identifier("get_user-profile2") == ["get", "user", "profile2"]
    assert vocab.split_identifier("") == []


def test_words_match_across_identifier_styles(project):
    w = project({"a.py": "def login_user(x):\n    pass\n", "b.js": "function loginUser(x) {}\n",
                 "c.txt": "nothing relevant here\n"})
    result = ps.search_project(w.project_id, "login user")
    assert set(paths(result)) == {"a.py", "b.js"}
    assert result["results"][0]["matches"][0]["match"] in ("exact", "words")


def test_exact_identifier_beats_substring(project):
    w = project({"a.py": "user = 1\n", "b.py": "username = 1\nuserland = 2\n"})
    result = ps.search_project(w.project_id, "user")
    assert paths(result)[0] == "a.py"


def test_definition_outranks_a_comment_mention(project):
    w = project({"defs.py": "def compute_total(items):\n    return sum(items)\n",
                 "notes.py": "# TODO: compute_total should round\nx = 1\n"})
    result = ps.search_project(w.project_id, "compute_total")
    assert paths(result) == ["defs.py", "notes.py"]
    assert "defines it on line 1" in " ".join(result["results"][0]["reasons"])
    assert "only in comments" in " ".join(result["results"][1]["reasons"])


def test_file_name_match_is_rewarded_and_explained(project):
    w = project({"auth_service.py": "x = 1\n", "other.py": "auth = 2\n"})
    result = ps.search_project(w.project_id, "auth service")
    assert result["results"][0]["path"] == "auth_service.py"
    assert any("file name" in r for r in result["results"][0]["reasons"])


def test_ranking_is_deterministic_and_ties_break_by_path(project):
    w = project({f"f{i:02d}.py": "def handler():\n    pass\n" for i in range(12)})
    first = ps.search_project(w.project_id, "handler")
    second = ps.search_project(w.project_id, "handler")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert paths(first) == sorted(paths(first))


def test_pagination_covers_every_file_exactly_once(project):
    w = project({f"m{i:02d}.py": "needle = 1\n" for i in range(25)})
    seen, offset = [], 0
    while True:
        page = ps.search_project(w.project_id, "needle", limit=10, offset=offset)
        assert page["total_files"] == 25 and page["returned_files"] == len(page["results"])
        seen += paths(page)
        if not page["has_more"]:
            assert page["next_offset"] is None
            break
        assert page["next_offset"] == offset + len(page["results"])
        offset = page["next_offset"]
    assert len(seen) == 25 and len(set(seen)) == 25


def test_pagination_arguments_are_validated(project):
    w = project({"a.py": "x = 1\n"})
    with pytest.raises(ValueError):
        ps.search_project(w.project_id, "  ")
    with pytest.raises(ValueError):
        ps.search_project(w.project_id, "x", limit="many")
    assert ps.search_project(w.project_id, "x", limit=9999)["limit"] == 50      # capped, not trusted


def test_synonyms_help_only_when_literal_matches_are_scarce(project):
    w = project({"session.py": "def sign_in(user):\n    return True\n", "readme.txt": "hello\n"})
    result = ps.search_project(w.project_id, "login")
    assert paths(result) == ["session.py"]
    assert "login" in result["synonyms_used"]
    assert any("related word" in r for r in result["results"][0]["reasons"])

    many = project({f"l{i}.py": "def login():\n    pass\n" for i in range(4)} | {"s.py": "def sign_in():\n    pass\n"})
    precise = ps.search_project(many.project_id, "login")
    assert "s.py" not in paths(precise) and precise["synonyms_used"] == {}


def test_test_files_rank_lower_unless_the_query_is_about_tests(project):
    w = project({"app.py": "def pay():\n    pass\n", "tests/test_app.py": "def pay():\n    pass\n"})
    assert paths(ps.search_project(w.project_id, "pay"))[0] == "app.py"
    result = ps.search_project(w.project_id, "pay tests")
    assert "tests/test_app.py" in paths(result)


def test_binary_and_oversized_files_are_not_searched_but_github_files_are(project, monkeypatch):
    w = project({"img.dat": "needle\x00\x01\x02binary", ".github/workflows/ci.yml": "run: needle\n",
                 "big.txt": "needle\n" * 10})
    monkeypatch.setattr(ps, "MAX_SEARCH_FILE_BYTES", 10)
    result = ps.search_project(w.project_id, "needle")
    assert ".github/workflows/ci.yml" not in paths(result)       # 12 bytes > the patched cap
    monkeypatch.setattr(ps, "MAX_SEARCH_FILE_BYTES", 1000)
    result = ps.search_project(w.project_id, "needle")
    assert ".github/workflows/ci.yml" in paths(result) and "big.txt" in paths(result)
    assert "img.dat" not in paths(result)


def test_mostly_control_characters_count_as_binary(project):
    w = project({"weird.txt": "needle " + "\x01\x02\x03\x04" * 400, "ok.txt": "needle\n"})
    assert paths(ps.search_project(w.project_id, "needle")) == ["ok.txt"]


def test_listing_and_search_use_one_file_policy(project):
    w = project({"src/a.py": "needle\n", "node_modules/x/index.js": "needle\n", ".github/ci.yml": "needle\n"})
    listed = ps.list_project_files(w.project_id)
    found = paths(ps.search_project(w.project_id, "needle"))
    assert ".github/ci.yml" in listed and ".github/ci.yml" in found
    assert not any(p.startswith("node_modules") for p in listed + found)


def test_coverage_of_several_words_beats_repetition_of_one(project):
    w = project({"one.py": "\n".join(["# payment"] * 8) + "\n",
                 "both.py": "payment = 1\nrefund = 2\n"})
    result = ps.search_project(w.project_id, "payment refund")
    assert paths(result)[0] == "both.py"
    assert any("covers 2 of 2 words" in r for r in result["results"][0]["reasons"])


def test_match_lines_carry_kind_and_matched_terms(project):
    w = project({"a.py": "def alpha():\n    # beta here\n    return beta\n"})
    result = ps.search_project(w.project_id, "alpha|beta")
    kinds = {m["line"]: m["kind"] for m in result["results"][0]["matches"]}
    assert kinds[1] == "definition" and kinds[2] == "comment" and kinds[3] == "code"
    assert result["results"][0]["matches"][0]["matched_terms"] == ["alpha"]


def test_search_reports_scan_truncation(project, monkeypatch):
    w = project({f"f{i}.py": "needle\n" for i in range(5)})
    monkeypatch.setattr(ps, "SEARCH_TIME_BUDGET_SECONDS", -1)
    result = ps.search_project(w.project_id, "needle")
    assert result["scan_truncated"] is True


def test_parse_query_handles_odd_input():
    assert search_engine.parse_query("|| ,, ") == []     # no words at all
    assert search_engine.parse_query("   ") == []
    terms = search_engine.parse_query("login_user|Get Profile")
    assert [t.raw for t in terms] == ["login_user", "Get Profile"]
    assert terms[0].synonyms == []                       # identifiers are not expanded


def test_scan_cursor_resumes_after_last_scanned_file():
    files = [(name, name) for name in ("a.py", "b.py", "c.py")]
    terms = search_engine.parse_query("needle")
    seen = []

    def read(path):
        seen.append(path)
        return "needle = 1\n"

    result = search_engine.run_search(
        files, terms, limit=10, offset=0, time_budget=10,
        read_text=read, scan_after="a.py",
    )
    assert seen == ["b.py", "c.py"]
    assert result["scanned_files"] == 2
    assert result["scan_cursor"] == "c.py"
    assert result["scan_truncated"] is False

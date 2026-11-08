"""A small, trustworthy record of what the agent has actually done in a conversation.

The model is never asked what it did. The backend writes these records itself from real
tool results, stores them with each reply, and shows them back to the model on later turns
(compactly, without re-sending whole files). The verifier in verify.py also checks answers
against them.
"""
import re

import config
from services import code_parsers
from services.textlines import source_lines

MAX_RECORDS = 40

DEFINITION = re.compile(
    r"^[ \t]*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:def|class|function|interface|enum|struct|fn|func|trait)\s+([A-Za-z_]\w*)",
    re.MULTILINE,
)
JAVA_METHOD = re.compile(
    r"^[ \t]*(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:abstract\s+)?"
    r"[\w<>\[\],.? ]+?\s+([A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)
CONST_FUNCTION = re.compile(
    r"^[ \t]*(?:export\s+)?const\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_]\w*)\s*=>",
    re.MULTILINE,
)
NOT_NAMES = {"if", "for", "while", "switch", "catch", "return", "new", "else"}


def outline(text, limit=14, path=None):
    """Names of the classes/functions/methods defined in a piece of source, in order.

    Uses the real parser for languages it supports (so nothing is missed or invented); falls
    back to patterns for partial or unparseable text.
    """
    if path:
        try:
            parsed = code_parsers.parse_source(path, text or "")
            names = []
            for symbol in sorted(parsed.symbols, key=lambda s: s.start):
                if symbol.name not in names and symbol.name not in NOT_NAMES:
                    names.append(symbol.name)
            if names and not parsed.parser.startswith(("generic", "failed", "skipped", "none")):
                return names[:limit]
        except Exception:
            pass
    found = []
    for pattern in (DEFINITION, JAVA_METHOD, CONST_FUNCTION):
        for match in pattern.finditer(text or ""):
            name = match.group(1)
            if name not in NOT_NAMES:
                found.append((match.start(), name))
    names = []
    for _, name in sorted(found):
        if name not in names:
            names.append(name)
    return names[:limit]


def test_status(result):
    """Normalise a run_project_tests result to: passed, failed, timed_out, error, not_run,
    or no_tests (the build passed but zero tests actually ran)."""
    if not isinstance(result, dict):
        return "error"
    status = result.get("status")
    if status == "not_run":
        return "not_run"
    if status == "timeout":
        return "timed_out"
    if status == "failed":
        return "failed"
    if status == "unverified":
        return "unverified"
    if status == "passed":
        summary = result.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        if summary and summary.get("tests_run") == 0:
            return "no_tests"
        if not summary:
            # A zero exit code with no test counts proves nothing was actually checked.
            return "unverified"
        return "passed"
    return "error"


GRAPH_TOOLS = {
    "find_symbol": ("symbols", "name"),
    "find_references": ("references", "name"),
    "find_callers": ("callers", "name"),
    "find_callees": ("callees", "name"),
    "find_imports": ("imports", "path"),
    "trace_route": ("routes", "query"),
    "find_tests_for_symbol": ("test_files", "name"),
    "find_entry_points": ("entry_points", None),
}


MAX_SOURCE_CHARS = 12000


def _declared_span(lines):
    """Return (start, end) from a ``"a-b"`` range string, or None when it is not numeric."""
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(lines or ""))
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    return (start, end) if end >= start else None


def _read_record(tool_name, path, lines, total, complete, content, symbol=None):
    """Build one source-read record.

    ``complete`` is only kept when the retained text really holds every line of the declared
    range. A record whose text was shortened, or whose line count disagrees with its declared
    range, is demoted to incomplete so no later stage can treat it as full coverage.
    """
    retained_text = content[:MAX_SOURCE_CHARS]
    truncated = len(content) > MAX_SOURCE_CHARS
    if truncated:
        # Keep whole lines only, so retained_lines describes exactly what is stored.
        cut = retained_text.rfind("\n")
        retained_text = retained_text[:cut] if cut > 0 else retained_text
    retained_lines = len(source_lines(retained_text))
    span = _declared_span(lines)
    covers_range = span is not None and retained_lines == span[1] - span[0] + 1
    record = {
        "tool": "read_project_file",
        "path": path,
        "lines": lines,
        "of": total,
        "complete": bool(complete) and not truncated and covers_range,
        "retained_lines": retained_lines,
        "outline": outline(content, path=path),
        "source_text": retained_text,
    }
    if truncated:
        record["source_truncated"] = True
    if symbol:
        record["symbol"] = symbol
    return record


def records_from_tool(tool_name, arguments, result):
    """Evidence records for one finished tool call (a list, possibly empty)."""
    arguments = arguments if isinstance(arguments, dict) else {}

    if tool_name == "calculate":
        return []

    if tool_name == "run_project_tests":
        record = {"tool": tool_name, "status": test_status(result)}
        if isinstance(result, dict) and isinstance(result.get("summary"), dict):
            record["summary"] = result["summary"]
        if isinstance(result, dict) and result.get("ecosystem"):
            record["ecosystem"] = result["ecosystem"]
        return [record]

    if not isinstance(result, dict) or result.get("error"):
        return []

    if tool_name == "read_project_file":
        content = result.get("content") if isinstance(result.get("content"), str) else ""
        return [_read_record(
            tool_name, result.get("path"), f"{result.get('start_line')}-{result.get('end_line')}",
            result.get("total_lines"), not result.get("truncated"), content)]

    if tool_name == "read_symbol":
        content = result.get("content") if isinstance(result.get("content"), str) else ""
        record = _read_record(
            tool_name, result.get("path"), f"{result.get('start_line')}-{result.get('end_line')}",
            result.get("total_lines"), not result.get("truncated"), content,
            symbol=result.get("qualname"))
        if result.get("qualname") and result["qualname"] not in record["outline"]:
            record["outline"] = [result["qualname"].split(".")[-1]] + record["outline"]
        return [record]

    if tool_name == "search_project":
        results = result.get("results")
        results = results if isinstance(results, list) else []
        records = [{
            "tool": tool_name,
            "query": str(arguments.get("query", "")),
            "files": [r["path"] for r in results[:6]
                      if isinstance(r, dict) and isinstance(r.get("path"), str)],
            "total": result.get("total_files", result.get("total_matches")),
        }]
        autos = result.get("automatically_read_files")
        if not isinstance(autos, list):
            single = result.get("automatically_read_file")
            autos = [single] if isinstance(single, dict) else []
        for auto in autos:
            if not isinstance(auto, dict) or not isinstance(auto.get("path"), str):
                continue
            content = auto.get("content") if isinstance(auto.get("content"), str) else ""
            start = auto.get("start_line") or 1
            end = auto.get("end_line") or (start + content.count(chr(10)))
            records.append(_read_record(
                "read_project_file", auto["path"], f"{start}-{end}",
                auto.get("total_lines"),
                auto.get("complete", "[File content truncated" not in content), content))
        return records

    if tool_name == "list_project_files":
        return [{"tool": tool_name, "total": result.get("total")}]

    if tool_name in GRAPH_TOOLS:
        key, arg = GRAPH_TOOLS[tool_name]
        items = result.get(key)
        items = items if isinstance(items, list) else []
        files = []
        for item in items[:8]:
            if isinstance(item, dict) and isinstance(item.get("path"), str) and item["path"] not in files:
                files.append(item["path"])
        return [{
            "tool": tool_name,
            "query": str(arguments.get(arg, "")) if arg else "",
            "total": result.get("total", len(items)),
            "files": files[:6],
        }]

    return []


def merge(record_lists):
    """Combine records from many replies (oldest first), dropping duplicates."""
    merged, seen = [], set()
    for records in record_lists:
        for record in records or []:
            if not isinstance(record, dict):
                continue
            key = (record.get("tool"), record.get("path"), record.get("lines"), record.get("query"))
            if record.get("tool") == "run_project_tests":
                key = ("tests", len(merged))      # every test run is its own event
            if key in seen:
                merged = [r for r in merged if (r.get("tool"), r.get("path"), r.get("lines"), r.get("query")) != key]
            seen.add(key)
            merged.append(record)
    kept = merged[-MAX_RECORDS:]
    # The latest test result must never fall off the list just because many files were read since.
    if not any(r.get("tool") == "run_project_tests" for r in kept):
        last_test = next((r for r in reversed(merged) if r.get("tool") == "run_project_tests"), None)
        if last_test:
            kept = [last_test] + kept[1:]
    return kept


def files_read(records):
    return {r["path"] for r in records or []
            if isinstance(r, dict) and r.get("tool") in ("read_project_file", "read_symbol") and r.get("path")}


def last_test_status(records):
    for record in reversed(records or []):
        if isinstance(record, dict) and record.get("tool") == "run_project_tests":
            return record.get("status", "error"), record.get("summary")
    return "never_run", None


TEST_WORDS = {
    "passed": "PASSED",
    "failed": "FAILED",
    "timed_out": "TIMED OUT (stopped by the time limit; results are incomplete)",
    "error": "COULD NOT RUN (tool error)",
    "not_run": "NOT RUN (the user has not approved running tests)",
    "no_tests": "BUILD PASSED BUT ZERO TESTS RAN (this proves nothing)",
    "unverified": "EXITED SUCCESSFULLY BUT NO TEST COUNT WAS FOUND (this proves nothing)",
    "never_run": "NEVER RUN in this conversation",
}


def render_block(records, max_chars=None):
    """The EVIDENCE section shown to the model. Compact: names and ranges, never file bodies.

    It has a hard size limit (a share of the context window) so a long conversation can never
    crowd out the user's actual message. When space runs short the oldest detail goes first:
    recent files keep their line ranges and definitions, older ones shrink to a bare name, and
    the test status line is always kept.
    """
    if max_chars is None:
        max_chars = int(config.PROMPT_BUDGET_TOKENS * config.CHARS_PER_TOKEN * config.EVIDENCE_SHARE)

    header = [
        "EVIDENCE: what has actually been done in this conversation",
        "(written by the app from real tool results, not by you)",
    ]
    status, summary = last_test_status(records)
    detail = ""
    if summary:
        detail = (f" ({summary.get('tests_run', '?')} run, {summary.get('failures', '?')} failures, "
                  f"{summary.get('errors', '?')} errors)")
    footer = [
        f"Tests, most recent run: {TEST_WORDS.get(status, status)}{detail}",
        "Earlier file contents are not repeated. Re-read a file before quoting it or describing it.",
    ]

    reads = []
    for record in records:
        if record.get("tool") == "read_project_file" and record.get("path"):
            reads = [r for r in reads if r["path"] != record["path"]] + [record]
    searches = [r for r in records if r.get("tool") == "search_project"][-3:]
    lookups = [r for r in records if r.get("tool") in GRAPH_TOOLS][-4:]

    def full_line(r):
        part = f"lines {r.get('lines', '?')}" + (f" of {r['of']}" if r.get("of") else "")
        defs = f"; defines {', '.join(r['outline'][:8])}" if r.get("outline") else ""
        return f"- {r['path']} ({part}{defs})"

    def build(detailed, bare, hidden):
        body = []
        if reads:
            incomplete = hidden > 0 or len(records) >= MAX_RECORDS
            body.append(
                "Files you have opened (this list is incomplete: older ones may be missing):"
                if incomplete
                else "Files you have opened (any file NOT listed here has NOT been read):"
            )
            body += [full_line(r) for r in detailed]
            if bare:
                body.append("- also opened earlier: " + ", ".join(r["path"] for r in bare)
                            + (f" (+{hidden} more)" if hidden else ""))
        else:
            body.append("Files you have opened: none. You have not read any file yet.")
        for r in searches:
            where = ", ".join(r.get("files", [])[:4]) if r.get("files") else "no matches"
            body.append(f'- searched "{str(r.get("query", ""))[:40]}": {where}')
        for r in lookups:
            body.append(f'- {r["tool"]} {str(r.get("query", ""))[:40]}: {r.get("total", 0)} result(s)')
        return "\n".join(header + body + footer)

    keep_detailed = min(len(reads), 8)
    keep_bare = len(reads) - keep_detailed
    while True:
        detailed = reads[len(reads) - keep_detailed:]
        older = reads[: len(reads) - keep_detailed]
        shown = older[-keep_bare:] if keep_bare else []
        text = build(detailed, shown, len(older) - len(shown))
        if len(text) <= max_chars:
            return text
        if keep_bare > 0:
            keep_bare = keep_bare // 2          # shrink the list of older names first
        elif keep_detailed > 1:
            keep_detailed -= 1                  # then the detailed entries
        else:
            return text[:max_chars]

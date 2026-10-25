"""A small, trustworthy record of what the agent has actually done in a conversation.

The model is never asked what it did. The backend writes these records itself from real
tool results, stores them with each reply, and shows them back to the model on later turns
(compactly, without re-sending whole files). The verifier in verify.py also checks answers
against them.
"""
import re

import config

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


def outline(text, limit=14):
    """Names of the classes/functions/methods defined in a piece of source, in order."""
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
    if status == "passed":
        summary = result.get("summary") or {}
        if summary and summary.get("tests_run") == 0:
            return "no_tests"
        return "passed"
    return "error"


def records_from_tool(tool_name, arguments, result):
    """Evidence records for one finished tool call (a list, possibly empty)."""
    if tool_name == "calculate":
        return []

    if tool_name == "run_project_tests":
        record = {"tool": tool_name, "status": test_status(result)}
        if isinstance(result, dict) and result.get("summary"):
            record["summary"] = result["summary"]
        return [record]

    if not isinstance(result, dict) or result.get("error"):
        return []

    if tool_name == "read_project_file":
        return [{
            "tool": tool_name,
            "path": result.get("path"),
            "lines": f"{result.get('start_line')}-{result.get('end_line')}",
            "of": result.get("total_lines"),
            "complete": not result.get("truncated"),
            "outline": outline(result.get("content", "")),
        }]

    if tool_name == "search_project":
        records = [{
            "tool": tool_name,
            "query": arguments.get("query", ""),
            "files": [r["path"] for r in result.get("results", [])[:6]],
        }]
        auto = result.get("automatically_read_file")
        if auto:
            content = auto.get("content", "")
            records.append({
                "tool": "read_project_file",
                "path": auto["path"],
                "lines": f"1-{content.count(chr(10)) + 1}",
                "of": None,
                "complete": "[File content truncated" not in content,
                "outline": outline(content),
            })
        return records

    if tool_name == "list_project_files":
        return [{"tool": tool_name, "total": result.get("total")}]

    return []


def merge(record_lists):
    """Combine records from many replies (oldest first), dropping duplicates."""
    merged, seen = [], set()
    for records in record_lists:
        for record in records or []:
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
    return {r["path"] for r in records if r.get("tool") == "read_project_file" and r.get("path")}


def last_test_status(records):
    for record in reversed(records):
        if record.get("tool") == "run_project_tests":
            return record.get("status", "error"), record.get("summary")
    return "never_run", None


TEST_WORDS = {
    "passed": "PASSED",
    "failed": "FAILED",
    "timed_out": "TIMED OUT (stopped by the time limit; results are incomplete)",
    "error": "COULD NOT RUN (tool error)",
    "not_run": "NOT RUN (the user has not approved running tests)",
    "no_tests": "BUILD PASSED BUT ZERO TESTS RAN (this proves nothing)",
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
        detail = (f" ({summary['tests_run']} run, {summary['failures']} failures, "
                  f"{summary['errors']} errors)")
    footer = [
        f"Tests, most recent run: {TEST_WORDS.get(status, status)}{detail}",
        "Earlier file contents are not repeated. Re-read a file before quoting it or describing it.",
    ]

    reads = []
    for record in records:
        if record.get("tool") == "read_project_file" and record.get("path"):
            reads = [r for r in reads if r["path"] != record["path"]] + [record]
    searches = [r for r in records if r.get("tool") == "search_project"][-3:]

    def full_line(r):
        part = f"lines {r['lines']}" + (f" of {r['of']}" if r.get("of") else "")
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
            where = ", ".join(r["files"][:4]) if r["files"] else "no matches"
            body.append(f'- searched "{r["query"][:40]}": {where}')
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

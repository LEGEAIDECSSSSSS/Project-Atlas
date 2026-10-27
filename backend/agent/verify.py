
"""Check project answers against actual files and tool evidence."""
import re
from pathlib import PurePosixPath

from agent.evidence import files_read, last_test_status

EXTENSIONS = (
    "py|java|js|jsx|ts|tsx|go|rs|cs|cpp|hpp|c|h|php|rb|swift|kt|kts|scala|sql|html|css|vue|svelte|"
    "xml|json|ya?ml|toml|gradle|properties|md|txt|sh|bat|ps1"
)
FILE_NAME = re.compile(
    rf"(?<![\w./\\-])((?:[\w.-]+/)*[\w-][\w.-]*\.(?:{EXTENSIONS}))(?![\w])",
    re.IGNORECASE,
)
FENCE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
FENCE_OPEN = re.compile(r"^```(.*)$", re.MULTILINE)

INSPECTION_CLAIM = re.compile(
    r"\b(?:I(?:'ve|\s+have)?|we(?:'ve|\s+have)?)\s+"
    r"(?:just\s+|now\s+|also\s+)?"
    r"(?:read|reviewed|inspected|examined|opened|looked\s+(?:at|through)|"
    r"checked|analy[sz]ed|gone\s+through|went\s+through|scanned|studied)\b"
    r"|\bafter\s+(?:reading|reviewing|inspecting|examining|checking)\b",
    re.IGNORECASE,
)
TEST_CLAIM = re.compile(
    r"\b(?:all\s+(?:\d+\s+)?tests?\s+(?:now\s+)?pass(?:ed|es)?"
    r"|tests?\s+(?:now\s+|all\s+)?(?:pass(?:ed|es)?|are\s+(?:now\s+)?passing|"
    r"succeed(?:ed)?)|build\s+(?:succeeded|passes|passed|is\s+successful|was\s+successful))\b",
    re.IGNORECASE,
)
HEDGE = re.compile(
    r"\b(?:should|would|will|might|may|could|if|whether|once|until|so\s+that|"
    r"to\s+(?:see|check|verify|confirm)|make\s+sure|ensure|expect|hope|want|when)\b",
    re.IGNORECASE,
)

SOURCE_EVIDENCE_REQUEST = re.compile(
    r"\b(?:evidence\s+table|exact\s+file\s+paths?|exact\s+function\s+names?|"
    r"code\s+excerpts?|quotations?|quote\s+proving|read\s+the\s+actual|"
    r"actual\s+implementations?|not\s+found|trace\s+.*request)\b",
    re.IGNORECASE,
)


def requires_source_evidence(user_message):
    """True for requests that require source-backed paths, names, or quotations."""
    return bool(SOURCE_EVIDENCE_REQUEST.search(user_message or ""))


def _basename(path):
    return PurePosixPath(path.replace("\\", "/")).name.lower()


def _clean_cell(value):
    value = value.strip()
    value = re.sub(r"^\s*`+|`+\s*$", "", value)
    value = re.sub(r"^\s*\*\*|\*\*\s*$", "", value)
    return value.strip().strip("\"'")


def _normalise_source(value):
    """Normalise whitespace so wrapped source lines can still be compared."""
    return " ".join((value or "").split())


def _unhedged(text, start):
    return not HEDGE.search(text[max(0, start - 40):start])


def prose_only(answer):
    return FENCE_BLOCK.sub(" ", answer)


def proposed_paths(answer):
    """Paths on code fences that are being offered as new/replacement files."""
    paths = set()
    for info in FENCE_OPEN.findall(answer):
        for token in info.replace(":", " ").split():
            token = token.strip("\"'`,;")
            if FILE_NAME.fullmatch(token):
                paths.add(token.replace("\\", "/"))
    return paths


def _read_sources(evidence):
    """Return source text actually supplied by read tools, keyed by exact path."""
    sources = {}
    for record in evidence or []:
        if record.get("tool") != "read_project_file":
            continue
        path = record.get("path")
        source = record.get("source_text")
        if path and isinstance(source, str):
            sources[path.replace("\\", "/")] = source
    return sources


def _table_rows(answer):
    """Find a Markdown evidence table and return (header-index-map, data-rows)."""
    lines = answer.splitlines()
    for start, line in enumerate(lines):
        if not line.strip().startswith("|"):
            continue

        headers = [_clean_cell(c).lower() for c in line.strip().strip("|").split("|")]
        if not any("role" in h for h in headers):
            continue
        if not any("path" in h for h in headers):
            continue
        if not any("function" in h for h in headers):
            continue

        indexes = {
            "path": next(i for i, h in enumerate(headers) if "path" in h),
            "function": next(i for i, h in enumerate(headers) if "function" in h),
            "excerpt": next(
                (i for i, h in enumerate(headers)
                 if any(word in h for word in ("excerpt", "quotation", "quote", "proof"))),
                None,
            ),
        }

        rows = []
        for data_line in lines[start + 1:]:
            stripped = data_line.strip()
            if not stripped.startswith("|"):
                if rows:
                    break
                continue

            cells = [_clean_cell(c) for c in stripped.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells):
                continue
            if len(cells) < len(headers):
                cells += [""] * (len(headers) - len(cells))
            rows.append(cells)

        return indexes, rows

    return None, []


def _function_is_defined(name, source):
    """Conservative check for common Python, JS/TS, Java and similar declarations."""
    if not name or not re.fullmatch(r"[A-Za-z_$][\w$]*", name):
        return False

    escaped = re.escape(name)
    patterns = [
        rf"(?m)^\s*(?:async\s+)?def\s+{escaped}\s*\(",             # Python
        rf"(?m)^\s*class\s+{escaped}\b",                           # classes
        rf"(?m)^\s*(?:export\s+)?(?:default\s+)?function\s+\*?{escaped}\s*\(",
        rf"(?m)^\s*(?:export\s+)?(?:const|let|var)\s+{escaped}\s*=",  # JS/TS
        rf"(?m)^\s*(?:(?:public|private|protected|static|final|abstract|async|override)\s+)*"
        rf"[\w<>\[\],.?]+\s+{escaped}\s*\(",                       # common methods
    ]
    return any(re.search(pattern, source) for pattern in patterns)


def _verify_evidence_table(answer, project_files, evidence):
    notes = []
    indexes, rows = _table_rows(answer)
    if indexes is None or not rows:
        return ["The request requires an evidence table, but no readable evidence table was provided."]

    exact_files = {p.replace("\\", "/") for p in project_files}
    sources = _read_sources(evidence)

    for row_number, cells in enumerate(rows, start=1):
        path = cells[indexes["path"]].replace("\\", "/")
        function = cells[indexes["function"]]
        excerpt_index = indexes["excerpt"]
        excerpt = cells[excerpt_index] if excerpt_index is not None else ""

        if path.upper() == "NOT FOUND":
            if function and function.upper() != "NOT FOUND":
                notes.append(f"Evidence table row {row_number} names a function despite its file being NOT FOUND.")
            continue

        if path not in exact_files:
            notes.append(f"Evidence table row {row_number} names `{path}`, which is not an exact path in the attached project.")
            continue

        source = sources.get(path)
        if source is None:
            notes.append(f"Evidence table row {row_number} uses `{path}`, but the backend has no source text from a file read to verify it.")
            continue

        if function.upper() != "NOT FOUND":
            if not _function_is_defined(function, source):
                notes.append(f"Evidence table row {row_number} names `{function}`, but that declaration was not found in `{path}`.")

        if excerpt and excerpt.upper() != "NOT FOUND":
            if _normalise_source(excerpt) not in _normalise_source(source):
                notes.append(f"Evidence table row {row_number} has an excerpt that does not match the source read from `{path}`.")

    return notes


def check_answer(answer, project_files, user_message, evidence):
    """Return deterministic warnings for unsupported claims."""
    strict = requires_source_evidence(user_message or "")
    if not project_files:
        return ["No project files were available to verify this source-evidence request."] if strict else []

    notes = []
    prose = prose_only(answer)
    by_basename = {}
    for path in project_files:
        by_basename.setdefault(_basename(path), []).append(path)

    offered = {_basename(p) for p in proposed_paths(answer)}
    user_text = (user_message or "").lower()

    mentioned = []
    for match in FILE_NAME.finditer(prose):
        name = match.group(1).replace("\\", "/")
        if name not in mentioned:
            mentioned.append(name)

    missing = []
    existing = []
    for name in mentioned:
        base = _basename(name)
        if name in project_files or base in by_basename:
            existing.append(name)
        elif base not in offered and base not in user_text:
            missing.append(name)

    if missing:
        names = ", ".join(f"`{n}`" for n in missing[:5])
        notes.append(
            f"It mentions {names}, which is not in the attached project. "
            "If describing existing code, that part is unsupported."
        )

    read = files_read(evidence)
    read_names = {_basename(p) for p in read}

    if INSPECTION_CLAIM.search(prose):
        unread = [n for n in existing if _basename(n) not in read_names]
        if unread:
            names = ", ".join(f"`{n}`" for n in unread[:5])
            notes.append(
                f"It claims to have inspected {names}, but those files were not recorded as read."
            )
        elif not existing and not read:
            notes.append("It claims to have reviewed the project, but no file was recorded as read.")

    claim = next((m for m in TEST_CLAIM.finditer(prose) if _unhedged(prose, m.start())), None)
    if claim:
        status, _ = last_test_status(evidence)
        if status != "passed":
            words = {
                "never_run": "no tests have been run",
                "failed": "the last test run failed",
                "timed_out": "the last test run timed out",
                "error": "the last test run could not start",
                "not_run": "you have not approved running tests",
                "no_tests": "the last run executed zero tests",
            }.get(status, status)
            notes.append(f"It says tests pass, but {words}.")

    if strict:
        if not any(r.get("tool") == "read_project_file" for r in (evidence or [])):
            notes.append("No project file was actually read for this source-evidence request.")
        notes.extend(_verify_evidence_table(answer, project_files, evidence))

    # Keep the returned warnings unique and in their original order.
    return list(dict.fromkeys(notes))


def format_notes(notes):
    if not notes:
        return ""
    lines = ["", "", "> **Check this reply before trusting it**"]
    lines += [f"> - {note}" for note in notes]
    return "\n".join(lines)

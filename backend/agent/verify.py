
"""Check project answers against actual files and tool evidence."""
import builtins
import keyword
import re
from pathlib import PurePosixPath

from agent.evidence import files_read, last_test_status

EXTENSIONS = (
    "py|java|js|jsx|ts|tsx|go|rs|cs|cpp|hpp|c|h|php|rb|swift|kt|kts|scala|sql|html|css|vue|svelte|"
    "xml|json|ya?ml|toml|gradle|properties|md|txt|sh|bat|ps1"
)
# Bounded repetition keeps a long run of word characters (a base64 blob, minified code) from
# making the match quadratic.
FILE_NAME = re.compile(
    rf"(?<![\w./\\-])((?:[\w.-]{{1,80}}/){{0,12}}[\w-][\w.-]{{0,80}}\.(?:{EXTENSIONS}))(?![\w])",
    re.IGNORECASE,
)
MAX_ANALYZED_CHARS = 60_000      # claims beyond this much text are not checked (and cannot stall a reply)
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

EVIDENCE_TABLE_REQUEST = re.compile(r"\bevidence\s+tables?\b", re.IGNORECASE)

SOURCE_EVIDENCE_REQUEST = re.compile(
    r"\b(?:evidence\s+table|exact\s+file\s+paths?|exact\s+function\s+names?|"
    r"code\s+excerpts?|quotations?|quote\s+proving|read\s+the\s+actual|"
    r"actual\s+implementations?|"
    r"trace\s+(?:an?\s+|the\s+|this\s+|each\s+|every\s+)?(?:[\w-]+\s+){0,3}requests?)\b",
    re.IGNORECASE,
)
# The literal marker the evidence-table format uses. Case-sensitive on purpose: an everyday
# sentence like "why is the file not found?" must not switch on strict mode.
NOT_FOUND_MARKER = re.compile(r"\bNOT FOUND\b")


def requires_source_evidence(user_message):
    """True for requests that require source-backed paths, names, or quotations."""
    text = user_message or ""
    return bool(SOURCE_EVIDENCE_REQUEST.search(text) or NOT_FOUND_MARKER.search(text))


# Product and library names that look like file names (Next.js, Node.js ...).
LIBRARY_NAMES = {
    "node.js", "next.js", "nuxt.js", "vue.js", "react.js", "express.js", "chart.js", "three.js",
    "d3.js", "backbone.js", "ember.js", "angular.js", "nest.js", "p5.js", "alpine.js", "socket.io",
    "vite.js", "deno.js", "bun.js", "ember.js", "knockout.js", "moment.js", "lodash.js",
}


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
    """Combine source chunks by path without letting a later chunk overwrite an earlier one."""
    chunks = {}
    for record in evidence or []:
        if not isinstance(record, dict) or record.get("tool") not in ("read_project_file", "read_symbol"):
            continue
        path = record.get("path")
        source = record.get("source_text")
        if not path or not isinstance(source, str):
            continue
        key = path.replace("\\", "/")
        match = re.match(r"\s*(\d+)-(\d+)\s*$", str(record.get("lines", "")))
        start = int(match.group(1)) if match else len(chunks.get(key, [])) + 1
        lines = chunks.setdefault(key, [])
        for offset, line in enumerate(source.splitlines()):
            line_no = start + offset
            while len(lines) < line_no:
                lines.append(None)
            if lines[line_no - 1] is None:
                lines[line_no - 1] = line
    return {path: "\n".join(line or "" for line in lines) for path, lines in chunks.items()}


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


def check_answer(answer, project_files, user_message, evidence, facts=None):
    """Return deterministic warnings for unsupported claims."""
    strict = requires_source_evidence(user_message or "")
    if not project_files:
        return ["No project files were available to verify this source-evidence request."] if strict else []

    notes = []
    prose = prose_only(answer)[:MAX_ANALYZED_CHARS]
    by_basename = {}
    for path in project_files:
        by_basename.setdefault(_basename(path), []).append(path)

    offered = {_basename(p) for p in proposed_paths(answer)}
    user_text = (user_message or "").lower()

    mentioned = []
    for match in FILE_NAME.finditer(prose):
        name = match.group(1).replace("\\", "/")
        if name.lower() in LIBRARY_NAMES:
            continue
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
                "unverified": "the last run reported no test counts, so nothing is proven",
            }.get(status, status)
            notes.append(f"It says tests pass, but {words}.")

    if strict:
        if not any(r.get("tool") in ("read_project_file", "read_symbol") for r in (evidence or [])):
            notes.append("No project source was actually read for this source-evidence request.")
        # Only enforce a table when the user explicitly requested that presentation format.
        if EVIDENCE_TABLE_REQUEST.search(user_message or ""):
            notes.extend(_verify_evidence_table(answer, project_files, evidence))

    notes.extend(_claim_notes(answer, user_message, evidence, facts))

    # Keep the returned warnings unique and in their original order.
    return list(dict.fromkeys(notes))


# ---------------------------------------------------------------------------
# Claim-level checks: symbols, line citations and call edges
# ---------------------------------------------------------------------------

BACKTICK = re.compile(r"(?<!`)`([^`\n]{2,80})`(?!`)")
IDENT = re.compile(r"^[A-Za-z_$][\w$]*$")
_COMMON_NAMES = (
    set(keyword.kwlist) | set(dir(builtins))
    | {"self", "cls", "this", "null", "undefined", "true", "false", "async", "await", "const",
       "let", "var", "void", "String", "Object", "Array", "Map", "Set", "List", "Optional",
       "HashMap", "ArrayList", "Promise", "JSON", "Date", "Math", "Number", "Boolean", "Error",
       "Exception", "React", "useState", "useEffect", "useRef", "useMemo", "useCallback",
       "__init__", "__name__", "__main__", "main", "args", "kwargs", "README", "Dockerfile"}
)
_LINE_CITE_A = re.compile(
    rf"(?P<path>(?:[\w.-]{{1,80}}/){{0,12}}[\w-][\w.-]{{0,80}}\.(?:{EXTENSIONS}))\s*(?::|#L|\(\s*lines?\s+)\s*"
    r"(?P<a>\d{1,6})(?:\s*[-\u2013]\s*L?(?P<b>\d{1,6}))?",
    re.IGNORECASE,
)
_LINE_CITE_B = re.compile(
    r"\blines?\s+(?P<a>\d{1,6})(?:\s*[-\u2013]\s*(?P<b>\d{1,6}))?\s+(?:of|in)\s+`?"
    rf"(?P<path>(?:[\w.-]{{1,80}}/){{0,12}}[\w-][\w.-]{{0,80}}\.(?:{EXTENSIONS}))`?",
    re.IGNORECASE,
)
_EDGE = re.compile(
    r"`([A-Za-z_$][\w$.]*)(?:\(\))?`\s*(?:\u2192|->|=>|calls|invokes|delegates\s+to|then\s+calls)\s*"
    r"`([A-Za-z_$][\w$.]*)(?:\(\))?`",
    re.IGNORECASE,
)
_DEFINED_IN_ANSWER = re.compile(
    r"\b(?:def|class|function|func|fn|interface|enum|struct|const|let|var)\s+([A-Za-z_$][\w$]*)"
)


def _last_component(name):
    return re.split(r"[.:#/]", name.replace("()", ""))[-1]


def _looks_like_code_name(raw):
    name = _last_component(raw)
    if not IDENT.match(name) or name in _COMMON_NAMES or len(name) < 4:
        return False
    call = raw.strip().endswith("()")
    snake = "_" in name.strip("_")
    camel = bool(re.search(r"[a-z][A-Z]", name)) or (name[0].isupper() and any(c.islower() for c in name)
                                                       and not name.isupper() and len(name) > 4)
    return call or snake or camel


def _word_in(name, text):
    return bool(re.search(r"(?<![A-Za-z0-9_$])" + re.escape(name) + r"(?![A-Za-z0-9_$])", text))


def _evidence_text(evidence):
    chunks, outlines = [], set()
    for record in evidence or []:
        if not isinstance(record, dict):
            continue
        if isinstance(record.get("source_text"), str):
            chunks.append(record["source_text"])
        for name in record.get("outline") or []:
            outlines.add(str(name))
    return "\n".join(chunks), outlines


def _line_totals(evidence, facts):
    totals = {}
    if facts and isinstance(facts.get("line_counts"), dict):
        totals.update(facts["line_counts"])
    for record in evidence or []:
        if isinstance(record, dict) and record.get("path") and isinstance(record.get("of"), int):
            totals.setdefault(record["path"], record["of"])
    return totals


def _match_total(path, totals):
    path = path.replace("\\", "/")
    if path in totals:
        return path, totals[path]
    base = "/" + path
    matches = [(p, t) for p, t in totals.items() if p.endswith(base) or p == path]
    if len(matches) == 1:
        return matches[0]
    return None, None


def claim_items(answer, user_message="", evidence=None, facts=None):
    """Every checkable code claim in an answer: {kind, text, supported, detail}.

    Kinds: symbol (a backticked function/class/variable name), line (a path:line citation) and
    edge (`A` calls `B`). A claim is only judged when there is something to judge it against.
    """
    items = []
    prose = prose_only(answer or "")[:MAX_ANALYZED_CHARS]
    source, outlines = _evidence_text(evidence)
    symbols = set((facts or {}).get("symbols") or [])
    can_judge_symbols = bool(symbols or source)
    own_definitions = set(_DEFINED_IN_ANSWER.findall(answer or ""))
    user_text = user_message or ""

    seen = set()
    for match in BACKTICK.finditer(prose):
        raw = match.group(1).strip()
        if FILE_NAME.fullmatch(raw) or " " in raw or not _looks_like_code_name(raw):
            continue
        name = _last_component(raw)
        if name in seen:
            continue
        seen.add(name)
        if not can_judge_symbols:
            continue
        supported = (
            name in symbols or name in outlines or name in own_definitions
            or _word_in(name, source) or _word_in(name, user_text)
        )
        items.append({"kind": "symbol", "text": name, "supported": supported,
                      "detail": "not defined in the project index and not seen in the code that was read"})

    totals = _line_totals(evidence, facts)
    cited = set()
    for pattern in (_LINE_CITE_A, _LINE_CITE_B):
        for match in pattern.finditer(prose):
            path = match.group("path")
            numbers = [int(g) for g in (match.group("a"), match.group("b")) if g]
            key = (path, tuple(numbers))
            if key in cited:
                continue
            cited.add(key)
            real, total = _match_total(path, totals)
            if total is None:
                continue
            bad = [n for n in numbers if n > total or n < 1]
            items.append({
                "kind": "line", "text": f"{path}:{'-'.join(map(str, numbers))}",
                "supported": not bad,
                "detail": f"`{real}` has only {total} lines",
            })

    edges = set((facts or {}).get("call_edges") or [])
    if edges and symbols:
        for match in _EDGE.finditer(prose):
            a, b = _last_component(match.group(1)), _last_component(match.group(2))
            if a in symbols and b in symbols:
                items.append({
                    "kind": "edge", "text": f"{a} -> {b}", "supported": (a, b) in edges,
                    "detail": "no such call was found in the project index (calls are matched by name, "
                              "so dynamic calls are invisible to it)",
                })
    return items


def _claim_notes(answer, user_message, evidence, facts):
    notes = []
    items = claim_items(answer, user_message, evidence, facts)
    for kind, label, sentence in (
        ("symbol", "symbols", "It names {names}, which could not be found in the project's definitions "
                              "or in any code that was read. They may be invented or renamed."),
        ("edge", "edges", "It says {names}, but {detail}."),
    ):
        bad = [i for i in items if i["kind"] == kind and not i["supported"]]
        if bad:
            names = ", ".join(f"`{i['text']}`" for i in bad[:5])
            notes.append(sentence.format(names=names, detail=bad[0]["detail"]))
    for item in items:
        if item["kind"] == "line" and not item["supported"]:
            notes.append(f"It cites {item['text']}, but {item['detail']}.")
    summary = summarize_items(items)
    if summary["checked"] >= 4 and summary["supported"] / summary["checked"] < 0.5:
        notes.append(
            f"Only {summary['supported']} of {summary['checked']} code references in this answer "
            "could be matched to the project. Treat the details with caution."
        )
    return notes


def summarize_items(items):
    checked = len(items)
    supported = sum(1 for i in items if i["supported"])
    return {
        "checked": checked,
        "supported": supported,
        "unsupported": [i["text"] for i in items if not i["supported"]][:10],
    }


def grounding_summary(answer, evidence, facts=None, user_message=""):
    """How much of an answer's checkable code claims is backed by the project and evidence."""
    try:
        return summarize_items(claim_items(answer, user_message, evidence, facts))
    except Exception:
        return {"checked": 0, "supported": 0, "unsupported": []}


def format_notes(notes):
    if not notes:
        return ""
    lines = ["", "", "> **Check this reply before trusting it**"]
    lines += [f"> - {note}" for note in notes]
    return "\n".join(lines)

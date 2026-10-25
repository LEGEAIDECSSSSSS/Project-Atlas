"""Check an answer against the evidence, and add a visible note when it does not hold up.

These are deterministic checks done by the backend, not instructions to the model.
They cannot prove an answer is right; they catch the specific failures that matter most:
naming files that do not exist, claiming to have read files that were never opened, and
claiming tests passed when no passing run exists.
"""
import re
from pathlib import PurePosixPath

from agent.evidence import files_read, last_test_status

EXTENSIONS = (
    "py|java|js|jsx|ts|tsx|go|rs|cs|cpp|hpp|c|h|php|rb|swift|kt|kts|scala|sql|html|css|vue|svelte|"
    "xml|json|ya?ml|toml|gradle|properties|md|txt|sh|bat|ps1"
)
FILE_NAME = re.compile(
    rf"(?<![\w./\\-])((?:[\w.-]+/)*[\w-][\w.-]*\.(?:{EXTENSIONS}))(?![\w])", re.IGNORECASE
)
FENCE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
FENCE_OPEN = re.compile(r"^```(.*)$", re.MULTILINE)

INSPECTION_CLAIM = re.compile(
    r"\b(?:I(?:'ve|\s+have)?|we(?:'ve|\s+have)?)\s+(?:just\s+|now\s+|also\s+)?"
    r"(?:read|reviewed|inspected|examined|opened|looked\s+(?:at|through)|checked|analy[sz]ed|"
    r"gone\s+through|went\s+through|scanned|studied)\b"
    r"|\bafter\s+(?:reading|reviewing|inspecting|examining|checking)\b",
    re.IGNORECASE,
)
TEST_CLAIM = re.compile(
    r"\b(?:all\s+(?:\d+\s+)?tests?\s+(?:now\s+)?pass(?:ed|es)?"
    r"|tests?\s+(?:now\s+|all\s+)?(?:pass(?:ed|es)?|are\s+(?:now\s+)?passing|succeed(?:ed)?)"
    r"|build\s+(?:succeeded|passes|passed|is\s+successful|was\s+successful))\b",
    re.IGNORECASE,
)
HEDGE = re.compile(
    r"\b(?:should|would|will|might|may|could|if|whether|once|until|so\s+that|to\s+(?:see|check|verify|confirm)|"
    r"make\s+sure|ensure|expect|hope|want|until|when)\b",
    re.IGNORECASE,
)


def _basename(path):
    return PurePosixPath(path.replace("\\", "/")).name.lower()


def prose_only(answer):
    return FENCE_BLOCK.sub(" ", answer)


def proposed_paths(answer):
    """Paths the answer puts on code fences: new or replacement files it is offering."""
    paths = set()
    for info in FENCE_OPEN.findall(answer):
        for token in info.replace(":", " ").split():
            token = token.strip("\"'`,;")
            if FILE_NAME.fullmatch(token):
                paths.add(token.replace("\\", "/"))
    return paths


def _unhedged(text, start):
    return not HEDGE.search(text[max(0, start - 40):start])


def check_answer(answer, project_files, user_message, evidence):
    """Return a list of short, plain-English problems found in the answer (maybe empty)."""
    if not project_files:
        return []

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
        exists = name in project_files or base in by_basename
        if exists:
            existing.append(name)
        elif base not in offered and base not in user_text:
            missing.append(name)

    if missing:
        names = ", ".join(f"`{n}`" for n in missing[:5])
        notes.append(
            f"It mentions {names}, which is not in the attached project. If it is proposing a "
            "new file that is fine; if it is describing existing code, that part is invented."
        )

    read = files_read(evidence)
    read_names = {_basename(p) for p in read}

    if INSPECTION_CLAIM.search(prose):
        unread = [n for n in existing if _basename(n) not in read_names]
        if unread:
            names = ", ".join(f"`{n}`" for n in unread[:5])
            notes.append(
                f"It talks about having looked at {names}, but it never opened "
                f"{'that file' if len(unread) == 1 else 'those files'}. Ask it to read "
                f"{'it' if len(unread) == 1 else 'them'} first before trusting the details."
            )
        elif not existing and not read:
            notes.append(
                "It says it reviewed the project, but it did not open any file in this "
                "conversation. Treat file-specific details as unverified."
            )

    claim = next((m for m in TEST_CLAIM.finditer(prose) if _unhedged(prose, m.start())), None)
    if claim:
        status, summary = last_test_status(evidence)
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

    return notes


def format_notes(notes):
    if not notes:
        return ""
    lines = ["", "", "> **Check this reply before trusting it**"]
    lines += [f"> - {note}" for note in notes]
    return "\n".join(lines)

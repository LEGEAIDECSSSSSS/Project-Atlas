"""Decide whether a message is a coding request or everyday chat."""
import re

CODE_WORDS = re.compile(
    r"\b("
    r"code|coding|function|method|bug|bugs|debug|compile|compiler|refactor|stack ?trace|"
    r"traceback|exception|syntax|regex|script|snippet|unit tests?|pytest|junit|maven|gradle|"
    r"npm|pip|git|docker|dockerfile|sql|database|schema|json|yaml|api|endpoint|backend|"
    r"frontend|react|vue|django|flask|fastapi|spring|python|javascript|typescript|java|"
    r"c\+\+|c#|golang|rust|php|html|css|bash|powershell|variable|loop|array|import|"
    r"algorithm|repo|repository|pull request|source file|\.py|\.js|\.java|\.ts"
    r")\b",
    re.IGNORECASE,
)

CODE_LINE = re.compile(
    r"^\s*(def |class |import |from \S+ import |function |const |let |var |public |private |"
    r"#include|using |package |return |if \(|for \(|while \(|<\w+[^>]*>|\}\s*$|.*;\s*$|.*=>.*)"
)


def looks_like_code(text: str) -> bool:
    if "```" in text:
        return True
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    hits = sum(1 for line in lines if CODE_LINE.match(line))
    return hits >= 3 and hits / len(lines) > 0.3


def detect_mode(message: str, has_project: bool, previous: str | None) -> str:
    if has_project or looks_like_code(message):
        return "code"

    if CODE_WORDS.search(message):
        return "code"

    # Short follow-ups ("make it shorter", "now add tests") stay in the current mode.
    if previous and len(message) < 300:
        return previous

    return "chat"


def resolve_mode(requested: str | None, message: str, has_project: bool, previous: str | None) -> str:
    if requested in ("chat", "code"):
        return requested
    return detect_mode(message, has_project, previous)

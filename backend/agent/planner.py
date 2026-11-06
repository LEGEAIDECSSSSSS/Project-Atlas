"""Turn a broad question about a project into a short, bounded investigation plan.

No model is involved. The plan is built from the words of the question: identifiers it names,
the plain words worth searching for (stop words and question phrasing removed), and a few
intent words ("starts", "endpoint", "who calls", "tests") that select extra tools. The agent
runs the plan itself, so a vague question like "what happens after a user signs in?" always
leads to real searches and real file reads instead of depending on the model to try.
"""
import re
from dataclasses import dataclass, field

from services import vocab

MAX_QUERIES = 2
MAX_IDENTIFIERS = 3
MAX_CONTENT_WORDS = 4

_IDENT_RE = re.compile(r"`?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`?(\(\))?")
_PATH_TOKEN_RE = re.compile(r"(?<![\w.])(/[A-Za-z0-9_\-./{}:]+)")
_FILE_LIKE = re.compile(r"\.[A-Za-z0-9]{1,5}$")


@dataclass
class Plan:
    queries: list = field(default_factory=list)       # search_project queries, best first
    symbols: list = field(default_factory=list)       # identifiers named in the question
    route_queries: list = field(default_factory=list)
    entry_points: bool = False
    callers_of: list = field(default_factory=list)
    tests_for: list = field(default_factory=list)
    intents: set = field(default_factory=set)
    words: list = field(default_factory=list)         # plain content words
    wants_tests: bool = False

    @property
    def is_empty(self):
        return not (self.queries or self.symbols or self.route_queries or self.entry_points)


def _identifiers(message):
    """Names that look like code: snake_case, camelCase/PascalCase, dotted names, name()."""
    found = []
    for match in _IDENT_RE.finditer(message or ""):
        token, call = match.group(1), match.group(2)
        if _FILE_LIKE.search(token) and "." in token and not call:
            continue                                       # a file name like agent.py
        base = token.split(".")[-1] if "." in token else token
        plain = ("_" in base.strip("_")) or bool(re.search(r"[a-z][A-Z]", base)) or bool(call)
        if plain and len(base) >= 3 and base.lower() not in vocab.STOPWORDS:
            if base not in found:
                found.append(base)
    return found


def _is_conceptual_word(word):
    return word in vocab.STOPWORDS or word in vocab.QUESTION_NOISE


MAX_PLANNED_CHARS = 4000


def make_plan(message):
    text = (message or "")[:MAX_PLANNED_CHARS]
    lowered_words = set(re.findall(r"[a-z]+", text.lower()))
    plan = Plan()

    plan.symbols = _identifiers(text)[:MAX_IDENTIFIERS]
    identifier_words = {w for s in plan.symbols for w in vocab.split_identifier(s)}

    symbol_names = {s.lower() for s in plan.symbols}
    words = [w for w in vocab.content_words(text)
             if w not in identifier_words and w not in symbol_names]
    plan.words = words[:MAX_CONTENT_WORDS + 2]

    for intent, intent_words in vocab.INTENT_WORDS.items():
        if lowered_words & intent_words:
            plan.intents.add(intent)

    plan.entry_points = "startup" in plan.intents
    plan.wants_tests = "tests" in plan.intents

    # words that only signalled the intent are not good search terms on their own
    signal_words = set()
    for intent in plan.intents:
        signal_words |= vocab.INTENT_WORDS[intent]
    topical = [w for w in plan.words if w not in signal_words] or plan.words

    route_paths = [m.group(1) for m in _PATH_TOKEN_RE.finditer(text) if len(m.group(1)) > 1]
    if "route" in plan.intents or route_paths:
        plan.route_queries = route_paths[:2] or [" ".join(topical[:3])] if (route_paths or topical) else []
        plan.route_queries = [q for q in plan.route_queries if q]

    if "callers" in plan.intents and plan.symbols:
        plan.callers_of = plan.symbols[:1]
    if "tests" in plan.intents and plan.symbols:
        plan.tests_for = plan.symbols[:1]

    primary = list(plan.symbols) + topical[:MAX_CONTENT_WORDS]
    if primary:
        plan.queries.append("|".join(primary[:MAX_IDENTIFIERS + MAX_CONTENT_WORDS]))
    if len(topical) >= 2:
        phrase = " ".join(topical[:3])
        if phrase not in plan.queries:
            plan.queries.append(phrase)
    elif plan.symbols:
        # a second, differently spelled search (login_user -> "login user"), so that finding
        # nothing is backed by two attempts rather than one
        phrase = " ".join(vocab.split_identifier(plan.symbols[0]))
        if phrase and phrase not in plan.queries and phrase != plan.queries[0].lower():
            plan.queries.append(phrase)
    plan.queries = plan.queries[:MAX_QUERIES]
    return plan

"""Deterministic project search: literal, identifier-aware, filename and synonym matching.

Ranking is explicit and explainable. Every result carries a score and the reasons for it, and
the whole result set is ordered by (score descending, path ascending), so the same query on
the same project always returns the same pages in the same order.

Stages, from most to least reliable:
  1. the exact text, or the exact identifier ("login_user", "LoginUser", "login user")
  2. all the words of the term, matched token-by-token ("login" inside loginUser)
  3. some of the words of a multi-word term
  4. code-vocabulary synonyms (login -> authenticate, session ...), used only when the
     literal stages find too little, so they never bury a precise answer
Filename and path matches, definition lines, and a lower weight for comments, tests, docs and
logs adjust the score.
"""
import re
import time
from dataclasses import dataclass, field

from services import vocab

MAX_MATCHES_STORED = 25          # lines remembered per file while scanning
MAX_MATCHES_SHOWN = 12           # lines returned per file
MIN_LITERAL_FILES = 3            # fewer literal hits than this -> synonyms are switched on
MAX_SYNONYMS_PER_WORD = 6
MAX_LINE_CHARS = 400             # longer lines (minified code) are shortened, not matched whole

DEFINITION_LINE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:public\s+|private\s+|protected\s+|static\s+|final\s+|"
    r"abstract\s+|async\s+)*(?:def|class|function|func|fn|interface|enum|record|struct|trait|"
    r"type|const|let|var|void|int|string|boolean)\b",
    re.IGNORECASE,
)
COMMENT_START = ("#", "//", "/*", "*", "<!--", "--", ";", "'''", '"""')
TEST_PATH = re.compile(
    r"(?:^|/)(?:tests?|spec|specs|__tests__|e2e)(?:/|$)|(?:^|/)test_[^/]*$|_test\.[a-z]+$|"
    r"\.(?:test|spec)\.[a-z]+$|(?:^|/)[^/]*Tests?\.java$",
    re.IGNORECASE,
)
LOW_VALUE_PATH = re.compile(
    r"(?:^|/)(?:logs?|docs?|vendor|third_party|fixtures|samples?|examples?)(?:/|$)|\.min\.|"
    r"(?:^|/)package-lock\.json$|(?:^|/)yarn\.lock$|\.lock$",
    re.IGNORECASE,
)


@dataclass
class Term:
    raw: str
    words: list
    exact: list
    synonyms: list = field(default_factory=list)
    is_identifier: bool = False

    @property
    def needles(self):
        return set(self.exact) | {w for w in self.words if len(w) >= 2}


def parse_query(query):
    """Split on '|' into search terms and work out the spellings and synonyms of each."""
    terms = []
    for raw in (query or "").split("|"):
        raw = raw.strip()
        if not raw:
            continue
        words = vocab.split_identifier(raw)
        if not words:
            continue
        identifier = (" " not in raw) and (
            "_" in raw.strip("_") or bool(re.search(r"[a-z][A-Z]", raw)) or "." in raw
            or "-" in raw
        )
        synonyms = []
        if not identifier:
            for word in words:
                if word in vocab.STOPWORDS:
                    continue
                synonyms.extend(vocab.synonyms_for(word)[:MAX_SYNONYMS_PER_WORD])
        seen, unique = set(words), []
        for synonym in synonyms:
            if synonym not in seen:
                seen.add(synonym)
                unique.append(synonym)
        terms.append(Term(
            raw=raw,
            words=words,
            exact=vocab.identifier_variants(raw),
            synonyms=unique[:12],
            is_identifier=identifier,
        ))
    return terms


def _needle_regex(terms, with_synonyms):
    needles = set()
    for term in terms:
        needles |= term.needles
        if with_synonyms:
            for synonym in term.synonyms:
                needles.add(synonym)
                needles.update(vocab.split_identifier(synonym))
    # one-character words (digits inside identifiers, 'x') match almost every line; they are
    # only used when nothing longer is available
    needles = {n for n in needles if len(n) >= 2} or {n for n in needles if n}
    if not needles:
        return None
    ordered = sorted(needles, key=lambda n: (-len(n), n))
    return re.compile("|".join(re.escape(n) for n in ordered))


def _is_comment(stripped, in_docstring):
    return in_docstring or stripped.startswith(COMMENT_START)


def _update_docstring_state(stripped, in_docstring):
    """Track Python triple-quoted strings so docstring lines count as comments."""
    count = stripped.count('"""') + stripped.count("'''")
    if count % 2 == 1:
        return not in_docstring
    return in_docstring


def _line_score(term, lower, token_set):
    """(weight, category, covered_words) of how well one line matches one term."""
    words = [w for w in term.words if w not in vocab.STOPWORDS] or term.words
    for variant in term.exact:
        if variant in lower:
            if re.search(r"(?<![a-z0-9_])" + re.escape(variant) + r"(?![a-z0-9_])", lower):
                return 5.0, "exact", list(words)
            return 3.0, "substring", list(words)

    token_set = token_set()
    covered = [w for w in words if w in token_set]
    if words and len(covered) == len(words):
        return 3.0, "words", covered
    if len(words) > 1 and len(covered) / len(words) >= 0.5:
        return 1.0 * len(covered) / len(words), "partial", covered
    if len(words) == 1 and words[0] in lower:
        return 1.5, "substring", list(words)
    return 0.0, None, []


def _synonym_score(term, token_set):
    """(weight, synonym, base_word_it_stands_for) for the first synonym present on the line."""
    token_set = token_set()
    words = [w for w in term.words if w not in vocab.STOPWORDS] or term.words
    for synonym in term.synonyms:
        parts = vocab.split_identifier(synonym)
        if parts and all(p in token_set for p in parts):
            base = next((w for w in words if synonym in vocab.synonyms_for(w)), words[0])
            return 0.8, synonym, base
    return 0.0, None, None


def scan_file(rel_path, text, terms, rx_literal, rx_all):
    """Per-term matches for one file. Returns dict term_index -> list of match dicts."""
    lower_text = text.lower()
    if not rx_all.search(lower_text):
        return {}

    results = {i: [] for i in range(len(terms))}
    in_docstring = False
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        was_docstring = in_docstring
        if rel_path.endswith((".py", ".pyi")):
            in_docstring = _update_docstring_state(stripped, in_docstring)
        if not stripped:
            continue
        lower = stripped.lower()
        if not rx_all.search(lower):
            continue
        cache = {}

        def token_set(stripped=stripped, cache=cache):
            """Word tokens of this line, computed only if a term needs them."""
            if "v" not in cache:
                cache["v"] = set(vocab.tokens_of(stripped[:MAX_LINE_CHARS * 3]))
            return cache["v"]

        comment = _is_comment(stripped, was_docstring or in_docstring)
        definition = bool(DEFINITION_LINE.match(stripped))
        for index, term in enumerate(terms):
            if len(results[index]) >= MAX_MATCHES_STORED:
                continue
            weight, category, covers = _line_score(term, lower, token_set)
            synonym = None
            if not category:
                weight, synonym, base = _synonym_score(term, token_set)
                category = "synonym" if synonym else None
                covers = [base] if synonym else []
            if not category:
                continue
            kind = "comment" if comment else ("definition" if definition and weight >= 3 else "code")
            results[index].append({
                "line": number,
                "content": stripped if len(stripped) <= MAX_LINE_CHARS
                else stripped[:MAX_LINE_CHARS] + "…",
                "matched_terms": [term.raw],
                "kind": kind,
                "match": category if not synonym else f"synonym:{synonym}",
                "covers": covers,
                "weight": weight * (0.35 if comment else 1.0) + (8.0 if kind == "definition" else 0.0),
            })
    return results


def _path_bonus(rel_path, terms):
    """Score and reasons from the file name and folders."""
    name = rel_path.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    stem_tokens = set(vocab.split_identifier(stem))
    dir_tokens = set()
    for part in rel_path.split("/")[:-1]:
        dir_tokens.update(vocab.split_identifier(part))
    bonus, reasons = 0.0, []
    for term in terms:
        compact = "".join(term.words)
        if stem.lower() in term.exact or stem.lower() == compact:
            bonus += 12
            reasons.append(f"file name is '{name}' ({term.raw})")
            continue
        words = [w for w in term.words if w not in vocab.STOPWORDS] or term.words
        if words and all(w in stem_tokens for w in words):
            bonus += 8
            reasons.append(f"file name contains '{term.raw}'")
        elif any(w in stem_tokens for w in words):
            bonus += 1.5
            reasons.append(f"file name shares a word with '{term.raw}'")
        elif words and all(w in dir_tokens or w in stem_tokens for w in words):
            bonus += 2
            reasons.append(f"path contains '{term.raw}'")
    return bonus, reasons


def _file_result(rel_path, per_term, terms, active_synonyms, wants_tests):
    matches = []
    for index, found in per_term.items():
        for match in found:
            if match["match"].startswith("synonym:") and not active_synonyms[index]:
                continue
            matches.append(match)

    bonus, reasons = _path_bonus(rel_path, terms)

    if not matches and not bonus:
        return None

    by_line = {}
    for match in matches:
        existing = by_line.get(match["line"])
        if existing is None:
            by_line[match["line"]] = dict(match)
        else:
            existing["matched_terms"] = sorted(set(existing["matched_terms"] + match["matched_terms"]))
            if match["weight"] > existing["weight"]:
                existing["weight"] = match["weight"]
                existing["kind"] = match["kind"]
                existing["match"] = match["match"]

    ordered = sorted(by_line.values(), key=lambda m: (-m["weight"], m["line"]))
    score = 0.0
    for position, match in enumerate(ordered[:20]):
        score += match["weight"] / (1 + 0.5 * position)
    score += bonus

    # A file that mentions several different words of a multi-word term (even on different
    # lines) is a better answer than one that repeats a single word.
    for index, term in enumerate(terms):
        words = [w for w in term.words if w not in vocab.STOPWORDS] or term.words
        if len(words) < 2:
            continue
        covered = set()
        for matches in (per_term.get(index) or []):
            if matches["match"].startswith("synonym:") and not active_synonyms[index]:
                continue
            covered.update(matches.get("covers", []))
        if len(covered) >= 2:
            score += 3.0 * (len(covered) - 1)
            reasons.append(f"covers {len(covered)} of {len(words)} words of '{term.raw}'")

    exact = sum(1 for m in ordered if m["match"] == "exact")
    defined = [m for m in ordered if m["kind"] == "definition"]
    if exact:
        reasons.append(f"{exact} exact match line{'s' if exact != 1 else ''}")
    if defined:
        reasons.append(f"defines it on line {defined[0]['line']}")
    synonym_hits = sorted({m["match"].split(":", 1)[1] for m in ordered if m["match"].startswith("synonym:")})
    if synonym_hits:
        reasons.append("matches related word(s): " + ", ".join(synonym_hits[:4]))
    if ordered and all(m["kind"] == "comment" for m in ordered):
        reasons.append("matches only in comments")
    other = len(ordered) - exact - len(defined)
    if other > 0 and not exact:
        reasons.append(f"{len(ordered)} matching line{'s' if len(ordered) != 1 else ''}")

    if TEST_PATH.search(rel_path) and not wants_tests:
        score *= 0.5
        reasons.append("test file (ranked lower)")
    if LOW_VALUE_PATH.search(rel_path):
        score *= 0.5
        reasons.append("docs/logs/generated path (ranked lower)")

    shown = sorted(ordered[:MAX_MATCHES_SHOWN], key=lambda m: m["line"])
    return {
        "path": rel_path,
        "score": round(score, 2),
        "reasons": reasons,
        "matches": [
            {"line": m["line"], "content": m["content"], "matched_terms": m["matched_terms"],
             "kind": m["kind"], "match": m["match"]}
            for m in shown
        ],
        "match_count": len(ordered),
        "filename_match": bonus > 0,
    }


def run_search(files, terms, limit, offset, time_budget, wants_tests=False, read_text=None, scan_after=None):
    """Search `files` (an iterable of (relative_path, absolute_path)) for `terms`."""
    rx_literal = _needle_regex(terms, with_synonyms=False)
    rx_all = _needle_regex(terms, with_synonyms=True)
    started = time.monotonic()
    if rx_all is None:                  # nothing searchable in the query
        return {"total_files": 0, "page": [], "scanned_files": 0, "scan_truncated": False,
                "synonyms_used": {}}

    per_file, scanned, truncated = {}, 0, False
    last_scanned = scan_after
    for rel_path, absolute in files:
        # Resume after the last fully considered path. _walk_files is deterministic and
        # lexically ordered, so this cursor avoids rescanning the timed-out prefix.
        if scan_after is not None and rel_path <= scan_after:
            continue
        if time.monotonic() - started > time_budget:
            truncated = True
            break
        scanned += 1
        last_scanned = rel_path
        text = read_text(absolute)
        if text is None:
            per_file_matches = {}
        else:
            per_file_matches = scan_file(rel_path, text, terms, rx_literal, rx_all)
        per_file[rel_path] = per_file_matches

    # A term's synonyms are only switched on when its literal matches are scarce.
    active = []
    for index, term in enumerate(terms):
        literal_files = 0
        for rel_path, found in per_file.items():
            strong = any(m["match"] in ("exact", "words", "substring") for m in found.get(index, []))
            if strong or _path_bonus(rel_path, [term])[0] >= 8:
                literal_files += 1
        active.append(bool(term.synonyms) and literal_files < MIN_LITERAL_FILES)

    results = []
    for rel_path, found in per_file.items():
        item = _file_result(rel_path, found, terms, active, wants_tests)
        if item:
            results.append(item)

    results.sort(key=lambda r: (-r["score"], r["path"]))
    total = len(results)
    page = results[offset: offset + limit]
    return {
        "total_files": total,
        "page": page,
        "scanned_files": scanned,
        "scan_truncated": truncated,
        "scan_cursor": last_scanned,
        "synonyms_used": {t.raw: t.synonyms for t, on in zip(terms, active) if on},
    }

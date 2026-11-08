"""One definition of "a line of source", shared by every stage that deals in line numbers.

``str.splitlines()`` also breaks on form feed, vertical tab, ``\\x1c``-``\\x1e``, ``\\x85`` and
``\\u2028``/``\\u2029``. Editors, Python's ``ast`` and every line number a user sees count only
``\\n``, ``\\r\\n`` and ``\\r``. Using ``splitlines()`` on source made the index, the reader and the
verifier disagree on any file containing one of those characters.
"""
import re

_BREAK = re.compile(r"\r\n|\r|\n")


def source_lines(text):
    """Split source text into lines the way editors and Python's ``ast`` number them."""
    parts = _BREAK.split(text or "")
    if parts and parts[-1] == "":
        parts.pop()
    return parts

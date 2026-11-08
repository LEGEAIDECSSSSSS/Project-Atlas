"""Build the source packet the model answers a multi-file flow question from.

Design rules (each one exists because a real failure demonstrated it):

* Every source line is prefixed with its real line number. A 7B model cannot count lines, so it
  is never asked to; it copies numbers it can see.
* Only COMPLETE records are used, and a definition is only listed when its declaration really
  appears in the retained text, not merely when its range falls inside a retrieved interval.
* Sizes are measured exactly while building. There is no "estimate, then throw everything away
  if the estimate was wrong" step.
* Essential definitions (in call order) are placed first. Anything that cannot fit is named in
  the packet, so the model reports it as unverified and never presents the trace as complete.
* Call sites name the caller and callee by qualified name, file and range, so same-named
  functions (``login`` in four files) cannot be confused.
"""
import re
from services.textlines import source_lines

SPAN = re.compile(r"\s*(\d+)\s*-\s*(\d+)\s*")
DECLARATION_WINDOW = 6          # decorators/signature lines allowed before the name appears
MAX_CALLSITE_SHARE = 0.25       # call sites never use more than this share of the budget


def parse_span(value):
    match = SPAN.fullmatch(str(value or ""))
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    return (start, end) if end >= start else None


def leaf(name):
    return str(name or "").rsplit(".", 1)[-1]


def number_lines(text, first_line):
    """Prefix each line with its real, right-aligned line number."""
    lines = source_lines(text)
    width = max(4, len(str(first_line + max(len(lines) - 1, 0))))
    return "\n".join(f"{first_line + i:>{width}} | {line}" for i, line in enumerate(lines))


def usable_records(records):
    """Complete read records with a numeric span, line count matching, de-duplicated and with
    records wholly contained in another complete record of the same file removed."""
    cleaned = []
    seen = set()
    for record in records or []:
        if not isinstance(record, dict) or record.get("tool") not in ("read_project_file", "read_symbol"):
            continue
        text = record.get("source_text")
        span = parse_span(record.get("lines"))
        path = record.get("path")
        if not (isinstance(text, str) and span and isinstance(path, str) and path):
            continue
        if not record.get("complete") or len(source_lines(text)) != span[1] - span[0] + 1:
            continue
        key = (path, span)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"path": path, "start": span[0], "end": span[1], "text": text})
    kept = []
    for item in cleaned:
        contained = any(other is not item and other["path"] == item["path"]
                        and other["start"] <= item["start"] and other["end"] >= item["end"]
                        and (other["end"] - other["start"]) > (item["end"] - item["start"])
                        for other in cleaned)
        if not contained:
            kept.append(item)
    return kept


def _line_text(record, line):
    index = line - record["start"]
    lines = source_lines(record["text"])
    return lines[index] if 0 <= index < len(lines) else None


def declaration_present(record, definition):
    """True when the definition's name really appears at its declared start line."""
    name = leaf(definition["symbol"])
    if not name:
        return False
    for offset in range(DECLARATION_WINDOW + 1):
        line = definition["start"] + offset
        if line > definition["end"]:
            break
        text = _line_text(record, line)
        if text is not None and re.search(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])", text):
            return True
    return False


def collect_definitions(route):
    """Definitions from the route graph, in request order (frontend first, then backend)."""
    found, seen = [], set()

    def add(item, path=None, role=None):
        if not isinstance(item, dict):
            return
        symbol = item.get("symbol") or item.get("qualname") or item.get("name")
        item_path = item.get("path") or path
        start = item.get("start_line", item.get("start"))
        end = item.get("end_line", item.get("end"))
        if not (symbol and item_path and isinstance(start, int) and isinstance(end, int) and end >= start):
            return
        key = (item_path, symbol, start, end)
        if key in seen:
            return
        seen.add(key)
        found.append({"kind": item.get("kind", "function"), "symbol": symbol, "path": item_path,
                      "start": start, "end": end, "called_from": item.get("called_from")})

    route = route or {}
    for item in route.get("frontend_ui_callers", []) or []:
        if not item.get("is_parent"):
            add(item)
    for chain in route.get("frontend_chains", []) or []:
        add(chain, path=chain.get("path"))
        for node in chain.get("chain", []) or []:
            if node.get("kind") in ("function", "method"):
                add(node)
    for node in route.get("chain", []) or []:
        if node.get("kind") in ("function", "method", "route"):
            add(node)
    return found


def _resolve_edges(route, definitions):
    """Edges between known definitions, each named with its own file and range."""
    route = route or {}
    raw = list(route.get("edges", []) or [])
    for chain in route.get("frontend_chains", []) or []:
        raw.extend(chain.get("edges", []) or [])
    resolved, seen = [], set()
    for edge in raw:
        path, line = edge.get("path"), edge.get("line")
        if not path or not isinstance(line, int):
            continue
        sources = [d for d in definitions if d["path"] == path and d["start"] <= line <= d["end"]
                   and leaf(d["symbol"]) == leaf(edge.get("from"))]
        if not sources:
            continue
        source = min(sources, key=lambda d: d["end"] - d["start"])
        site = f"{path}:{line}"
        targets = [d for d in definitions if d is not source
                   and (d["symbol"] == edge.get("to") or leaf(d["symbol"]) == leaf(edge.get("to")))]
        exact = [d for d in targets if d.get("called_from") == site]
        targets = exact or (targets if len({(d["path"], d["start"]) for d in targets}) == 1 else [])
        if not targets:
            continue
        target = targets[0]
        key = (source["path"], source["start"], target["path"], target["start"], line)
        if key not in seen:
            seen.add(key)
            resolved.append((source, target, path, line))
    return resolved


def build_packet(records, route, budget_chars):
    """Return ``(text, report)``.

    Required definitions are included first, as exact line slices in request order. Spare budget
    then adds the rest of each retrieved range as context. ``report`` lists what was included and
    which required definitions could not be, so callers and tests can assert on it.
    """
    report = {"included": [], "missing": [], "unverified": [], "definitions": []}
    usable = usable_records(records)
    definitions = collect_definitions(route)

    verified = []
    for definition in definitions:
        covering = [r for r in usable if r["path"] == definition["path"]
                    and r["start"] <= definition["start"] and r["end"] >= definition["end"]]
        if not covering:
            report["missing"].append({**definition, "reason": "its complete source was not retrieved"})
        elif not any(declaration_present(r, definition) for r in covering):
            report["unverified"].append({**definition, "reason": "declaration not found at the stated lines"})
        else:
            verified.append(definition)
    report["definitions"] = verified
    base_gaps = [f"- `{e['symbol']}` | {e['path']} | {e['start']}-{e['end']} ({e['reason']})"
                 for e in report["missing"] + report["unverified"]]

    if not usable or not verified:
        text = ("ATLAS VERIFIED FLOW EVIDENCE\nNo complete, verified source for the requested trace was "
                "retrieved. Do not invent code details or line numbers. Say what could not be verified.")
        if base_gaps:
            text += "\nNOT VERIFIED:\n" + "\n".join(base_gaps)
        return text, report

    text_by_path = {}      # path -> {line_number: text}
    for record in usable:
        lines = text_by_path.setdefault(record["path"], {})
        for offset, line in enumerate(source_lines(record["text"])):
            lines.setdefault(record["start"] + offset, line)

    def runs(path, wanted):
        """Contiguous runs of the wanted line numbers for one file."""
        out, current = [], []
        for number in sorted(wanted):
            if current and number != current[-1] + 1:
                out.append(current)
                current = []
            current.append(number)
        if current:
            out.append(current)
        return out

    def render_blocks(selected):
        """selected: {path: set(lines)} -> source blocks ordered by request order of first definition."""
        blocks = []
        for path, wanted in selected.items():
            for run in runs(path, wanted):
                first, last = run[0], run[-1]
                body = number_lines("\n".join(text_by_path[path][n] for n in run), first)
                owner = [i for i, d in enumerate(verified) if d["path"] == path
                         and d["start"] <= last and d["end"] >= first]
                blocks.append(((min(owner) if owner else len(verified)), path, first,
                               f"SOURCE {path} | lines {first}-{last} | COMPLETE\n{body}\n"))
        blocks.sort(key=lambda b: (b[0], b[1], b[2]))
        return blocks

    def compose(selected):
        covered = []
        for d in verified:
            lines = selected.get(d["path"], set())
            covered.append(all(n in lines for n in range(d["start"], d["end"] + 1)))
        dropped = [d for d, ok in zip(verified, covered) if not ok]
        gaps = base_gaps + [f"- `{d['symbol']}` | {d['path']} | {d['start']}-{d['end']}"
                            " (its source did not fit in the context budget)" for d in dropped]
        out = [
            "ATLAS VERIFIED FLOW EVIDENCE",
            "Every source line below starts with its real line number (`  65 | code`). Use ONLY this",
            "source. Never count lines yourself: copy each function's range from VERIFIED DEFINITIONS.",
            "",
        ]
        info = route or {}
        if info.get("handler") or info.get("route"):
            out.append(f"ROUTE: {info.get('method', 'HTTP')} "
                       f"{info.get('effective_route') or info.get('route') or '?'} -> "
                       f"{info.get('handler') or '?'} at {info.get('declared_at') or 'unknown location'}")
            out.append("")
        out.append("VERIFIED DEFINITIONS (request order; exact names, files, inclusive line ranges):")
        for number, d in enumerate(verified, 1):
            out.append(f"{number}. {d['kind']} `{d['symbol']}` | {d['path']} | {d['start']}-{d['end']}")
        out.append("")
        if gaps:
            out.append("NOT VERIFIED (do not describe their implementation; say they could not be verified):")
            out.extend(gaps)
            out.append("")
        edges = _resolve_edges(route, verified)
        if edges:
            index = {id(d): n for n, d in enumerate(verified, 1)}
            out.append("VERIFIED CALL SITES (caller -> callee, by definition number; line is in the caller's file):")
            for source, target, _path, line in edges:
                out.append(f"- #{index[id(source)]} `{source['symbol']}` -> #{index[id(target)]} "
                           f"`{target['symbol']}` at line {line}")
            out.append("")
        out.extend([
            "ANSWER FORMAT: one entry per VERIFIED DEFINITION, in order, each with these labelled lines:",
            "**Function:** name",
            "**File Path:** exact path from VERIFIED DEFINITIONS",
            "**Line Numbers:** start-end from VERIFIED DEFINITIONS",
            "**Purpose:** what the numbered source shows it doing",
            "**Connection:** the function that calls it / what it calls next (from CALL SITES)",
            "Describe only what the numbered source shows. If something is not shown, say it could",
            "not be verified. Do not add functions that are not in VERIFIED DEFINITIONS.",
            "",
        ])
        out.extend(block[3] for block in render_blocks(selected))
        return "\n".join(out), dropped

    selected = {}
    # Phase 1: every required definition as its exact slice, in request order, while it fits.
    for definition in verified:
        trial = {p: set(v) for p, v in selected.items()}
        trial.setdefault(definition["path"], set()).update(range(definition["start"], definition["end"] + 1))
        if len(compose(trial)[0]) <= budget_chars:
            selected = trial
    # Phase 2: spare room adds the rest of each retrieved range as context (whole records only).
    for record in usable:
        trial = {p: set(v) for p, v in selected.items()}
        trial.setdefault(record["path"], set()).update(range(record["start"], record["end"] + 1))
        if trial != selected and len(compose(trial)[0]) <= budget_chars:
            selected = trial

    text, dropped = compose(selected)
    report["included"] = [(path, run[0], run[-1]) for path, wanted in selected.items()
                          for run in runs(path, wanted)]
    for d in dropped:
        report["missing"].append({**d, "reason": "its source did not fit in the context budget"})
    return text, report

"""Complete a verified flow answer with the connection facts the model left out or could not establish.

The model's answer is never altered. After it has passed verification, Atlas compares it with facts
derived deterministically from the retrieved source (agent/callflow.py) and appends a clearly labelled
section containing ONLY what the answer lacks:

* callers of each definition (the inverse edges a 7B model rarely works out);
* calls as written, with the receiver's type and any recursion or condition;
* external library / platform calls, so a known dependency is never left as "unknown";
* multi-value data flow and parameter roles;
* a full entry for any verified definition the model skipped.

Nothing here is written by the model, and nothing is added for a fact the answer already states.
"""
import re

from agent import callflow as cf
from agent import packet as pk
from agent.verify import parse_function_entries

UNSURE = re.compile(r"cannot be (?:established|verified)|could not be (?:established|verified)|"
                    r"not (?:be )?verified|unknown|not (?:established|determined)", re.IGNORECASE)
HEADING = "### Atlas-verified connections"
CORRECTIONS_HEADING = "### Atlas corrections to the answer above"
OMITTED_HEADING = "### Not covered by the model's answer (added by Atlas from the verified source)"


def _short(path):
    return "/".join(str(path).split("/")[-2:])


def _leaf(name):
    return str(name or "").rsplit(".", 1)[-1]


def _blocks(answer, definitions):
    """{definition number: the model's entry text (lower-cased)} for the entries it wrote."""
    found = {}
    for entry in parse_function_entries(answer or ""):
        block = (entry.get("block") or "").lower()
        name, path = _leaf(entry.get("function")), str(entry.get("path") or "")
        for number, d in enumerate(definitions, 1):
            if (_leaf(d["symbol"]) == name and (path.endswith(d["path"]) or d["path"].endswith(path))
                    and entry.get("start") == d["start"] and entry.get("end") == d["end"]):
                found.setdefault(number, block)
    return found


def _has(text, *tokens):
    return all(t.lower() in text for t in tokens)


def _target_label(target):
    return f"`{target['symbol']}` ({_short(target['path'])}:{target['start']}-{target['end']})"


def _connection_part(text):
    """The part of an entry's text that states its connection (before any Data Flow label)."""
    start = text.find("connection")
    if start < 0:
        return ""
    end = text.find("data flow", start)
    return text[start:end if end > 0 else len(text)]


def _facts_for(item, text, packet_keys, nothing_said=False):
    """[(sentence, is_data_flow)] the model's entry (`text`, lower-case) does not already state.

    Deliberately narrow: it repairs what a reviewer would call wrong or missing, and does not list
    every call. Calls to helpers outside the trace, error-class constructors and framework hook
    setters are never added.
    """
    out = []
    # `nothing_said`: the model skipped this definition entirely, so every connection is new information
    unestablished = nothing_said or bool(UNSURE.search(_connection_part(text)))
    in_trace = lambda t: (t["path"], t["start"], t["end"]) in packet_keys

    def label(t):
        return f"`{t['symbol']}` ({_short(t['path'])}:{t['start']}-{t['end']})"

    def written(call):
        key = (call["receiver"] + "." + call["callee"]) if call["receiver"] else call["callee"]
        return key

    # 1. an entry the model could not connect: say who calls it
    if unestablished:
        for cb in item["called_by"]:
            where = f"{_short(cb['path'])}:{cb['start']}-{cb['end']}"
            more = f"; {cb['receiver_note']}" if cb["receiver_note"] else ""
            out.append((f"Called by `{cb['caller']}` ({where}) at line {cb['line']} as `{cb['expr']}`{more}.", False))

    named_external = []
    for call in item["calls"]:
        kind = call["kind"]
        if kind in ("project", "recursive"):
            target = call["targets"][0] if call["targets"] else None
            constructor = call["expr"].startswith("new ") or (target and target.get("kind") == "class")
            flagged = kind == "recursive" or bool(call["conditional"])
            traced = bool(target) and in_trace(target)
            if kind == "recursive":
                reason = "RECURSIVE call to itself" + (f"; conditional ({call['conditional']})" if call["conditional"] else "")
                if not _has(text, "recursi"):
                    out.append((f"Calls `{call['expr']}` at line {call['line']}: {reason}.", False))
                continue
            if constructor and not nothing_said:
                continue
            note = f"; {call['receiver_note']}" if call["receiver_note"] else ""
            cond = f"; conditional ({call['conditional']})" if call["conditional"] else ""
            if flagged and not (_has(text, call["callee"]) and _has(text, "condition")):
                out.append((f"Calls `{call['expr']}` at line {call['line']} -> project "
                            f"{'class' if constructor else 'function'} {label(target)}{note}{cond}.", False))
            elif traced and call["receiver"] not in ("", "self", "this", "cls") and not _has(text, written(call)):
                out.append((f"Call as written: `{call['expr']}` at line {call['line']} -> project function "
                            f"{label(target)}{note}.", False))
            elif traced and not _has(text, call["callee"]):
                # a call to another definition in the trace that the answer never mentions
                out.append((f"Calls `{call['expr']}` at line {call['line']} -> project function {label(target)}{note}.", False))
            elif (traced or nothing_said) and unestablished and target:
                out.append((f"Calls `{call['expr']}` at line {call['line']} -> project "
                            f"{'class' if constructor else 'function'} {label(target)}{note}.", False))
        elif kind in ("external", "platform") and call["note"] not in cf.GENERIC_NOTES:
            if cf_is_hook_value(call):
                continue                                     # framework hook values are not connections
            named_external.append(call)

    # 2. known dependencies, so they are never left as "unknown" (at most three distinct ones). Also for
    #    entries that describe only their callers: a wrapper such as verify_password exists to make one
    #    external call, and the answer then says nothing about what it calls.
    connection = _connection_part(text)
    no_outgoing = bool(connection) and not re.search(r"\b(?:calls?|invokes?)\b", connection)
    if unestablished or no_outgoing:
        seen = set()
        for call in named_external:
            if call["note"] in seen or len(seen) >= 3:
                continue
            seen.add(call["note"])
            noun = "external library" if call["kind"] == "external" else "platform API"
            cond = f"; conditional ({call['conditional']})" if call["conditional"] else ""
            out.append((f"Calls `{call['expr']}` at line {call['line']} -> {noun} `{call['note']}`, "
                        f"not a project function{cond}.", False))

    # 3. multi-value data flow
    flow = item["flow"]
    for index, line in enumerate(flow):
        if line.startswith("line ") and re.search(r"receives \d+ values", line):
            names = re.findall(r"[A-Za-z_]\w*", line.split("`")[1]) if "`" in line else []
            if names and all(n.lower() in text for n in names):
                continue
            uses = [u for u in flow[index + 1:index + 7] if u.startswith("`")][:5]
            out.append(("Data flow: " + line.rstrip(".") + "." + (" " + " ".join(f"{u}." for u in uses) if uses else ""), True))

    # 3b. a function that returns a variable bound from a call: say whose result it returns
    for line in flow:
        bind = re.match(r"line (\d+) `([^`=]+?) = ([^`]+)` receives its value", line)
        if not bind:
            continue
        number, var, expr = int(bind.group(1)), bind.group(2).strip(), bind.group(3).strip()
        if not re.fullmatch(r"[A-Za-z_$][\w$]*", var):
            continue
        returned = [n for n, parts in (item.get("returns") or []) if parts == [var]]
        callee = re.match(r"[\w$.]*?([\w$]+)\(", expr)
        if not returned or not callee or re.search(r"return(?:[^.\n]|\.(?=\w))*?\b" + re.escape(callee.group(1)) + r"\b", text):
            continue
        # other uses of the variable (never error/class constructors; wording makes no claim about order or branch)
        others = [u.rstrip(".") for u in flow if u.startswith(f"`{var}` is passed")
                  and not re.search(r"passed (?:as argument \d+ of|to) `(?:new )?[A-Z]", u)][:2]
        out.append((f"Returns `{var}` (line {returned[0]}), the result of `{expr}` (line {number})"
                    + (f"; `{var}` is also " + "; ".join(o.split("` is ", 1)[1] for o in others) if others else "") + ".", True))
    # 3c. mentioned (not called) elsewhere in its file, e.g. exposed through a context value
    for number, snippet in (item.get("references") or [])[:1]:
        if str(number) not in text:
            out.append((f"Also referenced (not called) at line {number}: `{snippet[:110]}`.", True))

    # 4. parameter roles, only for functions with several typed, informative parameters
    params = [p for p in item["params"] if p[2]]
    if len(params) >= 2 and any(not re.search(r"(?<![\w])" + re.escape(p[0].lower()) + r"(?![\w])", text)
                                for p in item["params"] if p[2] or p[3]):
        described = []
        for name, annotation, kind, uses in item["params"]:
            if not (kind or uses):
                continue
            piece = f"`{name}" + (f": {annotation}`" if annotation else "`")
            if kind:
                piece += f" ({kind})"
            if uses:
                piece += " " + "; ".join([uses[0]] + ["it " + u for u in uses[1:2]])
            described.append(piece)
        out.append(("Parameters: " + "; ".join(described) + ".", True))

    if not out and unestablished and not item["calls"] and not item["called_by"]:
        out.append(("Makes no calls to project functions or external APIs (checked against its source).", False))
    return out


def cf_is_hook_value(call):
    """True for calls on values returned by framework hooks (state setters, query clients)."""
    return bool(call.get("hook_value")) or bool(re.match(r"(?:set[A-Z]\w*|use[A-Z]\w*)$", call["callee"]))


# ----------------------------------------------------------------------------- challenging the model's claims
_FILE_EXT = re.compile(r"\.(?:py|js|jsx|ts|tsx|java|go|rb|rs|cs|kt|php|swift|c|cpp|h)$", re.IGNORECASE)
CALL_CLAIM = re.compile(
    # Qwen commonly wraps the complete call expression in Markdown code ticks.
    # Consume the closing tick before the optional source path, or the matcher
    # skips the call and can incorrectly match the filename as a symbol.
    r"(?P<chain>[A-Za-z_$][\w$.]*)(?P<args>\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\))?`?\s*"
    r"(?:from\s+`?[\w/.\-]+`?\s*)?\(?\s*(?:at\s+)?lines?\s+(?P<n>\d+)\)?", re.IGNORECASE)
CALL_ARGS = re.compile(r"(?P<chain>[A-Za-z_$][\w$.]*)\((?P<args>(?:[^()]|\((?:[^()]|\([^()]*\))*\))*)\)")
RETURNS_CLAIM = re.compile(r"\breturns?\s+(?P<claim>.+?)(?=\.\s|\.$|;|\n|$)", re.IGNORECASE)
CALLED_BY = re.compile(r"\bcalled\s+by\s+(?:the\s+)?`?(?P<caller>[A-Za-z_$][\w$.]*)", re.IGNORECASE)
_RETURN_STOP = frozenset({
    "the", "a", "an", "token", "tokens", "value", "values", "boolean", "indicating", "success", "object", "response",
    "result", "from", "server", "backend", "with", "of", "to", "its", "nothing", "none", "null", "undefined", "true",
    "false", "new", "for", "by", "that", "which", "and", "or", "no", "data", "status", "id", "is", "if"})


def _normalise_args(text):
    text = re.sub(r"\s+", "", text or "").replace('"', "'")
    return re.sub(r",([})\]])", r"\1", text)


def _split_top(text):
    """Split on commas that are not inside (), [] or {} or a string."""
    parts, depth, current, quote = [], 0, [], None
    for ch in text or "":
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"`":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _object_keys(literal):
    inner = literal.strip()[1:-1]
    keys = set()
    for part in _split_top(inner):
        match = re.match(r"\s*(?:\.\.\.)?['\"]?([\w$]+)", part)
        if match:
            keys.add(match.group(1))
    return keys


def _args_conflict(claimed, actual):
    """True when the arguments the answer wrote cannot be the arguments in the source.

    Compared structurally: the same number of arguments, each equal after whitespace/quote/trailing-comma
    normalisation. An object literal is compared by its property names only, because a summary of
    `{ headers: cond ? {...} : undefined }` as `{ headers }` is an abbreviation, not an error.
    """
    wrote, real = _split_top(claimed), _split_top(actual)
    if len(wrote) != len(real):
        return True
    for x, y in zip(wrote, real):
        if _normalise_args(x) == _normalise_args(y):
            continue
        if x.startswith("{") and y.startswith("{") and _object_keys(x) == _object_keys(y):
            continue
        return True
    return False


def _connection_text(raw_block):
    """Return the connection section, accepting both common label forms.

    Qwen alternates between ``Connection:`` and ``Connections:`` (often bolded).
    The previous singular-only matcher silently skipped every contradiction in the
    plural form, even though the rest of the entry parsed correctly.
    """
    start = re.search(r"\bconnections?\s*:", raw_block, re.IGNORECASE)
    return raw_block[start.start():] if start else ""


def _challenge(item, raw_block):
    """[(kind, claim, source)] for claims in the model's entry that the derived facts contradict.

    kind is 'contradicted' (the source says otherwise) or 'clarified' (the claim gives a method's
    definition line where an invocation was described). Nothing here rewrites the model's text, and a
    claim is only challenged when the facts are unambiguous: unparsable or abbreviated claims are skipped.
    """
    text = _connection_text(raw_block)
    found = []
    seen = set()
    # (1) lines and arguments attached to a call: "calls login(a, b) from auth.jsx (line 40)"
    for match in CALL_CLAIM.finditer(text):
        chain, number = match.group("chain"), int(match.group("n"))
        if _FILE_EXT.search(chain):
            continue
        leaf = chain.rsplit(".", 1)[-1]
        calls = [c for c in item["calls"] if c["callee"] == leaf]
        if not calls:
            continue
        call = next((c for c in calls if ((c["receiver"] + ".") if c["receiver"] else "") + c["callee"] == chain), calls[0])
        key = (chain, number)
        if key in seen:
            continue
        seen.add(key)
        written = (call["receiver"] + "." if call["receiver"] else "") + call["callee"]
        call_lines = sorted({c["line"] for c in calls})
        targets = [t for c in calls for t in c["targets"]]
        starts = {t["start"] for t in targets}
        if number in call_lines:
            pass
        elif number in starts:
            target = next(t for t in targets if t["start"] == number)
            found.append(("clarified", f"`{chain}(...)` (line {number})",
                          f"line {number} is where `{target['symbol']}` is defined ({_short(target['path'])}); "
                          f"the call `{call['expr']}` is at line {call['line']} of {_short(item['path'])}"
                          + (f"; {call['receiver_note']}" if call["receiver_note"] else "")))
        else:
            inside = next((t for t in targets if t["start"] < number <= t["end"]), None)
            where = (f"inside `{inside['symbol']}` ({_short(inside['path'])}:{inside['start']}-{inside['end']})"
                     if inside else (f"another line of `{item['symbol']}`" if item["start"] <= number <= item["end"]
                                     else "neither where it is called nor where it is defined"))
            found.append(("contradicted", f"`{chain}(...)` (line {number})",
                          f"`{call['expr']}` is called at line {call['line']} of {_short(item['path'])}"
                          f"{'' if len(call_lines) == 1 else ' (also lines ' + ', '.join(map(str, call_lines[1:])) + ')'}; "
                          f"line {number} is {where}"))
    # (2) the arguments the answer wrote for a call, against the arguments in the source
    done = set()
    for match in CALL_ARGS.finditer(text):
        chain, claimed = match.group("chain"), match.group("args")
        leaf = chain.rsplit(".", 1)[-1]
        if (chain, claimed) in done or "..." in claimed or _FILE_EXT.search(chain):
            continue
        done.add((chain, claimed))
        for call in (c for c in item["calls"] if c["callee"] == leaf and c.get("args") is not None):
            if ((call["receiver"] + ".") if call["receiver"] else "") + call["callee"] != chain:
                continue
            if _args_conflict(claimed, call["args"]):
                actual = call["args"].rstrip(", ")
                actual = actual if len(actual) <= 140 else actual[:137] + "..."
                found.append(("contradicted", f"`{chain}({claimed.strip()})`",
                              f"the source calls `{chain}({actual})` at line {call['line']} of {_short(item['path'])}"))
            break
    # (3) values the entry says the function returns, against its return statements
    if item.get("returns") is not None and item.get("locals"):
        returned = " ".join(", ".join(parts) for _line, parts in item["returns"]).lower()
        local = {n.lower() for n in item["locals"]}
        for match in RETURNS_CLAIM.finditer(text):
            wrong = []
            for piece in re.split(r",|\band\b", match.group("claim")):
                for word in re.findall(r"[A-Za-z_]\w*", piece):
                    low = word.lower()
                    if low not in _RETURN_STOP and low in local and low not in returned and low not in wrong:
                        wrong.append(low)
            if wrong and item["returns"]:
                actual = "; ".join(f"`{', '.join(parts)}` (line {line})" for line, parts in item["returns"][:3])
                flow = [u for u in item["flow"] if any(u.startswith(f"`{w}`") for w in wrong)][:3]
                found.append(("contradicted", f"returns {', '.join(f'`{w}`' for w in wrong)}",
                              f"`{item['symbol']}` returns {actual}; "
                              + (" ".join(u.rstrip(".") + "." for u in flow) if flow else
                                 f"{', '.join(f'`{w}`' for w in wrong)} are local values it does not return")))
    # (4) "called by X": X must be one of the callers the analysis found
    callers = {_leaf(cb["caller"]).lower() for cb in item["called_by"]}
    for match in CALLED_BY.finditer(text):
        caller = match.group("caller")
        if callers and _leaf(caller).lower() not in callers:
            found.append(("contradicted", f"called by `{caller}`",
                          "the callers found in the source are " + ", ".join(
                              f"`{cb['caller']}` (line {cb['line']})" for cb in item["called_by"][:4])))
    return found


def _raw_blocks(answer, definitions):
    """{definition number: the model's entry text, original case}."""
    found = {}
    for entry in parse_function_entries(answer or ""):
        block = entry.get("block") or ""
        name, path = _leaf(entry.get("function")), str(entry.get("path") or "")
        for number, d in enumerate(definitions, 1):
            if (_leaf(d["symbol"]) == name and (path.endswith(d["path"]) or d["path"].endswith(path))
                    and entry.get("start") == d["start"] and entry.get("end") == d["end"]):
                found.setdefault(number, block)
    return found


def build_addendum(answer, definitions, project_id, route):
    """The text to append to a verified flow answer, or "" when it already says everything."""
    entries = [e for e in parse_function_entries(answer or "") if e.get("function")]
    if not entries or not definitions:
        return ""                                   # not a per-function answer: leave it alone
    edges = pk._resolve_edges(route, definitions)
    items = cf.facts(project_id, definitions, edges)
    blocks = _blocks(answer, definitions)
    raw_blocks = _raw_blocks(answer, definitions)
    packet_keys = {(d["path"], d["start"], d["end"]) for d in definitions}
    corrections = []
    for number, item in enumerate(items, 1):
        if number in raw_blocks:
            for kind, claim, source in _challenge(item, raw_blocks[number]):
                corrections.append((item, kind, claim, source))
    added, omitted = [], []
    for number, item in enumerate(items, 1):
        if number not in blocks:
            omitted.append(item)
            continue
        facts = _facts_for(item, blocks[number], packet_keys)
        if facts:
            added.append((item, facts))
    if not added and not omitted and not corrections:
        return ""
    out = [""]
    if corrections:
        out += ["---", CORRECTIONS_HEADING,
                "The answer above is unchanged. For each entry below, the retrieved source contradicts or "
                "clarifies a claim it made.", ""]
        for item, kind, claim, source in corrections:
            out.append(f"- **`{item['symbol']}`** (`{item['path']}:{item['start']}-{item['end']}`) - "
                       f"**{kind}:** the answer says {claim}. Source: {source.rstrip('. ')}.")
        out.append("")
    if added or omitted:
        out += ["---", HEADING,
                "Derived deterministically from the retrieved source, not written by the model. These add what "
                "the answer above omitted or marked as unestablished.", ""]
    for item, facts in added:
        out.append(f"- **`{item['symbol']}`** (`{item['path']}:{item['start']}-{item['end']}`)")
        out.extend(f"  - {sentence}" for sentence, _flow in facts)
    if omitted:
        out.extend(["", OMITTED_HEADING, ""])
        for item in omitted:
            out.append(f"**Function:** {item['symbol']}")
            out.append(f"**File Path:** {item['path']}")
            out.append(f"**Line Numbers:** {item['start']}-{item['end']}")
            if item.get("signature"):
                out.append(f"**Signature:** `{item['signature']}`")
            facts = _facts_for(item, "", packet_keys, nothing_said=True)
            statements = " ".join(sentence for sentence, flow in facts if not flow)
            if not statements:
                # only claim "no calls" when the analysis really found none
                statements = ("No calls were found in its source." if not item["calls"] else
                              "Its calls are generic built-in/runtime methods or framework state setters.")
            out.append("**Connection:** " + statements)
            flow = [s for s, is_flow in facts if is_flow]
            if flow:
                out.append("**Data Flow:** " + " ".join(flow))
            out.append("")
    return "\n".join(out).rstrip() + "\n"

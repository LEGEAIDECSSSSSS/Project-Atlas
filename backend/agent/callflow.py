"""Deterministic call analysis for the definitions in a flow packet.

For every verified definition this derives, from the real source (never from the model):

* each call as it is WRITTEN (``self.sessions.create(...)``), kept apart from the definition it
  resolves to (``SessionRepository.create``);
* what the callee is: a project function, an external library, a platform/built-in API, a
  recursive call to itself, or genuinely not establishable from the source;
* whether the call is conditional (inside ``if``/``except``/short-circuit/ternary) and why;
* data flow: which values a call returns, which names receive them, and where each goes next.

Nothing here knows any project's file or function names. "Project function" comes from the code
index's own resolver; "external" comes from the file's imports and annotations; "platform" from
language-level built-in vocabularies. A call is only reported as unknown when none of those
establishes what it is, and the reason says so.
"""
import ast
import builtins
import re
import sys
from dataclasses import replace

from services import code_index
from services import project_service
from services.textlines import source_lines

JS_GLOBALS = {
    "fetch", "window", "document", "console", "JSON", "Math", "Object", "Array", "Promise", "localStorage",
    "sessionStorage", "setTimeout", "setInterval", "clearTimeout", "clearInterval", "URL", "URLSearchParams",
    "FormData", "Date", "Number", "String", "Boolean", "parseInt", "parseFloat", "navigator", "location",
    "history", "Error", "Map", "Set", "WeakMap", "Symbol", "Intl", "Headers", "Request", "Response",
    "AbortController", "structuredClone", "encodeURIComponent", "decodeURIComponent", "atob", "btoa",
    "queueMicrotask", "requestAnimationFrame", "Buffer", "process", "require",
}
JS_BUILTIN_METHODS = {
    "startsWith", "endsWith", "includes", "indexOf", "slice", "splice", "split", "join", "map", "filter",
    "reduce", "forEach", "find", "some", "every", "push", "pop", "shift", "unshift", "concat", "keys",
    "values", "entries", "trim", "toLowerCase", "toUpperCase", "replace", "match", "toString", "then",
    "catch", "finally", "json", "text", "blob", "get", "set", "has", "delete", "add", "append", "stringify",
    "parse", "assign", "freeze", "isArray", "from", "of", "resolve", "reject", "all", "now", "padStart",
    "localeCompare", "sort", "flat", "flatMap", "preventDefault", "stopPropagation", "getItem", "setItem",
    "removeItem", "addEventListener", "removeEventListener", "querySelector", "querySelectorAll",
}
PY_BUILTIN_METHODS = {
    "get", "items", "keys", "values", "append", "extend", "pop", "update", "join", "format", "split",
    "strip", "lstrip", "rstrip", "lower", "upper", "startswith", "endswith", "replace", "encode", "decode",
    "add", "remove", "discard", "setdefault", "copy", "sort", "index", "count", "hexdigest", "isoformat",
}
PY_BUILTINS = set(dir(builtins))
NOT_CALLS = {"if", "for", "while", "switch", "catch", "function", "return", "typeof", "await", "async",
             "new", "else", "with", "elif", "except", "lambda", "yield", "print_"}
CONDITION_WORDS = re.compile(r"\b(if|else|catch|switch|case|for|while)\b")
STRIP_LITERALS = re.compile(r"""'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|`(?:\\.|[^`\\])*`|//.*$|/\*.*?\*/""")
MAX_SHOWN_ARGS = 26


def _short_path(path):
    parts = str(path).split("/")
    return "/".join(parts[-2:])


def _a(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def _leaf(name):
    return str(name or "").rsplit(".", 1)[-1]


def _blank_literals(line):
    """Replace string/comment contents with spaces so columns still line up."""
    return STRIP_LITERALS.sub(lambda m: " " * len(m.group(0)), line)


# ---------------------------------------------------------------------------- source facts
class FileFacts:
    """Imports, module-level instances and the full text of one project file."""

    def __init__(self, index, path, text):
        self.index, self.path, self.text = index, path, text
        self.lines = source_lines(text)
        info = index.files.get(path)
        self.lang = getattr(info, "lang", "") if info else ""
        self.is_python = path.endswith(".py")
        self.imported = {}                 # local name -> (module, external?)
        resolved = dict((id(i), t) for i, t in index.imports_resolved.get(path, []))
        for imp, targets in index.imports_resolved.get(path, []):
            external = (not targets) and not self._is_project_module(imp.module)
            for name in imp.names or [imp.module]:
                self.imported[name] = (imp.module, external)
        self.tree = None
        if self.is_python:
            try:
                self.tree = ast.parse(text)
            except SyntaxError:
                self.tree = None
        self.instances = self._module_instances()
        self.bindings = self._call_bindings()

    def _is_project_module(self, module):
        first = str(module).lstrip("./@~").split("/")[0].split(".")[0]
        if not first or str(module).startswith((".", "@/", "~/")):
            return True
        for path in self.index.all_paths:
            parts = path.split("/")
            if first in parts[:-1] or parts[-1].split(".")[0] == first:
                return True
        return False

    def _call_bindings(self):
        """name -> root callee for ``const [a, setA] = useState()`` / ``const x = useFoo()`` anywhere
        in a JS file, so values returned by imported hooks are traced to their import."""
        found = {}
        if self.is_python:
            return found
        pattern = re.compile(r"(?:const|let|var)\s+(\{[^}]*\}|\[[^\]]*\]|[\w$]+)\s*=\s*(?:await\s+)?(?:new\s+)?([\w$]+)(?:\.[\w$]+)*\s*\(")
        for line in self.lines:
            match = pattern.search(line)
            if match:
                for name in re.findall(r"[\w$]+", match.group(1)):
                    found.setdefault(name, match.group(2))
        return found

    def _module_instances(self):
        """Module-level ``NAME = Callee(...)``: NAME is an instance of Callee."""
        found = {}
        if self.tree is not None:
            for node in self.tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    root = _call_root(node.value.func)
                    for target in node.targets:
                        if isinstance(target, ast.Name) and root:
                            found[target.id] = root
        else:
            pattern = re.compile(r"^(?:export\s+)?(?:const|let|var)\s+([\w$]+)\s*=\s*(?:new\s+)?([\w$.]+)\s*\(")
            for line in self.lines:
                match = pattern.match(line)
                if match:
                    found[match.group(1)] = match.group(2).split(".")[0]
        return found


def _call_root(func):
    while isinstance(func, (ast.Attribute, ast.Call, ast.Subscript)):
        func = func.value if not isinstance(func, ast.Call) else func.func
    return func.id if isinstance(func, ast.Name) else None


# ---------------------------------------------------------------------------- call expressions
def _receiver_before(before):
    """The dotted receiver written immediately before a call, e.g. ``hashlib.sha256(x).``.

    Scans backwards with balanced brackets so chained calls keep their root. Returns
    ``(chain as written, cleaned receiver)``; both empty when the call has no receiver.
    """
    i, depth = len(before), 0
    while i > 0:
        ch = before[i - 1]
        if ch in ")]":
            depth += 1
        elif ch in "([":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and not (ch.isalnum() or ch in "_$.?"):
            break
        i -= 1
    chain = before[i:]
    if not (chain.endswith(".") or chain.endswith("?.")):
        return "", ""
    clean = chain
    for _ in range(3):
        clean = re.sub(r"\([^()]*\)", "", clean)
    clean = re.sub(r"\[[^\]]*\]", "", clean).replace("?.", ".").rstrip(".")
    return chain, clean


def _call_expression(line, callee, occurrence):
    """The call as written on this line: (receiver chain, shown text, is_constructor)."""
    matches = list(re.finditer(r"(?<![\w$])" + re.escape(callee) + r"\s*\(", line))
    if not matches:
        return "", f"{callee}(...)", False, None
    match = matches[min(occurrence, len(matches) - 1)]
    before = line[:match.start()]
    chain, receiver = _receiver_before(before)
    constructor = bool(re.search(r"\bnew\s+$", before[:len(before) - len(chain)]))
    depth, end = 0, None
    for i in range(match.end() - 1, len(line)):
        depth += line[i] == "("
        depth -= line[i] == ")"
        if depth == 0:
            end = i
            break
    args = line[match.end():end].strip() if end is not None else "..."
    full_args = args if end is not None else None             # None: the call continues on later lines
    args = args if end is not None and 0 < len(args) <= MAX_SHOWN_ARGS and "\n" not in args else ("..." if args else "")
    shown_chain = chain
    for _ in range(3):
        shown_chain = re.sub(r"\((?:[^()]|\([^()]*\))+\)", "(...)", shown_chain)
    if len(shown_chain) > 56:
        shown_chain = shown_chain[:26] + "..." + shown_chain[-26:]
    return receiver, f"{'new ' if constructor else ''}{shown_chain}{callee}({args})", constructor, full_args


# ---------------------------------------------------------------------------- conditional calls
def _python_conditions(facts, start, end, symbol):
    """{(line, callee): reason} for conditional calls in the definition, via the real AST."""
    if facts.tree is None:
        return {}
    target = None
    for node in ast.walk(facts.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == _leaf(symbol):
            first = min([d.lineno for d in node.decorator_list] + [node.lineno])
            if first == start:
                target = node
                break
    if target is None:
        return {}
    parents = {}
    for parent in ast.walk(target):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    found = {}
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        child, reason = node, None
        while child is not target and child in parents and reason is None:
            parent = parents[child]
            if isinstance(parent, ast.If) and child is not parent.test:
                reason = f"in `if` block from line {parent.lineno}"
            elif isinstance(parent, ast.IfExp) and child is not parent.test:
                reason = f"in ternary on line {parent.lineno}"
            elif isinstance(parent, ast.BoolOp) and child is not parent.values[0]:
                reason = f"`{type(parent.op).__name__.lower()}` short-circuit on line {parent.lineno}"
            elif isinstance(parent, ast.ExceptHandler):
                reason = f"in `except` handler from line {parent.lineno}"
            child = parent
        if reason and name:
            found.setdefault((node.lineno, name), reason)
    return found


def _javascript_conditions(facts, start, end, calls):
    """{(line, callee): reason} by tracking which brace blocks were opened by if/else/catch/..."""
    stack, found = [], {}
    wanted = {}
    for call in calls:
        wanted.setdefault(call.line, []).append(call.callee)
    opened_function = False
    for number in range(start, min(end, len(facts.lines)) + 1):
        raw = facts.lines[number - 1]
        line = _blank_literals(raw)
        events = []
        for match in re.finditer(r"[{}]", line):
            events.append((match.start(), match.group(0)))
        for name in wanted.get(number, []):
            for m in re.finditer(r"(?<![\w$])" + re.escape(name) + r"\s*\(", line):
                events.append((m.start(), ("call", name)))
        events.sort(key=lambda e: (e[0], 0 if e[1] in ("{", "}") else 1))
        for column, event in events:
            if event == "{":
                head = line[:column]
                head = head[max(head.rfind("{"), head.rfind("}"), head.rfind(";")) + 1:]
                word = CONDITION_WORDS.search(head)
                if not opened_function:
                    stack.append((False, number, "function"))
                    opened_function = True
                else:
                    stack.append((bool(word), number, word.group(1) if word else "block"))
            elif event == "}":
                if stack:
                    stack.pop()
            else:
                name = event[1]
                reason = next((f"in `{kw}` block from line {ln}" for cond, ln, kw in stack[1:] if cond), None)
                before = line[:column]
                if reason is None and re.search(r"&&|\|\||\?\?|\?\.\s*$", before):
                    reason = f"short-circuit on line {number}"
                if reason is None and re.search(r"[^?]\?(?![.?])[^:]*$", before):
                    reason = f"in ternary on line {number}"
                if reason is None:
                    single = None
                    for m in re.finditer(r"\b(?:if|while|for)\s*\(", before):
                        depth, close = 0, None
                        for i in range(m.end() - 1, len(line)):
                            depth += line[i] == "("
                            depth -= line[i] == ")"
                            if depth == 0:
                                close = i
                                break
                        if close is not None and close < column:
                            single = m.group(0).split("(")[0].strip()
                    if single:
                        reason = f"guarded by `{single}` on line {number}"
                if reason:
                    found.setdefault((number, name), reason)
    return found


# ---------------------------------------------------------------------------- classification
class Analyzer:
    def __init__(self, project_id):
        self.project_id = project_id
        self.index = code_index.get_index(project_id)
        self._facts = {}

    def facts(self, path):
        if path not in self._facts:
            try:
                text = project_service.read_project_file(self.project_id, path)
            except Exception:
                text = ""
            self._facts[path] = FileFacts(self.index, path, text if isinstance(text, str) else "")
        return self._facts[path]

    # -- resolving what a call is --------------------------------------------------------
    def _project_target(self, facts, call, receiver, own, edge_targets):
        probe = replace(call, qualifier=receiver)
        candidates = code_index._resolve_callee(self.index, facts.path, probe)
        if not candidates:
            return []
        site = edge_targets.get((facts.path, call.line, _leaf(call.callee)))
        if site:
            return [site]
        found = [{"path": p, "symbol": s.qualname, "start": s.start, "end": s.end, "kind": s.kind}
                 for p, s in candidates]
        unique = {(d["path"], d["start"], d["end"]): d for d in found}
        return list(unique.values())

    def _external(self, facts, root, receiver, call, local_origin, params):
        """(label, kind) when the call's receiver/callee is known to be external or platform."""
        names = [root]
        if root in params:                               # typed parameter: response: Response
            for word in re.findall(r"[A-Za-z_$][\w$]*", params[root]):
                if word in facts.imported:
                    names.append(word)
        if root in facts.instances:                      # _hasher = PasswordHasher()
            names.append(facts.instances[root])
        if root in facts.bindings:                       # const [x, setX] = useState(...)
            names.append(facts.bindings[root])
        for name in names:
            if name in facts.imported:
                module, external = facts.imported[name]
                if external:
                    stdlib = module.split(".")[0] in getattr(sys, "stdlib_module_names", ())
                    return (f"{module}{' stdlib' if stdlib else ''}", "external")
        if root in local_origin:
            origin_note, origin_kind = local_origin[root]
            return (f"{origin_note}", origin_kind)
        if facts.is_python and root == "self" and "." in receiver:
            external = self._attribute_external(receiver.split(".")[1])
            if external:
                return external
        if facts.is_python:
            if root in PY_BUILTINS:
                return (f"Python built-in {root}", "platform")
        elif root in JS_GLOBALS:
            return (f"browser/runtime {root}", "platform")
        return None

    def _attribute_external(self, attribute):
        """(label, 'external') when every ``self.<attribute> = <annotated param>`` in the project
        is typed with an external import (``self.session = session`` with ``session: Session``)."""
        if not hasattr(self, "_attr_types"):
            self._attr_types = {}
            for path, info in self.index.files.items():
                if not path.endswith(".py"):
                    continue
                facts = self.facts(path)
                if facts.tree is None:
                    continue
                for cls in [n for n in ast.walk(facts.tree) if isinstance(n, ast.ClassDef)]:
                    for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]:
                        annotations = {a.arg: ast.unparse(a.annotation) for a in fn.args.args if a.annotation is not None}
                        for node in ast.walk(fn):
                            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
                                    and node.value.id in annotations):
                                for target in node.targets:
                                    if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                                            and target.value.id == "self"):
                                        self._attr_types.setdefault(target.attr, []).append((path, annotations[node.value.id]))
        labels = set()
        for path, annotation in self._attr_types.get(attribute, []):
            facts = self.facts(path)
            hit = next(((facts.imported[w]) for w in re.findall(r"[A-Za-z_]\w*", annotation)
                        if w in facts.imported), None)
            if not (hit and hit[1]):
                return None
            labels.add(hit[0])
        if len(labels) == 1:
            module = labels.pop()
            stdlib = module.split(".")[0] in getattr(sys, "stdlib_module_names", ())
            return (f"{module}{' stdlib' if stdlib else ''}, injected as self.{attribute}", "external")
        return None

    def classify(self, path, definition, edge_targets):
        """Classified calls for one definition."""
        facts = self.facts(path)
        info = self.index.files.get(path)
        if not (info and facts.lines):
            return []
        symbol, start, end = definition["symbol"], definition["start"], definition["end"]
        own_calls = [c for c in info.parsed.calls
                     if start <= c.line <= end and (c.caller == symbol or _leaf(c.caller) == _leaf(symbol))]
        conditions = (_python_conditions(facts, start, end, symbol) if facts.is_python
                      else _javascript_conditions(facts, start, end, own_calls))
        params = self._parameters(facts, definition)
        local_origin, seen, out = {}, {}, []
        for call in sorted(own_calls, key=lambda c: c.line):
            if call.callee in NOT_CALLS:
                continue
            line = facts.lines[call.line - 1] if call.line - 1 < len(facts.lines) else ""
            occurrence = seen.get((call.line, call.callee), 0)
            seen[(call.line, call.callee)] = occurrence + 1
            receiver, shown, constructor, full_args = _call_expression(line, call.callee, occurrence)
            root = (receiver.split(".")[0] if receiver else call.callee)
            if full_args is None:                              # the call continues on the next lines
                joined = " ".join(row.strip() for row in facts.lines[call.line - 1: call.line + 11])
                full_args = _call_expression(joined, call.callee, occurrence)[3]
            item = {"line": call.line, "expr": shown, "callee": call.callee, "receiver": receiver, "args": full_args,
                    "conditional": conditions.get((call.line, call.callee)), "targets": [], "note": ""}
            targets = self._project_target(facts, call, receiver, definition, edge_targets)
            if targets and not (receiver and root in params and root not in {"self", "cls", "this"}
                                and not any(root == t["symbol"].split(".")[0] for t in targets)
                                and self._external(facts, root, receiver, call, local_origin, params)):
                recursive = [t for t in targets if t["path"] == path and t["start"] == start and t["end"] == end]
                if recursive and len(recursive) == len(targets):
                    item["kind"], item["targets"] = "recursive", recursive
                else:
                    item["kind"], item["targets"] = "project", targets
            else:
                known = self._external(facts, root, receiver, call, local_origin, params)
                if known:
                    item["kind"], item["note"] = known[1], known[0]
                elif call.callee == _leaf(symbol) and receiver in ("", "self", "this"):
                    item["kind"], item["targets"] = "recursive", [{"path": path, "symbol": symbol,
                                                                  "start": start, "end": end}]
                elif (JS_BUILTIN_METHODS if not facts.is_python else PY_BUILTIN_METHODS).__contains__(call.callee) \
                        and receiver and not self.index.by_name.get(call.callee):
                    item["kind"], item["note"] = "platform", ("built-in method" if facts.is_python else "runtime method")
                elif re.search(r"(?:\(\s*|,\s*)" + re.escape(call.callee) + r"\s*[,)][^=]*=>|function\s*\([^)]*\b"
                               + re.escape(call.callee) + r"\b", line) and not receiver:
                    item["kind"], item["note"] = "unknown", (
                        f"`{call.callee}` is a callback parameter here; which function it runs is decided at runtime")
                else:
                    item["kind"] = "unknown"
                    item["note"] = (f"no project definition, import or known platform API matches `{root}`"
                                    if root != call.callee else f"no project definition or import matches `{call.callee}`")
            item["hook_value"] = bool(root in facts.bindings and root not in facts.imported)
            assigned = re.match(r"\s*(?:const|let|var)?\s*([\w$]+)\s*=\s*(?:await\s+)?", line)
            if assigned and item["kind"] in ("external", "platform"):
                local_origin[assigned.group(1)] = (item["note"], item["kind"])
            out.append(item)
        return out

    def _parameters(self, facts, definition):
        """{parameter: annotation text} for Python; empty for languages without annotations."""
        params = {}
        if facts.tree is None:
            return params
        for node in ast.walk(facts.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == _leaf(definition["symbol"]):
                first = min([d.lineno for d in node.decorator_list] + [node.lineno])
                if first == definition["start"]:
                    for arg in node.args.args + node.args.kwonlyargs:
                        if arg.annotation is not None:
                            params[arg.arg] = ast.unparse(arg.annotation)
        return params

    def _function_node(self, facts, definition):
        if facts.tree is None:
            return None
        for node in ast.walk(facts.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == _leaf(definition["symbol"]):
                if min([d.lineno for d in node.decorator_list] + [node.lineno]) == definition["start"]:
                    return node
        return None

    def _describe_annotation(self, facts, annotation):
        """What a parameter's type is: external library, project class, or project type alias."""
        for word in re.findall(r"[A-Za-z_]\w*", annotation):
            if word in facts.imported and facts.imported[word][1]:
                return f"external type from `{facts.imported[word][0]}`"
            hits = [(p, sym) for p, sym in self.index.by_name.get(word, []) if sym.kind == "class"]
            if hits:
                path, sym = hits[0]
                return f"project class `{sym.qualname}` ({path}:{sym.start}-{sym.end})"
        for word in re.findall(r"[A-Za-z_]\w*", annotation):
            alias = self._find_alias(word)
            if alias:
                return alias
        return ""

    def _find_alias(self, word):
        """``Name = <expression>`` at module level in a project Python file: a project type alias."""
        pattern = re.compile(r"^" + re.escape(word) + r"\s*=\s*(.+)$")
        for path in self.index.files:
            if not path.endswith(".py"):
                continue
            for number, line in enumerate(self.facts(path).lines, 1):
                match = pattern.match(line)
                if match:
                    return f"project type alias `{word} = {match.group(1)[:80]}` ({path}:{number})"
        return ""

    def parameter_flow(self, path, definition):
        """[(name, description)] of a Python function's parameters: their types and where each is used."""
        facts = self.facts(path)
        node = self._function_node(facts, definition)
        if node is None:
            return []
        out = []
        for arg in node.args.args + node.args.kwonlyargs:
            if arg.arg in ("self", "cls"):
                continue
            annotation = ast.unparse(arg.annotation) if arg.annotation is not None else ""
            parts = []
            kind = self._describe_annotation(facts, annotation) if annotation else ""
            if kind:
                parts.append(kind)
            used = [u for u in self._python_uses(node, arg.arg, node.lineno)]
            for sub in ast.walk(node):                          # receiver uses: request.headers.get(...)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and _call_root(sub.func) == arg.arg:
                    used.append(f"is the receiver of `{ast.unparse(sub.func)}(...)` (line {sub.lineno})")
            out.append((arg.arg, annotation, "; ".join(parts), used[:3]))
        return out

    def local_names(self, path, definition):
        """Names bound inside the definition: parameters and assigned variables."""
        facts = self.facts(path)
        names = set()
        node = self._function_node(facts, definition)
        if node is not None:
            for arg in node.args.args + node.args.kwonlyargs:
                names.add(arg.arg)
            for sub in ast.walk(node):
                targets = []
                if isinstance(sub, ast.Assign):
                    targets = sub.targets
                elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                    targets = [sub.target]
                for target in targets:
                    for leaf in ast.walk(target):
                        if isinstance(leaf, ast.Name):
                            names.add(leaf.id)
        else:
            for number in range(definition["start"], min(definition["end"], len(facts.lines)) + 1):
                for match in re.finditer(r"(?:const|let|var)\s+(\{[^}]*\}|\[[^\]]*\]|[\w$]+)", facts.lines[number - 1]):
                    names.update(re.findall(r"[\w$]+", match.group(1)))
        return sorted(names)

    def references(self, path, definition):
        """[(line, text)] where a JS definition is mentioned but not called, outside its own range
        (``{ user, login, logout }`` shared through a context value, passed as a callback, ...)."""
        if path.endswith(".py"):
            return []
        facts = self.facts(path)
        name = _leaf(definition["symbol"])
        pattern = re.compile(r"(?<![\w$.])" + re.escape(name) + r"(?![\w$])(?!\s*\()")
        declaration = re.compile(r"\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:const|let|var|function)\s+" + re.escape(name) + r"\b")
        found = []
        for number, row in enumerate(facts.lines, 1):
            if definition["start"] <= number <= definition["end"] or declaration.match(row) or row.lstrip().startswith(("import ", "//", "*", "/*")):
                continue
            if pattern.search(_blank_literals(row)):
                found.append((number, row.strip()))
        return found[:3]

    # -- data flow -------------------------------------------------------------------------
    def returns(self, path, definition):
        """[(line, [element expressions])] for each return statement with a value."""
        facts = self.facts(path)
        found = []
        if facts.tree is not None:
            for node in ast.walk(facts.tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == _leaf(definition["symbol"]):
                    first = min([d.lineno for d in node.decorator_list] + [node.lineno])
                    if first == definition["start"]:
                        for sub in ast.walk(node):
                            if isinstance(sub, ast.Return) and sub.value is not None:
                                value = sub.value
                                parts = value.elts if isinstance(value, ast.Tuple) else [value]
                                found.append((sub.lineno, [ast.unparse(p) for p in parts]))
        else:
            for number in range(definition["start"], min(definition["end"], len(facts.lines)) + 1):
                match = re.match(r"\s*return\s+(.+?);?\s*$", facts.lines[number - 1])
                if match:
                    found.append((number, [match.group(1)]))
        return sorted(found)

    def data_flow(self, path, definition, calls):
        """Lines describing what each call returns, who receives it and where it goes."""
        facts = self.facts(path)
        out = []
        if facts.tree is not None:
            out.extend(self._python_flow(facts, definition, calls))
        else:
            out.extend(self._javascript_flow(facts, definition, calls))
        for line, parts in self.returns(path, definition)[:3]:
            shown = ", ".join(p if len(p) <= 40 else p[:37] + "..." for p in parts)
            out.append(f"returns `{shown}` (line {line})" + (f" [{len(parts)} values]" if len(parts) > 1 else ""))
        return out

    def _python_flow(self, facts, definition, calls):
        target = None
        for node in ast.walk(facts.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == _leaf(definition["symbol"]):
                first = min([d.lineno for d in node.decorator_list] + [node.lineno])
                if first == definition["start"]:
                    target = node
                    break
        if target is None:
            return []
        by_line = {}
        for call in calls:
            by_line.setdefault((call["line"], call["callee"]), call)
        out = []
        for node in ast.walk(target):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            value = node.value.value if isinstance(node.value, ast.Await) else node.value
            if not isinstance(value, ast.Call):
                continue
            name = value.func.attr if isinstance(value.func, ast.Attribute) else getattr(value.func, "id", None)
            call = by_line.get((value.lineno, name))
            tgt = node.targets[0]
            names = [e.id for e in tgt.elts if isinstance(e, ast.Name)] if isinstance(tgt, ast.Tuple) \
                else ([tgt.id] if isinstance(tgt, ast.Name) else [])
            if not names or not call:
                continue
            text = f"line {node.lineno} `{ast.unparse(tgt)} = {call['expr']}` receives " + (
                f"{len(names)} values" if len(names) > 1 else "its value")
            projects = [t for t in call["targets"] if call["kind"] == "project"]
            if len(projects) == 1:
                text += f" from `{projects[0]['symbol']}`"
                rets = self.returns(projects[0]["path"], projects[0])
                short = lambda x: x if len(x) <= 36 else x[:33] + "..."
                mapped = next(((ln, parts) for ln, parts in rets if len(parts) == len(names) > 1), None)
                if mapped:
                    ret_line, parts = mapped
                    pairs = "; ".join(f"`{n}` = `{short(p)}`" for n, p in zip(names, parts))
                    text += f". The callee's return (line {ret_line}) maps by position: {pairs}"
                    others = [ln for ln, _ in rets if ln != ret_line]
                elif len(names) == 1 and rets:
                    text += ". The callee returns " + " or ".join(
                        f"`{short(', '.join(parts))}` (line {ln})" for ln, parts in rets[:2])
                    others = [ln for ln, _ in rets[2:]]
                else:
                    others = []
                if others:
                    text += f" (other returns at lines {', '.join(str(n) for n in others[:3])})"
            out.append(text)
            for var in names:
                for use in self._python_uses(target, var, node.lineno):
                    out.append(f"`{var}` {use}")
        return out

    def _python_uses(self, function_node, var, after):
        uses = []
        for node in ast.walk(function_node):
            if isinstance(node, ast.Call) and node.lineno >= after:
                for position, arg in enumerate(node.args, 1):
                    if any(isinstance(n, ast.Name) and n.id == var for n in ast.walk(arg)):
                        name = ast.unparse(node.func)
                        direct = isinstance(arg, ast.Name)
                        uses.append((node.lineno, f"{'is passed as' if direct else 'is used in'} argument {position} "
                                                  f"of `{name}(...)` (line {node.lineno})"))
                for keyword in node.keywords:
                    if any(isinstance(n, ast.Name) and n.id == var for n in ast.walk(keyword.value)):
                        uses.append((node.lineno, f"is passed as `{keyword.arg}=` to `{ast.unparse(node.func)}(...)` "
                                                  f"(line {node.lineno})"))
        uses.sort()
        return [u[1] for u in uses[:3]]

    def _javascript_flow(self, facts, definition, calls):
        out = []
        lines = facts.lines
        for call in calls:
            line = lines[call["line"] - 1] if call["line"] - 1 < len(lines) else ""
            match = re.match(r"\s*(?:const|let|var)?\s*(\{[^}]*\}|\[[^\]]*\]|[\w$]+)\s*=\s*(?:await\s+)?[\w$.?]*" +
                             re.escape(call["callee"]), line)
            if not match:
                continue
            raw = match.group(1)
            names = [n for n in re.findall(r"[\w$]+", raw)] if raw[0] in "{[" else [raw]
            out.append(f"line {call['line']} `{raw} = {call['expr']}` receives its value"
                       + (f" from `{call['targets'][0]['symbol']}`" if call["kind"] == "project" and call["targets"] else ""))
            for var in names[:3]:
                hits = 0
                for number in range(call["line"] + 1, min(definition["end"], len(lines)) + 1):
                    text = _blank_literals(lines[number - 1])
                    if re.search(r"(?<![\w$.])" + re.escape(var) + r"(?![\w$])", text):
                        passed = re.search(r"([\w$.]+)\([^()]*(?<![\w$.])" + re.escape(var) + r"(?![\w$])", text)
                        if re.match(r"\s*return\b", text):
                            out.append(f"`{var}` is returned (line {number})")
                        elif passed:
                            out.append(f"`{var}` is passed to `{passed.group(1)}(...)` (line {number})")
                        else:
                            continue
                        hits += 1
                        if hits >= 2:
                            break
        return out


# ---------------------------------------------------------------------------- packet interface
RANK_KEY = 0          # recursion and multi-value data flow
RANK_EXTERNAL = 1     # named external libraries / platform APIs (known dependencies, not unknown)
RANK_COND = 2         # conditional calls to project functions
RANK_VERIFIED = 3     # merged calls to definitions listed in VERIFIED DEFINITIONS
RANK_OTHER = 4        # other project functions (source not shown)
RANK_GENERIC = 5      # generic built-in / runtime methods (startsWith, forEach, preventDefault ...)
RANK_USES = 6         # where a received value is passed next
RANK_SINGLE = 7       # single-value data flow
RANK_RETURNS = 8      # return statements
RANK_UNKNOWN = 9      # calls the source does not establish
GENERIC_NOTES = ("runtime method", "built-in method")


def _drop_inner_chain_calls(calls):
    """``a.b(x).c()`` also contains the call ``a.b(x)``; keep only the outermost on a line."""
    kept = []
    for call in calls:
        prefix = call["expr"].split("(", 1)[0]
        contained = any(
            other is not call and other["line"] == call["line"] and other["kind"] == call["kind"]
            and other["note"] == call["note"] and len(other["expr"]) > len(call["expr"])
            and other["expr"].startswith(prefix + "(") and "(...)." in other["expr"][len(prefix):]
            for other in calls)
        if not contained:
            kept.append(call)
    return kept


def self_external_lines(calls, generic, rank):
    """One line for the external/platform calls of a definition: named dependencies, or the generic
    built-in/runtime methods. Long lines carry shorter variants so they can be shortened under space
    pressure instead of being dropped."""
    per_note = {}
    for call in calls:
        if call["kind"] in ("external", "platform") and (call["note"] in GENERIC_NOTES) == generic:
            per_note.setdefault(call["note"], []).append(call)
    if not per_note:
        return []
    shown, external = 0, []
    for note, group in per_note.items():
        if len(external) >= 3:
            break
        call = group[0]
        cond = f" (conditional: {call['conditional']})" if call["conditional"] else ""
        extra = f" (+{len(group) - 1} more)" if len(group) > 1 else ""
        external.append(f"line {call['line']} `{call['expr']}` [{note}]{extra}{cond}")
        shown += len(group)
    total = sum(len(g) for g in per_note.values())
    rest = total - shown
    full = "; ".join(external) + (f"; {rest} more calls" if rest > 0 else "")
    first_note = next(iter(per_note))
    first = per_note[first_note][0]
    cond = f" (conditional: {first['conditional']})" if first["conditional"] else ""
    short = f"line {first['line']} `{first['expr']}` [{first_note}]{cond}" + \
        (f"; {total - 1} more calls" if total > 1 else "")
    bare = re.sub(r"\((?:[^()]|\([^()]*\))*\)$", "(...)", first["expr"])
    tiny = f"line {first['line']} `{bare}` [{first_note}]" + (f" +{total - 1} more" if total > 1 else "")
    variants = [full]
    for variant in (short, tiny):
        if len(variant) < len(variants[-1]):
            variants.append(variant)
    return [(rank, variants if len(variants) > 1 else full)]


def describe(project_id, definitions, edges):
    """``({definition number: [(rank, text)]}, {(path, line, callee): call as written})``.

    The first is the packet's CALL ANALYSIS section; the second lets the call-site list show the
    expression that was actually written next to the definition it resolves to."""
    analyzer = Analyzer(project_id)
    edge_targets = {}
    for source, target, path, line in edges:
        edge_targets[(path, line, _leaf(target["symbol"]))] = {
            "path": target["path"], "symbol": target["symbol"], "start": target["start"], "end": target["end"]}
    in_packet = {(d["path"], d["start"], d["end"]): n for n, d in enumerate(definitions, 1)}
    result, expressions = {}, {}
    for number, definition in enumerate(definitions, 1):
        try:
            calls = _drop_inner_chain_calls(analyzer.classify(definition["path"], definition, edge_targets))
            lines, merged, others = [], [], []
            for call in calls:
                if call["kind"] not in ("project", "recursive"):
                    continue
                expressions.setdefault((definition["path"], call["line"], call["callee"]), call["expr"])
                cond = f"; conditional ({call['conditional']})" if call["conditional"] else ""
                if call["kind"] == "recursive":
                    lines.append((RANK_KEY, f"line {call['line']} `{call['expr']}` -> RECURSIVE call to itself{cond}"))
                    continue
                refs = [in_packet.get((t["path"], t["start"], t["end"])) for t in call["targets"][:2]]
                verified_hit = any(r is not None for r in refs)
                many = f" (ambiguous: {len(call['targets'])} project definitions)" if len(call["targets"]) > 1 else ""
                names = [f"#{r} `{t['symbol']}`" if r is not None else
                         f"`{t['symbol']}` ({_short_path(t['path'])}:{t['start']}-{t['end']}, source not shown)"
                         for r, t in zip(refs, call["targets"][:2])]
                if call["conditional"]:
                    lines.append((RANK_COND, f"line {call['line']} `{call['expr']}` -> {' or '.join(names)}{many}{cond}"))
                elif verified_hit:
                    merged.append(f"line {call['line']} `{call['expr']}` -> {' or '.join(names)}{many}")
                else:
                    others.append(f"line {call['line']} `{call['expr']}` -> {' or '.join(names)}{many}")
            if merged:
                lines.append((RANK_VERIFIED, "calls definitions above: " + "; ".join(merged)))
            if others:
                lines.append((RANK_OTHER, "calls other project functions: " + "; ".join(others)))
            for generic, rank in ((False, RANK_EXTERNAL), (True, RANK_GENERIC)):
                lines.extend(self_external_lines(calls, generic, rank))
            for call in [c for c in calls if c["kind"] == "unknown"][:3]:
                lines.append((RANK_UNKNOWN, f"line {call['line']} `{call['expr']}` -> cannot be established from "
                                            f"the source: {call['note']}"))
            for text in analyzer.data_flow(definition["path"], definition, calls):
                if text.startswith("returns"):
                    rank = RANK_RETURNS
                elif text.startswith("line "):
                    rank = RANK_KEY if "values" in text.split(" receives ", 1)[-1][:12] else RANK_SINGLE
                else:
                    rank = RANK_USES
                lines.append((rank, f"data flow: {text}"))
            if lines:
                result[number] = lines
        except Exception:                      # analysis is an aid; it must never break a trace
            continue
    return result, expressions


def fit(analysis, definitions, limit, skip_ranks=()):
    """Render the analysis within ``limit`` characters.

    Returns ``(lines, omitted, lost_edges)``. A line may carry shorter variants: when space runs
    short the longest line of the least valuable tier is first degraded to its shorter variant and
    only dropped when it has none left. Lines are grouped under their definition number.
    """
    entries = []
    for number, rows in analysis.items():
        for order, (rank, text) in enumerate(rows):
            if rank in skip_ranks:
                continue
            variants = text if isinstance(text, list) else [text]
            entries.append({"rank": rank, "number": number, "order": order, "variants": variants, "level": 0})

    def current(entry):
        return entry["variants"][entry["level"]]

    def render(chosen):
        keep, named, generic = {}, [], []
        for entry in sorted(chosen, key=lambda e: (e["number"], e["order"])):
            if entry["rank"] == RANK_EXTERNAL:
                named.append(f"  - #{entry['number']} {current(entry)}")
            elif entry["rank"] == RANK_GENERIC:
                generic.append(f"  - #{entry['number']} {current(entry)}")
            else:
                keep.setdefault(entry["number"], []).append(current(entry))
        out = []
        for number, rows in keep.items():
            out.append(f"#{number} `{definitions[number - 1]['symbol']}`")
            out.extend(f"  - {row}" for row in rows)
        if named:
            out.append("NOT project functions (external libraries / platform APIs), by definition #:")
            out.extend(named)
        if generic:
            out.append("NOT project functions (generic built-in / runtime methods), by definition #:")
            out.extend(generic)
        return out

    chosen = list(entries)
    while chosen and sum(len(x) + 1 for x in render(chosen)) > limit:
        worst_rank = max(e["rank"] for e in chosen)
        tier = [e for e in chosen if e["rank"] == worst_rank]
        shrinkable = [e for e in tier if e["level"] + 1 < len(e["variants"])]
        if shrinkable:                                  # shorten every line of the tier before dropping any
            max(shrinkable, key=lambda e: len(current(e)))["level"] += 1
        else:                                           # then the deepest helper (latest in request order) goes first
            chosen.remove(max(tier, key=lambda e: (e["number"], e["order"])))
    lost_edges = any(current(e).startswith("calls definitions above") for e in entries if e not in chosen)
    return render(chosen), len(entries) - len(chosen), lost_edges


def facts(project_id, definitions, edges):
    """Structured connection facts for each definition, in order (same order as ``definitions``).

    Each item has: called_by, calls (classified, with written expression and receiver type), flow
    (data-flow lines), params. Used after generation to complete the answer with what the model
    omitted or could not establish. Pure function of the source and index.
    """
    analyzer = Analyzer(project_id)
    edge_targets = {}
    for source, target, path, line in edges:
        edge_targets[(path, line, _leaf(target["symbol"]))] = {
            "path": target["path"], "symbol": target["symbol"], "start": target["start"], "end": target["end"]}
    number_of = {(d["path"], d["start"], d["end"]): n for n, d in enumerate(definitions, 1)}
    result = [{"number": n, "symbol": d["symbol"], "path": d["path"], "start": d["start"], "end": d["end"],
               "called_by": [], "calls": [], "flow": [], "params": []} for n, d in enumerate(definitions, 1)]
    for n, definition in enumerate(definitions, 1):
        try:
            calls = _drop_inner_chain_calls(analyzer.classify(definition["path"], definition, edge_targets))
        except Exception:
            continue
        item = result[n - 1]
        for call in calls:
            targets = call["targets"][:2]
            note = ""
            if call["kind"] == "project" and len(targets) == 1 and call["receiver"] not in ("", "self", "this", "cls"):
                owner = targets[0]["symbol"].rsplit(".", 1)[0] if "." in targets[0]["symbol"] else ""
                if owner and owner != call["receiver"]:
                    note = f"receiver `{call['receiver']}` is a `{owner}`"
            entry = {"line": call["line"], "expr": call["expr"], "callee": call["callee"], "kind": call["kind"],
                     "targets": targets, "note": call["note"], "conditional": call["conditional"],
                     "receiver": call["receiver"], "receiver_note": note, "hook_value": call.get("hook_value", False),
                     "args": call.get("args")}
            item["calls"].append(entry)
            if call["kind"] == "project":
                for target in targets:
                    ref = number_of.get((target["path"], target["start"], target["end"]))
                    if ref is not None and ref != n:
                        result[ref - 1]["called_by"].append(
                            {"caller": definition["symbol"], "caller_no": n, "line": call["line"],
                             "expr": call["expr"], "receiver_note": note,
                             "path": definition["path"], "start": definition["start"], "end": definition["end"]})
        try:
            item["flow"] = analyzer.data_flow(definition["path"], definition, calls)
            item["returns"] = analyzer.returns(definition["path"], definition)
            item["references"] = analyzer.references(definition["path"], definition)
            item["locals"] = analyzer.local_names(definition["path"], definition)
            item["params"] = analyzer.parameter_flow(definition["path"], definition)
            rows = analyzer.facts(definition["path"]).lines
            window = rows[definition["start"] - 1: definition["start"] + 6]
            declaration = next((r for r in window if re.match(r"\s*(?:export\s+)?(?:async\s+)?(?:def|function|class)\b", r)
                                or "=>" in r), window[0] if window else "")
            item["signature"] = declaration.strip()
        except Exception:
            pass
    return result

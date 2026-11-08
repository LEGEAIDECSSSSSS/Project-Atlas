"""Turn source text into symbols, imports, calls, routes and entry points.

Python uses the standard library's real parser (`ast`). JavaScript/TypeScript, Java and other
brace languages use careful line scanners with brace matching: not a full grammar, but
accurate for the declarations, imports and routes that matter for navigating a project.
A parser that fails on a file falls back to the generic scanner, so one odd file can never
break indexing of a whole project.
"""
import ast
import re
from dataclasses import dataclass, field
from services.textlines import source_lines

LANG_BY_EXT = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust", ".kt": "kotlin", ".kts": "kotlin",
    ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift", ".scala": "scala",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
}
HTTP_VERBS = {"get", "post", "put", "delete", "patch", "head", "options"}
CALL_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "function", "new", "else", "typeof",
    "await", "async", "super", "this", "import", "require", "def", "class", "elif", "with",
    "print", "len", "range", "str", "int", "float", "list", "dict", "set", "tuple", "isinstance",
    "do", "try", "finally", "throw", "yield", "in", "of", "and", "or", "not", "is", "lambda",
    "assert", "raise", "except", "del", "pass", "from", "as", "global", "nonlocal",
}
MAX_FILE_CHARS_PARSED = 600_000
MAX_PARSED_LINE = 1000          # longer lines (minified code, data blobs) are skipped by the scanners


@dataclass
class Symbol:
    name: str
    qualname: str
    kind: str                 # class | function | method | component | interface | enum
    start: int
    end: int
    parent: str = ""
    signature: str = ""
    doc: str = ""
    decorators: list = field(default_factory=list)
    end_known: bool = True


@dataclass
class Call:
    callee: str
    line: int
    caller: str = "<module>"
    qualifier: str = ""


@dataclass
class Import:
    module: str
    names: list
    line: int


@dataclass
class Route:
    method: str
    path: str
    handler: str
    line: int
    framework: str = ""


@dataclass
class ApiCall:
    method: str
    path: str
    line: int


@dataclass
class EntryPoint:
    kind: str
    line: int
    detail: str = ""


@dataclass
class Parsed:
    symbols: list = field(default_factory=list)
    imports: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    routes: list = field(default_factory=list)
    api_calls: list = field(default_factory=list)
    entry_points: list = field(default_factory=list)
    parser: str = ""


def language_of(path):
    name = path.rsplit("/", 1)[-1]
    if "." not in name:
        return None
    return LANG_BY_EXT.get("." + name.rsplit(".", 1)[-1].lower())


# --------------------------------------------------------------------------- Python

def _unparse(node, limit=80):
    try:
        text = ast.unparse(node)
    except Exception:
        return ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _const_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _route_from_decorator(decorator, handler, line):
    """FastAPI / Flask / Starlette style decorators -> list of Route."""
    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
        return []
    verb = decorator.func.attr.lower()
    owner = _unparse(decorator.func.value, 40)
    if verb not in HTTP_VERBS and verb not in {"route", "api_route", "websocket"}:
        return []

    path = _const_str(decorator.args[0]) if decorator.args else None
    if path is None:
        for keyword in decorator.keywords:
            if keyword.arg in ("path", "rule", "url"):
                path = _const_str(keyword.value)
    if path is None:
        return []

    methods = []
    if verb in HTTP_VERBS:
        methods = [verb.upper()]
    elif verb == "websocket":
        methods = ["WEBSOCKET"]
    else:
        for keyword in decorator.keywords:
            if keyword.arg == "methods" and isinstance(keyword.value, (ast.List, ast.Tuple)):
                methods = [str(_const_str(e) or "").upper() for e in keyword.value.elts]
        methods = [m for m in methods if m] or ["GET"]

    framework = "flask" if verb == "route" else "fastapi"
    return [Route(m, path, handler, line, f"{framework} ({owner})") for m in methods]


def parse_python(text, path=""):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        parsed = parse_generic(text)
        parsed.parser = "generic (python syntax error)"
        return parsed

    lines = source_lines(text)
    out = Parsed(parser="python-ast")
    is_urls = path.rsplit("/", 1)[-1] == "urls.py"

    def collect_calls(node, caller):
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name):
                    out.calls.append(Call(func.id, child.lineno, caller))
                elif isinstance(func, ast.Attribute):
                    out.calls.append(Call(func.attr, child.lineno, caller,
                                          _unparse(func.value, 40)))

    def walk(body, prefix, in_class, caller):
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([stmt.lineno] + [d.lineno for d in stmt.decorator_list])
                qual = f"{prefix}{stmt.name}"
                kind = "method" if in_class else "function"
                out.symbols.append(Symbol(
                    name=stmt.name, qualname=qual, kind=kind, start=start,
                    end=getattr(stmt, "end_lineno", start) or start,
                    parent=prefix.rstrip("."),
                    signature=(lines[stmt.lineno - 1].strip() if stmt.lineno - 1 < len(lines) else ""),
                    doc=(ast.get_docstring(stmt) or "").strip().split("\n")[0][:120],
                    decorators=[_unparse(d) for d in stmt.decorator_list][:4],
                ))
                for decorator in stmt.decorator_list:
                    out.routes.extend(_route_from_decorator(decorator, stmt.name, decorator.lineno))
                    collect_calls(decorator, caller)
                if stmt.name == "main" and not prefix:
                    out.entry_points.append(EntryPoint("main()", stmt.lineno, "function main"))
                walk(stmt.body, qual + ".", False, qual)
            elif isinstance(stmt, ast.ClassDef):
                start = min([stmt.lineno] + [d.lineno for d in stmt.decorator_list])
                qual = f"{prefix}{stmt.name}"
                out.symbols.append(Symbol(
                    name=stmt.name, qualname=qual, kind="class", start=start,
                    end=getattr(stmt, "end_lineno", start) or start, parent=prefix.rstrip("."),
                    signature=(lines[stmt.lineno - 1].strip() if stmt.lineno - 1 < len(lines) else ""),
                    doc=(ast.get_docstring(stmt) or "").strip().split("\n")[0][:120],
                    decorators=[_unparse(d) for d in stmt.decorator_list][:4],
                ))
                for base in stmt.bases:
                    collect_calls(base, caller)
                walk(stmt.body, qual + ".", True, qual)
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    out.imports.append(Import(alias.name, [alias.asname or alias.name.split(".")[-1]],
                                              stmt.lineno))
            elif isinstance(stmt, ast.ImportFrom):
                module = "." * (stmt.level or 0) + (stmt.module or "")
                out.imports.append(Import(module, [a.name for a in stmt.names], stmt.lineno))
            elif isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With,
                                   ast.AsyncWith, ast.Try, ast.Match)
                            ) or type(stmt).__name__ == "TryStar":
                if isinstance(stmt, ast.If) and _is_main_guard(stmt.test):
                    out.entry_points.append(EntryPoint("main-guard", stmt.lineno,
                                                       'if __name__ == "__main__"'))
                for _name, value in ast.iter_fields(stmt):
                    items = value if isinstance(value, list) else [value]
                    stmts = []
                    for item in items:
                        if isinstance(item, ast.stmt):
                            stmts.append(item)
                        elif isinstance(item, (ast.ExceptHandler, ast.match_case)):
                            stmts.extend(item.body)
                        elif isinstance(item, ast.AST):
                            collect_calls(item, caller)
                    if stmts:
                        walk(stmts, prefix, in_class, caller)
            else:
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and not prefix:
                    _detect_app_object(stmt, out)
                collect_calls(stmt, caller)
                if is_urls:
                    _detect_django_route(stmt, out)

    def _detect_django_route(stmt, parsed):
        for child in ast.walk(stmt):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) \
                    and child.func.id in ("path", "re_path", "url") and len(child.args) >= 2:
                route = _const_str(child.args[0])
                view = child.args[1]
                name = view.attr if isinstance(view, ast.Attribute) else (
                    view.id if isinstance(view, ast.Name) else "<view>")
                if route is not None:
                    parsed.routes.append(Route("ANY", "/" + route.lstrip("/"), name,
                                               child.lineno, "django"))

    walk(tree.body, "", False, "<module>")
    return out


def _is_main_guard(test):
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "__name__" and test.comparators
            and _const_str(test.comparators[0]) == "__main__")


def _detect_app_object(stmt, out):
    value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
    if isinstance(value, ast.Call):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else "")
        if name in {"FastAPI", "Flask", "Starlette", "Quart", "Sanic", "APIRouter", "Blueprint"}:
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            label = ", ".join(t.id for t in targets if isinstance(t, ast.Name))
            kind = "app-object" if name != "APIRouter" and name != "Blueprint" else "router-object"
            out.entry_points.append(EntryPoint(kind, stmt.lineno, f"{label} = {name}()"))


# --------------------------------------------------------------------------- brace languages

def block_end(lines, start_index, max_lines=2500):
    """1-based last line of the {...} block that starts at or after lines[start_index].

    Returns None when no block opens within a few lines (expression bodies, interfaces...).
    """
    depth = 0
    opened = False
    in_str = None
    in_block_comment = False
    paren_depth = 0
    bracket_depth = 0
    last = min(len(lines), start_index + max_lines)
    for i in range(start_index, last):
        line = lines[i]
        j = 0
        while j < len(line):
            ch = line[j]
            two = line[j:j + 2]
            if in_block_comment:
                if two == "*/":
                    in_block_comment = False
                    j += 2
                else:
                    j += 1
                continue
            if in_str:
                if ch == "\\":
                    j += 2
                    continue
                if ch == in_str:
                    in_str = None
                j += 1
                continue
            if two == "//":
                break
            if two == "/*":
                in_block_comment = True
                j += 2
                continue
            if ch in "\"'`":
                in_str = ch
            elif ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == "{" and paren_depth == 0 and bracket_depth == 0:
                depth += 1
                opened = True
            elif ch == "}" and paren_depth == 0 and bracket_depth == 0:
                depth -= 1
                if opened and depth <= 0:
                    return i + 1
            j += 1
        if in_str in ("'", '"'):
            in_str = None
        if not opened and i - start_index >= 3:
            return None
        if not opened and i == start_index and line.rstrip().endswith((";", ",")):
            return i + 1
    return last if opened else None


def _finish_ends(symbols, total_lines):
    """Give symbols with an unknown end a sensible one: up to the next symbol, bounded."""
    ordered = sorted(symbols, key=lambda s: (s.start, s.end))
    for index, symbol in enumerate(ordered):
        if symbol.end_known:
            continue
        following = [s.start for s in ordered[index + 1:] if s.start > symbol.start]
        limit = (following[0] - 1) if following else total_lines
        symbol.end = max(symbol.start, min(limit, symbol.start + 120, total_lines))


def _owner_at(symbols, line):
    """Qualified name of the innermost function/method containing `line` (else '<module>')."""
    best = None
    for symbol in symbols:
        if symbol.kind in ("function", "method", "component") and symbol.start <= line <= symbol.end:
            if best is None or (symbol.end - symbol.start) < (best.end - best.start):
                best = symbol
    return best.qualname if best else "<module>"


def _collect_line_calls(lines, symbols, skip_lines, parsed):
    for number, raw in enumerate(lines, start=1):
        if number in skip_lines:
            continue
        stripped = raw.strip()
        if not stripped or len(stripped) > MAX_PARSED_LINE or stripped.startswith(("//", "*", "/*", "#", "@")):
            continue
        for match in re.finditer(r"(?<![\w$])(?:new\s+)?([A-Za-z_$][\w$]*)\s*\(", stripped):
            name = match.group(1)
            if name in CALL_KEYWORDS:
                continue
            parsed.calls.append(Call(name, number, _owner_at(symbols, number)))


JS_FUNC = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*[<(]")
JS_ARROW = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*"
    r"(?:async\s*)?(?:function\b[^(]*\(|\([^)]*\)\s*(?::\s*[^=]+?)?=>|<[^>]*>\s*\(|"
    r"[A-Za-z_$][\w$]*\s*=>)")
JS_WRAPPED_ARROW = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
    r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*\s*\(")


def _wrapped_callback_body(lines, declaration_index, match_end):
    """Find a block-bodied arrow callback inside the assigned wrapper call.

    An expression callback such as ``useMemo(() => ({...}))`` is not the assigned
    variable's function. Only a callback whose own body starts with ``{`` qualifies.
    This distinction prevents nested arrows inside object literals from swallowing the
    following declaration and producing a bogus symbol range.
    """
    depth = 1  # The wrapper call's matched opening parenthesis.
    arrow_tail = None
    for line_index in range(declaration_index, min(len(lines), declaration_index + 41)):
        raw = lines[line_index]
        fragment = raw[match_end:] if line_index == declaration_index else raw
        clean = re.sub(
            r"""'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|`(?:\\.|[^`\\])*`|//.*$|/\*.*?\*/""",
            "", fragment,
        )
        if arrow_tail is None and "=>" in clean:
            arrow_tail = clean.split("=>", 1)[1]
        if arrow_tail is not None:
            candidate = arrow_tail.strip()
            if candidate:
                if not candidate.startswith("{"):
                    return None
                return line_index, raw.find("{", raw.find("=>") + 2)
            # Arrow and its body may be on separate lines.
            if clean.strip() and "=>" not in clean:
                candidate = clean.strip()
                if not candidate.startswith("{"):
                    return None
                return line_index, raw.find("{")
        depth += clean.count("(") - clean.count(")")
        if depth <= 0 and arrow_tail is None:
            return None
    return None



def _arrow_expression_end(lines, start_index, arrow_column):
    """Return the last line of an expression-bodied arrow without crossing declarations."""
    parens = brackets = braces = 0
    saw_expression = False
    for line_index in range(start_index, len(lines)):
        raw = lines[line_index]
        fragment = raw[arrow_column + 2:] if line_index == start_index else raw
        if not fragment.strip():
            if saw_expression:
                continue
            continue
        # Remove comments and quoted strings before balancing delimiters.
        clean = re.sub(
            r"""'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|`(?:\\.|[^`\\])*`|//.*$|/\*.*?\*/""",
            "", fragment,
        )
        if clean.strip():
            saw_expression = True
        parens += clean.count("(") - clean.count(")")
        brackets += clean.count("[") - clean.count("]")
        braces += clean.count("{") - clean.count("}")
        if min(parens, brackets, braces) < 0:
            return line_index + 1
        if parens == brackets == braces == 0 and saw_expression:
            # Operators and punctuation at the end indicate a continued expression.
            if clean.rstrip().endswith(("=>", ".", "?.", ",", "?", ":", "+", "&&", "||", "??")):
                continue
            return line_index + 1
    return len(lines) if saw_expression else None

JS_OBJECT = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*\{\s*$")
JS_OBJECT_ARROW = re.compile(
    r"^\s{2,}([A-Za-z_$][\w$]*)\s*:\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>")
JS_OBJECT_METHOD = re.compile(
    r"^\s{2,}(?:async\s+)?(?:\*\s*)?([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{")
JS_CONTROL_WORDS = frozenset({"if", "for", "while", "switch", "catch", "function", "return", "with", "else"})
JS_CLASS = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)")
JS_INTERFACE = re.compile(r"^\s*(?:export\s+)?(?:interface|enum|type)\s+([A-Za-z_$][\w$]*)")
JS_METHOD = re.compile(
    r"^\s+(?:(?:public|private|protected|static|async|readonly|override|get|set)\s+)*"
    r"([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?::\s*[^{=]+)?\s*\{")
JS_IMPORT_FROM = re.compile(
    r"import\s+(?:type\s+)?([\w$*{},\s]+?)\s+from\s+['\"]([^'\"]+)['\"]")
JS_IMPORT_BARE = re.compile(r"^\s*import\s+['\"]([^'\"]+)['\"]", re.MULTILINE)
JS_REQUIRE = re.compile(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)")
JS_DYNAMIC = re.compile(r"\bimport\(\s*['\"]([^'\"]+)['\"]\s*\)")
JS_ROUTE = re.compile(
    r"\b([A-Za-z_$][\w$]*)\.(get|post|put|delete|patch|all)\(\s*(['\"`])([^'\"`]+)\3\s*(?:,\s*([^\n]*))?")
JS_FETCH = re.compile(r"""\bfetch\(\s*([`'"])((?:(?!\1).)*)\1""")
JS_HTTP_CLIENT = re.compile(
    r"""\b(?:axios|api|http|client|request|\$http)\.(get|post|put|delete|patch)\(\s*([`'"])((?:(?!\2).)*)\2""")
JS_METHOD_OPT = re.compile(r"""method\s*:\s*['"](\w+)['"]""")
ROUTER_NAMES = re.compile(r"^(?:app|router|server|api|route|routes|r|express|\w*[Rr]outer\w*)$")


def _path_from_literal(text):
    cleaned = re.sub(r"\$\{[^}]*\}", "", text)
    match = re.search(r"/[A-Za-z0-9_\-./{}:%]*", cleaned)
    return match.group(0) if match else None


def _line_of(text, offset):
    return text.count("\n", 0, offset) + 1


def _parse_js_raw(text):
    lines = source_lines(text)
    out = Parsed(parser="js-scan")
    symbols = out.symbols
    class_ranges = []

    # Named arrow callbacks passed to higher-order wrappers are real functions too.
    # Detect the assignment and callback syntax rather than hard-coding a framework helper
    # such as React's useCallback; this also supports memoizers and project-specific wrappers.
    for index, line in enumerate(lines):
        if len(line) > MAX_PARSED_LINE:
            continue
        callback = JS_WRAPPED_ARROW.match(line)
        if callback:
            body_start = _wrapped_callback_body(lines, index, callback.end())
            if body_start is None:
                continue
            body_index, body_column = body_start
            # Start brace matching at the callback's actual body, not at the wrapper call.
            # The wrapper's opening parenthesis must not hide the callback's opening brace.
            body_lines = [lines[body_index][body_column:]] + lines[body_index + 1:]
            relative_end = block_end(body_lines, 0)
            end = body_index + relative_end if relative_end is not None else None
            symbols.append(Symbol(callback.group(1), callback.group(1), "function",
                                  index + 1, end or index + 1, signature=line.strip()[:120],
                                  end_known=end is not None))

    # Arrow-function properties on exported API objects (for example api.post) are
    # indexed as methods. This lets the call graph follow api.post -> request.
    for index, line in enumerate(lines):
        object_match = JS_OBJECT.match(line)
        if not object_match:
            continue
        object_name = object_match.group(1)
        object_end = block_end(lines, index)
        if object_end is None:
            continue
        skip_until = -1
        for property_index in range(index + 1, min(object_end, len(lines))):
            property_match = JS_OBJECT_ARROW.match(lines[property_index])
            if not property_match:
                shorthand = JS_OBJECT_METHOD.match(lines[property_index])
                if (shorthand and property_index >= skip_until
                        and shorthand.group(1) not in CALL_KEYWORDS | JS_CONTROL_WORDS):
                    method_end = block_end(lines, property_index)
                    name = shorthand.group(1)
                    symbols.append(Symbol(name, f"{object_name}.{name}", "method", property_index + 1,
                                          method_end or property_index + 1, parent=object_name,
                                          signature=lines[property_index].strip()[:120],
                                          end_known=method_end is not None))
                    skip_until = (method_end or property_index + 1) + 0
                continue
            prop_line = lines[property_index]
            body_end = block_end(lines, property_index) if "{" in prop_line.split("=>", 1)[-1] else property_index + 1
            property_name = property_match.group(1)
            skip_until = max(skip_until, body_end or property_index + 1)
            symbols.append(Symbol(property_name, f"{object_name}.{property_name}",
                                  "method", property_index + 1, body_end or property_index + 1,
                                  parent=object_name, signature=prop_line.strip()[:120],
                                  end_known=body_end is not None))
            expression = prop_line.split("=>", 1)[-1]
            for call_match in re.finditer(r"(?<![\w$])([A-Za-z_$][\w$]*)\s*\(", expression):
                callee = call_match.group(1)
                if callee not in CALL_KEYWORDS:
                    out.calls.append(Call(callee, property_index + 1, f"{object_name}.{property_name}"))

    for index, line in enumerate(lines):
        number = index + 1
        if len(line) > MAX_PARSED_LINE:
            continue
        match = JS_CLASS.match(line)
        if match:
            end = block_end(lines, index)
            symbols.append(Symbol(match.group(1), match.group(1), "class", number,
                                  end or number, signature=line.strip()[:120],
                                  end_known=end is not None))
            class_ranges.append((number, end or number, match.group(1)))
            continue
        function_match = JS_FUNC.match(line)
        arrow_match = JS_ARROW.match(line)
        match = function_match or arrow_match
        if match and not line.startswith((" " * 8, "\t\t\t")):
            name = match.group(1)
            if function_match:
                end = block_end(lines, index)
            else:
                arrow_column = line.find("=>", match.end() - 2)
                body_start = line[arrow_column + 2:].lstrip() if arrow_column >= 0 else ""
                end = (block_end(lines, index) if body_start.startswith("{")
                       else _arrow_expression_end(lines, index, arrow_column)
                       if arrow_column >= 0 else number)
            kind = "component" if name[:1].isupper() else "function"
            symbols.append(Symbol(name, name, kind, number, end or number,
                                  signature=line.strip()[:120], end_known=end is not None))
            continue
        match = JS_INTERFACE.match(line)
        if match:
            end = block_end(lines, index)
            symbols.append(Symbol(match.group(1), match.group(1), "interface", number,
                                  end or number, signature=line.strip()[:120],
                                  end_known=end is not None))

    for index, line in enumerate(lines):
        number = index + 1
        if len(line) > MAX_PARSED_LINE:
            continue
        owner = next((r for r in class_ranges if r[0] < number <= r[1]), None)
        if not owner:
            continue
        match = JS_METHOD.match(line)
        if match and match.group(1) not in CALL_KEYWORDS:
            end = block_end(lines, index)
            name = match.group(1)
            symbols.append(Symbol(name, f"{owner[2]}.{name}", "method", number, end or number,
                                  parent=owner[2], signature=line.strip()[:120],
                                  end_known=end is not None))

    _finish_ends(symbols, len(lines))

    for match in JS_IMPORT_FROM.finditer(text):
        names = [n.strip().split(" as ")[-1].strip() for n in re.split(r"[{},]", match.group(1))
                 if n.strip() and n.strip() != "*"]
        out.imports.append(Import(match.group(2), names, _line_of(text, match.start())))
    for pattern in (JS_IMPORT_BARE, JS_REQUIRE, JS_DYNAMIC):
        for match in pattern.finditer(text):
            out.imports.append(Import(match.group(1), [], _line_of(text, match.start())))

    object_receivers = {
        m.group(1) for line in lines
        if (m := re.match(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*\{", line))
    }
    for match in JS_ROUTE.finditer(text):
        receiver, verb, _q, path, rest = match.groups()
        if (not ROUTER_NAMES.match(receiver) or not path.startswith("/")
                or receiver in object_receivers):
            continue
        rest = (rest or "").strip()
        # api.post('/path', {email, password}) is an HTTP client call, not an Express
        # route. A server route must pass a callback or a named handler as its next arg.
        if not rest or (rest.startswith("{") and "=>" not in rest and "function" not in rest):
            continue
        if "=>" in rest or re.search(r"\bfunction\b", rest):
            handler = "<inline>"
        else:
            first = rest.split(",", 1)[0].strip()
            handler_match = re.match(r"([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)", first)
            if not handler_match:
                continue
            handler = handler_match.group(1).split(".")[-1]
        out.routes.append(Route(verb.upper(), path, handler, _line_of(text, match.start()),
                                f"express ({receiver})"))

    for match in JS_FETCH.finditer(text):
        path = _path_from_literal(match.group(2))
        if path:
            window = text[match.end(): match.end() + 250]
            method = JS_METHOD_OPT.search(window)
            out.api_calls.append(ApiCall((method.group(1) if method else "GET").upper(), path,
                                         _line_of(text, match.start())))
    for match in JS_HTTP_CLIENT.finditer(text):
        path = _path_from_literal(match.group(3))
        if path:
            out.api_calls.append(ApiCall(match.group(1).upper(), path, _line_of(text, match.start())))

    if re.search(r"createRoot\(|ReactDOM\.render\(|createApp\(|app\.listen\(|new Vue\(", text):
        match = re.search(r"createRoot\(|ReactDOM\.render\(|createApp\(|app\.listen\(|new Vue\(", text)
        out.entry_points.append(EntryPoint("app-bootstrap", _line_of(text, match.start()),
                                           match.group(0).rstrip("(")))

    skip = {s.start for s in symbols if s.kind in ("function", "method", "component", "class")}
    _collect_line_calls(lines, symbols, skip, out)
    return out


JAVA_CLASS = re.compile(
    r"^\s*(?:(?:public|private|protected|abstract|final|static|sealed)\s+)*"
    r"(class|interface|enum|record)\s+(\w+)")
JAVA_METHOD = re.compile(
    r"^\s*(?:(?:public|private|protected|static|final|abstract|synchronized|native|default)\s+)+"
    r"(?:<[^>]+>\s+)?(?:([\w<>\[\],.?]+(?:\s*<[^>]*>)?)\s+)?(\w+)\s*\(([^)]*)\)?")
JAVA_IFACE_METHOD = re.compile(
    r"^\s*(?:<[^>]+>\s+)?([\w<>\[\],.?]+(?:\s*<[^>]*>)?)\s+(\w+)\s*\([^)]*\)\s*(?:throws[\w\s,.]+)?;")
JAVA_IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?([\w.*]+)\s*;", re.MULTILINE)
JAVA_MAPPING = re.compile(r"@(Get|Post|Put|Delete|Patch|Request)Mapping\s*(?:\(([^)]*)\))?")


def _mapping_path(args):
    if not args:
        return ""
    match = re.search(r"""(?:value|path)\s*=\s*\{?\s*"([^"]*)"|^\s*\{?\s*"([^"]*)"|\(\s*"([^"]*)"\s*\)""",
                      args)
    if match:
        return next(g for g in match.groups() if g is not None)
    return ""


def parse_java(text):
    lines = source_lines(text)
    out = Parsed(parser="java-scan")
    symbols = out.symbols
    class_stack = []        # (name, start, end, kind, prefix)

    def annotations_above(index):
        found = []
        i = index - 1
        while i >= 0 and lines[i].strip().startswith("@"):
            found.append(lines[i].strip())
            i -= 1
        return list(reversed(found))

    for index, line in enumerate(lines):
        number = index + 1
        if len(line) > MAX_PARSED_LINE:
            continue
        while class_stack and class_stack[-1][2] < number:
            class_stack.pop()
        match = JAVA_CLASS.match(line)
        if match:
            end = block_end(lines, index)
            name = match.group(2)
            annotations = annotations_above(index)
            prefix = ""
            for annotation in annotations:
                m = JAVA_MAPPING.match(annotation)
                if m and m.group(1) == "Request":
                    prefix = _mapping_path(m.group(2))
            parent = class_stack[-1][0] if class_stack else ""
            symbols.append(Symbol(name, f"{parent + '.' if parent else ''}{name}",
                                  "interface" if match.group(1) == "interface" else
                                  ("enum" if match.group(1) == "enum" else "class"),
                                  min([number] + [number - i - 1 for i in range(len(annotations))]),
                                  end or number, parent=parent, signature=line.strip()[:120],
                                  decorators=annotations[:4], end_known=end is not None))
            class_stack.append((name, number, end or number, match.group(1), prefix))
            if any("@SpringBootApplication" in a for a in annotations):
                out.entry_points.append(EntryPoint("spring-boot-app", number, f"class {name}"))
            continue

        owner = next((c for c in reversed(class_stack) if c[1] < number <= c[2]), None)
        if not owner:
            continue
        stripped = line.strip()
        method = JAVA_METHOD.match(line)
        name = None
        if method and not stripped.startswith(("return ", "new ", "throw ", "else")):
            name = method.group(2)
            if name in CALL_KEYWORDS and name != owner[0]:
                name = None
        if name is None and owner[3] == "interface":
            im = JAVA_IFACE_METHOD.match(line)
            if im:
                name = im.group(2)
        if not name:
            continue

        end = block_end(lines, index)
        annotations = annotations_above(index)
        symbols.append(Symbol(
            name, f"{owner[0]}.{name}", "method",
            min([number] + [number - i - 1 for i in range(len(annotations))]),
            end or number, parent=owner[0], signature=stripped[:140], decorators=annotations[:4],
            end_known=end is not None,
        ))
        for annotation in annotations:
            m = JAVA_MAPPING.match(annotation)
            if not m:
                continue
            verb = m.group(1).upper()
            if verb == "REQUEST":
                rm = re.search(r"RequestMethod\.(\w+)", m.group(2) or "")
                verb = rm.group(1).upper() if rm else "ANY"
            path = (owner[4].rstrip("/") + "/" + _mapping_path(m.group(2)).lstrip("/")).rstrip("/") \
                or "/"
            out.routes.append(Route(verb, path if path.startswith("/") else "/" + path, name,
                                    number, "spring"))
        if name == "main" and "static" in stripped and "void" in stripped:
            out.entry_points.append(EntryPoint("main()", number, f"{owner[0]}.main"))

    _finish_ends(symbols, len(lines))
    for match in JAVA_IMPORT.finditer(text):
        module = match.group(1)
        out.imports.append(Import(module, [module.rsplit(".", 1)[-1]], _line_of(text, match.start())))
    skip = {s.start for s in symbols}
    _collect_line_calls(lines, symbols, skip, out)
    return out


GENERIC_DEF = re.compile(
    r"^\s*(?:export\s+|pub\s+|public\s+|private\s+|static\s+)*"
    r"(?:async\s+)?(def|function|func|fn|class|struct|trait|interface|impl|type|module)\s+"
    r"(?:\([^)]*\)\s*)?([A-Za-z_][\w]*)")


def parse_generic(text):
    lines = source_lines(text)
    out = Parsed(parser="generic")
    for index, line in enumerate(lines):
        if len(line) > MAX_PARSED_LINE:
            continue
        match = GENERIC_DEF.match(line)
        if not match:
            continue
        keyword, name = match.groups()
        end = block_end(lines, index) if "{" in line or (index + 1 < len(lines) and "{" in lines[index + 1]) else None
        out.symbols.append(Symbol(
            name, name, "class" if keyword in ("class", "struct", "trait", "interface", "impl") else "function",
            index + 1, end or index + 1, signature=line.strip()[:120], end_known=end is not None))
    _finish_ends(out.symbols, len(lines))
    skip = {s.start for s in out.symbols}
    _collect_line_calls(lines, out.symbols, skip, out)
    return out


def parse_source(path, text):
    """Parse one file by its language. Never raises."""
    if len(text) > MAX_FILE_CHARS_PARSED:
        return Parsed(parser="skipped (too large)")
    language = language_of(path)
    try:
        if language == "python":
            return parse_python(text, path)
        if language in ("javascript", "typescript"):
            return parse_js(text)
        if language == "java":
            return parse_java(text)
        if language:
            return parse_generic(text)
    except Exception:       # a parser bug must never take indexing down
        try:
            parsed = parse_generic(text)
            parsed.parser = "generic (parser error)"
            return parsed
        except Exception:
            return Parsed(parser="failed")
    return Parsed(parser="none")


def parse_js(text):
    """Parse JS/TS and drop exact duplicate definitions (same name, start and end).

    Several recognisers can describe one declaration; a duplicate would make a cited range
    ambiguous for the index, the evidence packet and the verifier.
    """
    parsed = _parse_js_raw(text)
    seen, unique = set(), []
    for symbol in parsed.symbols:
        key = (symbol.qualname, symbol.start, symbol.end)
        if key not in seen:
            seen.add(key)
            unique.append(symbol)
    parsed.symbols[:] = unique
    return parsed

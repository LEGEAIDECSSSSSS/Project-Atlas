"""A structural index of an uploaded project, and the tools that navigate it.

The index records, for every source file: its symbols (with exact line ranges), imports, call
sites, route declarations, outgoing HTTP calls, entry points and whether it is a test. From
that the tools below answer the questions a plain text search cannot: who calls this, what
does this call, what imports what, which route reaches which handler and service, which
tests cover this symbol.

Every tool returns a plain dict. Anything list-shaped carries total / returned / has_more /
next_offset so a caller can never mistake a partial page for the whole answer.

Limits, stated honestly: calls and references are matched by name (there is no type
inference), so tools report every candidate and say when a name is ambiguous.
"""
import re
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import config
from services import code_parsers as cp
from services import project_service as ps
from services import vocab
from services.search_engine import TEST_PATH
from services.textlines import source_lines

MAX_INDEX_FILES = 6000
MAX_INDEX_FILE_BYTES = 500_000
MAX_CACHED_PROJECTS = 4
TRACE_MAX_NODES = 20
ROLE_PATTERNS = [
    ("test", re.compile(r"(?:^|/)(?:tests?|spec|__tests__)(?:/|$)|test_|_test\.|\.test\.|\.spec\.|Tests?\.java$", re.I)),
    ("controller", re.compile(r"controller|router|routes?|views?|handlers?|endpoints?|(?:^|/)api(?:/|$)|resource", re.I)),
    ("service", re.compile(r"service|usecase|manager|interactor|agent|logic", re.I)),
    ("data", re.compile(r"repositor|(?:^|/)dao|(?:^|/)db|database|models?|entit|schema|store|persistence", re.I)),
    ("util", re.compile(r"util|helper|common|lib/", re.I)),
    ("config", re.compile(r"config|settings|\.env|properties", re.I)),
]
MANIFESTS = {
    "readme.md", "readme.txt", "readme", "readme.rst", "pom.xml", "package.json",
    "requirements.txt", "pyproject.toml", "build.gradle", "build.gradle.kts", "cargo.toml",
    "go.mod", "composer.json", "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "pytest.ini", "setup.py", "setup.cfg", "tox.ini", "vite.config.js", "vite.config.ts",
    "tsconfig.json", "settings.gradle", "makefile",
}
FRAMEWORKS = [
    ("FastAPI", ("fastapi",)), ("Flask", ("flask",)), ("Django", ("django",)),
    ("Express", ("express",)), ("React", ("react",)), ("Vue", ("vue",)),
    ("Next.js", ("next",)), ("Angular", ("@angular",)), ("Spring Boot", ("spring-boot", "springframework")),
    ("SQLAlchemy", ("sqlalchemy",)), ("pytest", ("pytest",)), ("JUnit", ("junit",)),
    ("Jest", ("jest",)), ("Vitest", ("vitest",)), ("Vite", ("vite",)), ("Docker", ("dockerfile", "docker-compose")),
    ("Ollama", ("ollama",)), ("Tailwind", ("tailwind",)), ("Gin", ("gin-gonic",)),
    ("Rust/Cargo", ("[package]",)), ("Maven", ("<project",)), ("Gradle", ("build.gradle",)),
]


@dataclass
class FileInfo:
    path: str
    lang: str
    size: int
    line_count: int
    parsed: cp.Parsed
    is_test: bool
    role: str


@dataclass
class ProjectIndex:
    project_id: str
    signature: tuple
    files: dict = field(default_factory=dict)             # path -> FileInfo
    all_paths: list = field(default_factory=list)
    by_name: dict = field(default_factory=lambda: defaultdict(list))   # name -> [(path, Symbol)]
    calls_by_callee: dict = field(default_factory=lambda: defaultdict(list))
    routes: list = field(default_factory=list)            # [(path, Route)]
    api_calls: list = field(default_factory=list)         # [(path, ApiCall)]
    entry_points: list = field(default_factory=list)      # [(path, EntryPoint)]
    imports_resolved: dict = field(default_factory=dict)  # path -> [(Import, [resolved paths])]
    imported_by: dict = field(default_factory=lambda: defaultdict(set))
    router_prefixes: dict = field(default_factory=dict)  # path -> router variable -> URL prefix
    manifests: dict = field(default_factory=dict)         # path -> text (bounded)
    languages: dict = field(default_factory=dict)         # lang -> file count
    skipped_large: int = 0
    unparsed: int = 0


# --------------------------------------------------------------------------- building

def role_of(path):
    for role, pattern in ROLE_PATTERNS:
        if pattern.search(path):
            return role
    return "source"


def _signature(project_dir):
    count = total = newest = 0
    for path in ps._walk_files(project_dir):
        try:
            stat = path.stat()
        except OSError:
            continue
        count += 1
        total += stat.st_size
        newest = max(newest, stat.st_mtime_ns)
    return (count, total, newest)


def build_index(project_id):
    project_dir = ps.project_path(project_id)
    index = ProjectIndex(project_id=project_id, signature=_signature(project_dir))
    parsed_files = 0

    for absolute in ps._walk_files(project_dir):
        rel = absolute.relative_to(project_dir).as_posix()
        index.all_paths.append(rel)
        try:
            size = absolute.stat().st_size
        except OSError:
            continue

        name = absolute.name.lower()
        language = cp.language_of(rel)

        if name in MANIFESTS and size <= 200_000:
            try:
                index.manifests[rel] = ps.read_text_file(absolute)[:6000]
            except (ValueError, OSError):
                pass

        if not language:
            continue
        if size > MAX_INDEX_FILE_BYTES:
            index.skipped_large += 1
            continue
        if parsed_files >= MAX_INDEX_FILES:
            continue
        try:
            text = ps.read_text_file(absolute)
        except (ValueError, OSError):
            index.unparsed += 1
            continue

        parsed = cp.parse_source(rel, text)
        parsed_files += 1
        info = FileInfo(rel, language, size, len(source_lines(text)), parsed,
                        bool(TEST_PATH.search(rel)), role_of(rel))
        if language == "python":
            prefixes = {}
            for match in re.finditer(
                r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*(?:APIRouter|Blueprint|Router)\s*\((.*?)\)",
                text, re.DOTALL,
            ):
                prefix = re.search(r"\bprefix\s*=\s*(['\"])(.*?)\1", match.group(2), re.DOTALL)
                if prefix:
                    prefixes[match.group(1)] = prefix.group(2)
            if prefixes:
                index.router_prefixes[rel] = prefixes
        index.files[rel] = info
        index.languages[language] = index.languages.get(language, 0) + 1

        for symbol in parsed.symbols:
            index.by_name[symbol.name].append((rel, symbol))
        for call in parsed.calls:
            index.calls_by_callee[call.callee].append((rel, call))
        index.routes.extend((rel, r) for r in parsed.routes)
        index.api_calls.extend((rel, a) for a in parsed.api_calls)
        index.entry_points.extend((rel, e) for e in parsed.entry_points)

    index.all_paths.sort()
    for rel, info in index.files.items():
        resolved = []
        for imp in info.parsed.imports:
            targets = resolve_import(index, rel, info.lang, imp)
            resolved.append((imp, targets))
            for target in targets:
                index.imported_by[target].add(rel)
        index.imports_resolved[rel] = resolved
    return index


def _suffix_matches(index, suffix):
    suffix = suffix.strip("/")
    return [p for p in index.all_paths if p == suffix or p.endswith("/" + suffix)]


def resolve_import(index, from_path, language, imp):
    """Project files an import statement refers to (empty for external packages)."""
    module = imp.module
    found = []
    folder = str(PurePosixPath(from_path).parent)
    folder = "" if folder == "." else folder

    if language == "python":
        dots = len(module) - len(module.lstrip("."))
        bare = module.lstrip(".")
        if dots:
            base = PurePosixPath(folder or ".")
            for _ in range(dots - 1):
                base = base.parent
            prefix = "" if str(base) == "." else str(base)
            roots = [prefix]
        else:
            roots = [""]
        parts = bare.split(".") if bare else []
        names = [n for n in imp.names if n and n != "*"]
        attempts = []
        for root in roots:
            joined = "/".join(p for p in [root] + parts if p)
            attempts.append([joined + ".py", joined + "/__init__.py"] if joined else [])
            for name in names:
                sub = "/".join(p for p in [root] + parts + [name] if p)
                attempts.append([sub + ".py", sub + "/__init__.py"])
        for group in attempts:
            for candidate in group:
                if candidate in index.files:
                    found.append(candidate)
                elif not dots:
                    matches = _suffix_matches(index, candidate)
                    if len(matches) == 1 and matches[0] in index.files:
                        found.append(matches[0])
        return sorted(set(found))

    if language in ("javascript", "typescript"):
        if not module.startswith((".", "@/", "~/")):
            return []
        if module.startswith(("@/", "~/")):
            bases = ["src/" + module[2:], module[2:]]
        else:
            bases = [str(PurePosixPath(folder) / module) if folder else module]
        for base in bases:
            norm = str(PurePosixPath(base))
            parts = []
            for part in norm.split("/"):
                if part == "..":
                    if parts:
                        parts.pop()
                elif part not in (".", ""):
                    parts.append(part)
            norm = "/".join(parts)
            for ext in ("", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
                        "/index.js", "/index.jsx", "/index.ts", "/index.tsx"):
                if norm + ext in index.files:
                    found.append(norm + ext)
                    break
        return found

    if language == "java":
        if module.endswith("*"):
            return []
        candidate = module.replace(".", "/") + ".java"
        matches = _suffix_matches(index, candidate)
        return [m for m in matches if m in index.files][:1]

    return []


_CACHE = OrderedDict()
_LOCK = threading.Lock()


def get_index(project_id):
    """The cached index of a project, rebuilt when any file changed."""
    project_dir = ps.project_path(project_id)
    signature = _signature(project_dir)
    with _LOCK:
        cached = _CACHE.get(project_id)
        if cached and cached.signature == signature:
            _CACHE.move_to_end(project_id)
            return cached
    index = build_index(project_id)
    index.signature = signature
    with _LOCK:
        _CACHE[project_id] = index
        _CACHE.move_to_end(project_id)
        while len(_CACHE) > MAX_CACHED_PROJECTS:
            _CACHE.popitem(last=False)
    return index


def clear_cache(project_id=None):
    with _LOCK:
        if project_id is None:
            _CACHE.clear()
        else:
            _CACHE.pop(project_id, None)


# --------------------------------------------------------------------------- helpers

def _page(items, limit, offset):
    limit = max(1, min(int(limit or 20), 100))
    offset = max(0, int(offset or 0))
    page = items[offset: offset + limit]
    more = offset + len(page) < len(items)
    return page, {
        "total": len(items),
        "returned": len(page),
        "offset": offset,
        "limit": limit,
        "has_more": more,
        "next_offset": offset + len(page) if more else None,
    }


def _last_name(name):
    name = (name or "").strip()
    if "(" in name:
        name = name.split("(", 1)[0]
    return re.split(r"[.:#/]|->|::", name.replace("()", ""))[-1].strip() or name


def _symbol_dict(path, symbol, match_type="exact"):
    return {
        "name": symbol.name, "qualname": symbol.qualname, "kind": symbol.kind,
        "path": path, "start_line": symbol.start, "end_line": symbol.end,
        "signature": symbol.signature, "doc": symbol.doc, "match_type": match_type,
    }


def _match_path(index, wanted):
    """The one project path a typed path refers to, or raises ValueError."""
    wanted = (wanted or "").replace("\\", "/").strip().lstrip("./")
    if wanted in index.files or wanted in index.all_paths:
        return wanted
    matches = [p for p in index.all_paths if p.endswith("/" + wanted) or p.lower() == wanted.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f'"{wanted}" was not found in the project.')
    raise ValueError(f'"{wanted}" matches several files: ' + ", ".join(matches[:6]))


def _symbols_for(index, name, path=None):
    last = _last_name(name)
    found = list(index.by_name.get(last, []))
    if "." in name or "::" in name or "#" in name:
        qual = re.sub(r"[:#]+", ".", name.replace("()", "")).strip(".")
        narrowed = [(p, s) for p, s in found if s.qualname == qual or s.qualname.endswith("." + qual)]
        found = narrowed or found
    if path:
        wanted = _match_path(index, path)
        found = [(p, s) for p, s in found if p == wanted]
    return sorted(found, key=lambda item: (item[0], item[1].start))


def _enclosing(index, path, line):
    info = index.files.get(path)
    if not info:
        return None
    inside = [s for s in info.parsed.symbols if s.start <= line <= s.end]
    if not inside:
        return None
    return min(inside, key=lambda s: (s.end - s.start, -s.start))


# --------------------------------------------------------------------------- tools

def find_symbol(project_id, name, kind=None, path=None, limit=20, offset=0):
    index = get_index(project_id)
    if not name or not str(name).strip():
        raise ValueError("A symbol name is required.")
    name = str(name).strip()
    found = [(p, s, "exact") for p, s in _symbols_for(index, name, path)]

    if not found:
        wanted = set(vocab.split_identifier(_last_name(name)))
        if wanted:
            for symbol_name, entries in index.by_name.items():
                tokens = set(vocab.split_identifier(symbol_name))
                if wanted <= tokens or symbol_name.lower().startswith(_last_name(name).lower()):
                    found.extend((p, s, "partial") for p, s in entries)
            if path:
                wanted_path = _match_path(index, path)
                found = [f for f in found if f[0] == wanted_path]

    if kind:
        found = [f for f in found if f[1].kind == kind]
    found.sort(key=lambda f: (f[2] != "exact", f[0], f[1].start))
    page, meta = _page(found, limit, offset)
    return {"project_id": project_id, "name": name,
            "symbols": [_symbol_dict(p, s, t) for p, s, t in page], **meta,
            "note": None if found else f"No symbol named '{name}' (or containing those words) is "
                                       "defined in the indexed source files."}


def _import_header(info, start_line):
    lines = []
    for imp in info.parsed.imports:
        if imp.line < start_line and imp.line <= 80:
            lines.append(imp.line)
    return sorted(set(lines))


def read_symbol(project_id, path=None, name=None, line=None, max_chars=None):
    """Read one complete function/class (with the file's import lines), by name or by line."""
    index = get_index(project_id)
    max_chars = int(max_chars or config.MAX_FILE_CHARS)
    symbol = None
    note = None

    if name:
        found = _symbols_for(index, str(name), path)
        if not found:
            raise ValueError(f"No definition named '{name}'"
                             + (f" in {path}" if path else "") + " was found in the index.")
        distinct = sorted({p for p, _ in found})
        if len(distinct) > 1 and not path:
            raise ValueError(f"'{name}' is defined in several files: {', '.join(distinct[:6])}. "
                             "Pass 'path' to choose one.")
        path, symbol = found[0]
        if line is not None:
            inside = [s for p, s in found if p == path and s.start <= int(line) <= s.end]
            symbol = inside[0] if inside else symbol
    elif path is not None and line is not None:
        path = _match_path(index, path)
        symbol = _enclosing(index, path, int(line))
    else:
        raise ValueError("Give 'name', or both 'path' and 'line'.")

    text = ps.read_project_file(project_id, path)
    lines = source_lines(text)
    total = len(lines)
    info = index.files.get(path)

    if symbol is None:
        centre = int(line)
        start, end = max(1, centre - 25), min(total, centre + 25)
        kind_name, qual, symbol_end = "excerpt", f"{path}:{centre}", end
        note = "No function or class contains that line, so nearby lines are shown."
    else:
        start, end = symbol.start, min(symbol.end, total)
        kind_name, qual, symbol_end = symbol.kind, symbol.qualname, end

    body_lines = lines[start - 1:end]
    body = "\n".join(body_lines)
    truncated = False
    if len(body) > max_chars:
        cut = body[:max_chars]
        cut = cut[:cut.rfind("\n")] if "\n" in cut else cut
        end = start + cut.count("\n")
        body = cut
        truncated = True

    imports = ""
    header_lines = _import_header(info, start) if info else []
    if header_lines and start > 1:
        imports = "\n".join(lines[n - 1] for n in header_lines if n - 1 < total)[:1200]

    return {
        "project_id": project_id, "path": path, "name": symbol.name if symbol else (name or ""),
        "qualname": qual, "kind": kind_name, "total_lines": total,
        "start_line": start, "end_line": end, "symbol_end_line": symbol_end,
        "complete": not truncated, "truncated": truncated,
        "next_start_line": end + 1 if truncated else None,
        "imports": imports, "content": body, "note": note,
    }


def symbols_for_lines(project_id, path, lines):
    """The definitions that contain the given lines, best first.

    Ranked by how many of the lines fall inside (the part of the file the search is really about),
    then by size, so a question about "chat message processing" lands on the main handler rather
    than on a three-line helper that happens to match once.
    """
    index = get_index(project_id)
    counts = {}
    for line in lines:
        symbol = _enclosing(index, path, int(line))
        if symbol is not None:
            entry = counts.setdefault(symbol.qualname, [symbol, 0])
            entry[1] += 1
    ranked = sorted(counts.values(), key=lambda e: (-e[1], -(e[0].end - e[0].start), e[0].start))
    return [{"qualname": s.qualname, "kind": s.kind, "start": s.start, "end": s.end, "count": c}
            for s, c in ranked]


_ident_cache = {}


def _identifier_regex(name):
    regex = _ident_cache.get(name)
    if regex is None:
        regex = re.compile(r"(?<![A-Za-z0-9_$])" + re.escape(name) + r"(?![A-Za-z0-9_$])")
        if len(_ident_cache) > 500:
            _ident_cache.clear()
        _ident_cache[name] = regex
    return regex


def _iter_text_files(project_id, index, only_tests=False):
    project_dir = ps.project_path(project_id)
    for rel in index.all_paths:
        info = index.files.get(rel)
        if only_tests and not (info and info.is_test):
            continue
        absolute = project_dir / rel
        name = absolute.name
        try:
            size = absolute.stat().st_size
        except OSError:
            continue
        if size > ps.MAX_SEARCH_FILE_BYTES or not ps.is_text_candidate(name, size):
            continue
        try:
            yield rel, ps.read_text_file(absolute)
        except (ValueError, OSError):
            continue


def _classify_reference(index, path, line_no, content, name, regex):
    info = index.files.get(path)
    stripped = content.strip()
    if info:
        for symbol in info.parsed.symbols:
            if symbol.name == name and symbol.start <= line_no <= max(symbol.start, symbol.start + 3) \
                    and re.match(r"\s*(?:@|export|async|public|private|protected|static|def|class|function|"
                                 r"const|let|var|interface|enum|func|fn)", stripped):
                return "definition"
        for imp in info.parsed.imports:
            if imp.line == line_no:
                return "import"
    if stripped.startswith(("#", "//", "*", "/*")):
        return "comment"
    if re.search(r"(?<![A-Za-z0-9_$])" + re.escape(name) + r"\s*\(", stripped):
        return "call"
    return "other"


def find_references(project_id, name, limit=30, offset=0):
    index = get_index(project_id)
    name = _last_name(str(name or ""))
    if not name:
        raise ValueError("A name is required.")
    regex = _identifier_regex(name)
    refs = []
    for rel, text in _iter_text_files(project_id, index):
        if name not in text:
            continue
        for number, line in enumerate(source_lines(text), start=1):
            if regex.search(line):
                refs.append({"path": rel, "line": number,
                             "content": line.strip()[:240],
                             "kind": _classify_reference(index, rel, number, line, name, regex)})
    page, meta = _page(refs, limit, offset)
    summary = defaultdict(int)
    for ref in refs:
        summary[ref["kind"]] += 1
    return {"project_id": project_id, "name": name, "references": page, "by_kind": dict(summary),
            **meta}


def find_callers(project_id, name, limit=30, offset=0):
    index = get_index(project_id)
    name = _last_name(str(name or ""))
    if not name:
        raise ValueError("A name is required.")
    definitions = [(p, s) for p, s in index.by_name.get(name, []) if s.kind != "interface"]
    entries = []
    for rel, call in index.calls_by_callee.get(name, []):
        entries.append({"caller": call.caller, "path": rel, "line": call.line,
                        "qualifier": call.qualifier,
                        "caller_is_test": index.files[rel].is_test})
    entries.sort(key=lambda e: (e["caller_is_test"], e["path"], e["line"]))
    page, meta = _page(entries, limit, offset)
    note = None
    if not definitions:
        note = f"'{name}' is not defined in the indexed project files; these are calls to an external or unknown function."
    elif len({p for p, _ in definitions}) > 1 or len(definitions) > 1:
        note = f"'{name}' is defined {len(definitions)} times, so callers are matched by name and may include calls to a different definition."
    return {"project_id": project_id, "name": name,
            "definitions": [_symbol_dict(p, s) for p, s in definitions[:6]],
            "callers": page, "note": note, **meta}


def _self_attribute_type(index, from_path, call):
    """Infer a self/cls attribute's constructed type from its owning class initializer."""
    qualifier = (call.qualifier or "").strip()
    match = re.match(r"^(?:self|cls)\.([A-Za-z_]\w*)$", qualifier)
    if not match or "." not in (call.caller or ""):
        return None
    attribute = match.group(1)
    owner_name = call.caller.rsplit(".", 1)[0]
    owner = next((symbol for path, symbol in index.by_name.get(owner_name, [])
                  if path == from_path and symbol.kind == "class"), None)
    if owner is None:
        return None
    try:
        source = ps.read_project_file(index.project_id, from_path)
        class_lines = source_lines(source)[owner.start - 1:owner.end]
    except (OSError, ValueError, FileNotFoundError):
        return None

    prefix = rf"(?:self|cls)\.{re.escape(attribute)}"
    constructed = re.search(
        prefix + rf"\s*(?::\s*[A-Za-z_]\w*)?\s*=\s*"
        r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(",
        "\n".join(class_lines),
    )
    if constructed:
        return constructed.group(1).rsplit(".", 1)[-1]
    annotated = re.search(prefix + r"\s*:\s*([A-Za-z_]\w*)", "\n".join(class_lines))
    return annotated.group(1) if annotated else None


def _resolve_callee(index, from_path, call):
    candidates = [(p, s) for p, s in index.by_name.get(call.callee, [])
                  if s.kind in ("function", "method", "class", "component")]
    if not candidates:
        return []

    # Attribute calls such as service.login() must not resolve back to the current
    # controller's login() merely because the names match. Likewise, request.headers.get()
    # must not be linked to an unrelated repository.get() elsewhere in the project.
    qualifier = (call.qualifier or "").strip()
    same = [c for c in candidates if c[0] == from_path]
    if qualifier and qualifier not in {"self", "cls", "super", "this"}:
        same = [c for c in same if c[1].qualname != call.caller]
    if same:
        return same

    imported = {t for _imp, targets in index.imports_resolved.get(from_path, []) for t in targets}
    near = [c for c in candidates if c[0] in imported]
    if near:
        receiver_type = _self_attribute_type(index, from_path, call)
        if receiver_type:
            typed = [c for c in near if c[1].parent == receiver_type
                     or c[1].qualname.startswith(receiver_type + ".")]
            if typed:
                return typed
        return near

    # A simple receiver (e.g. an injected `service`) may have a type supplied through a
    # dependency alias rather than a direct import. Keep the fallback for that case, but
    # never guess targets for dotted framework/data objects such as request.headers.
    if qualifier and "." in qualifier:
        return []
    if qualifier:
        external = [c for c in candidates if c[0] != from_path]
        # A simple variable named `service` usually holds a service-layer dependency.
        # Prefer that layer over unrelated same-named frontend callbacks or controllers.
        if qualifier.lower() in {"service", "authservice", "userservice", "repository", "repo"}:
            preferred = [c for c in external if index.files.get(c[0])
                         and index.files[c[0]].role in {"service", "data"}]
            return preferred or external
        return external
    return candidates


def find_callees(project_id, name, path=None, limit=40, offset=0):
    index = get_index(project_id)
    found = _symbols_for(index, str(name or ""), path)
    found = [(p, s) for p, s in found if s.kind in ("function", "method", "component", "class")]
    if not found:
        raise ValueError(f"No function named '{name}' was found in the index.")
    distinct = sorted({p for p, _ in found})
    if len(distinct) > 1 and not path:
        raise ValueError(f"'{name}' is defined in several files: {', '.join(distinct[:6])}. "
                         "Pass 'path' to choose one.")
    rel, symbol = found[0]
    info = index.files[rel]
    seen, callees = set(), []
    for call in info.parsed.calls:
        if call.caller != symbol.qualname and not (symbol.kind == "class"
                                                    and call.caller.startswith(symbol.qualname + ".")):
            continue
        key = (call.callee, call.line)
        if key in seen:
            continue
        seen.add(key)
        targets = _resolve_callee(index, rel, call)
        callees.append({
            "callee": call.callee, "line": call.line, "qualifier": call.qualifier,
            "resolved": [{"path": p, "qualname": s.qualname, "kind": s.kind,
                          "start_line": s.start, "end_line": s.end} for p, s in targets[:4]],
            "external": not targets,
        })
    page, meta = _page(callees, limit, offset)
    return {"project_id": project_id, "name": symbol.qualname, "path": rel,
            "start_line": symbol.start, "end_line": symbol.end, "callees": page, **meta}


def find_imports(project_id, path):
    index = get_index(project_id)
    rel = _match_path(index, path)
    info = index.files.get(rel)
    imports = []
    if info:
        for imp, targets in index.imports_resolved.get(rel, []):
            imports.append({"module": imp.module, "names": imp.names[:10], "line": imp.line,
                            "resolved_paths": targets, "external": not targets})
    return {"project_id": project_id, "path": rel, "imports": imports,
            "imported_by": sorted(index.imported_by.get(rel, set()))[:50],
            "indexed": info is not None,
            "note": None if info else "This file type is not parsed, so its imports are unknown."}


def _norm_route(path):
    text = re.sub(r"\{[^}/]*\}|:[A-Za-z_]\w*|<[^>]*>|\$\{[^}]*\}", "{}", path or "")
    return text.rstrip("/") or "/"


def _role_label(path, symbol_name=""):
    return role_of(path + "/" + symbol_name)


def find_entry_points(project_id):
    index = get_index(project_id)
    items = [{"path": p, "line": e.line, "kind": e.kind, "detail": e.detail}
             for p, e in index.entry_points if not (index.files.get(p) and index.files[p].is_test)]
    for path, text in index.manifests.items():
        name = path.rsplit("/", 1)[-1].lower()
        if name == "package.json":
            for key in ("main", "module"):
                m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', text)
                if m:
                    items.append({"path": path, "line": 1, "kind": f"package.json {key}", "detail": m.group(1)})
            scripts = re.search(r'"scripts"\s*:\s*\{([^}]*)\}', text, re.S)
            if scripts:
                for sm in re.finditer(r'"(start|dev|serve|build|test)"\s*:\s*"([^"]+)"', scripts.group(1)):
                    items.append({"path": path, "line": 1, "kind": f"npm script '{sm.group(1)}'",
                                  "detail": sm.group(2)})
        elif name == "dockerfile":
            for dm in re.finditer(r"^\s*(CMD|ENTRYPOINT)\s+(.+)$", text, re.M):
                items.append({"path": path, "line": text[:dm.start()].count("\n") + 1,
                              "kind": dm.group(1), "detail": dm.group(2)[:120]})
    order = {"app-object": 0, "main-guard": 1, "main()": 2, "spring-boot-app": 0, "app-bootstrap": 1}
    items.sort(key=lambda i: (order.get(i["kind"], 5), i["path"], i["line"]))
    return {"project_id": project_id, "entry_points": items[:40], "total": len(items),
            "has_more": len(items) > 40,
            "note": None if items else "No entry points were detected (no main guard, app object, main method or start script)."}


def _handler_symbol(index, path, handler):
    if not handler or handler.startswith("<"):
        return None
    entries = [(p, s) for p, s in index.by_name.get(_last_name(handler), [])]
    same = [e for e in entries if e[0] == path]
    chosen = same or entries
    return chosen[0] if chosen else None


def _chain_from(index, start_path, start_symbol, depth):
    """Breadth-first walk of resolved callees, bounded in depth and node count."""
    nodes, edges = [], []
    seen = {(start_path, start_symbol.qualname)}
    frontier = [(start_path, start_symbol, 0)]
    while frontier and len(nodes) < TRACE_MAX_NODES:
        path, symbol, level = frontier.pop(0)
        if level >= depth:
            continue
        info = index.files.get(path)
        if not info:
            continue
        for call in info.parsed.calls:
            if call.caller != symbol.qualname:
                continue
            for target_path, target in _resolve_callee(index, path, call)[:2]:
                key = (target_path, target.qualname)
                edges.append({"from": symbol.qualname, "to": target.qualname,
                              "path": path, "line": call.line})
                if key in seen:
                    continue
                seen.add(key)
                nodes.append({"role": _role_label(target_path, target.name), "path": target_path,
                              "symbol": target.qualname, "kind": target.kind,
                              "start_line": target.start, "end_line": target.end,
                              "called_from": f"{path}:{call.line}"})
                frontier.append((target_path, target, level + 1))
                if len(nodes) >= TRACE_MAX_NODES:
                    break
            if len(nodes) >= TRACE_MAX_NODES:
                break
    return nodes, edges


def _effective_route_path(index, path, route):
    receiver_match = re.search(r"\(([^)]+)\)", route.framework or "")
    receiver = receiver_match.group(1).strip() if receiver_match else ""
    prefix = index.router_prefixes.get(path, {}).get(receiver, "")
    return _norm_route((prefix.rstrip("/") + "/" + route.path.lstrip("/")) if prefix else route.path)


def _api_path_matches(route_path, api_path):
    route_path, api_path = _norm_route(route_path), _norm_route(api_path)
    return bool(route_path and api_path and (
        route_path == api_path or route_path.endswith(api_path)
        or api_path.endswith(route_path)
    ))


def _enclosing_symbols(index, path, line):
    info = index.files.get(path)
    if not info:
        return []
    return sorted(
        [symbol for symbol in info.parsed.symbols if symbol.start <= line <= symbol.end],
        key=lambda symbol: (symbol.end - symbol.start, -symbol.start),
    )


def trace_route(project_id, query, depth=3, limit=3):
    """Trace a backend route, including router prefixes and frontend API callers.

    Frontend HTTP-client calls are not server routes. Their enclosing functions and
    caller components are returned separately so the UI -> API -> backend path stays clear.
    """
    index = get_index(project_id)
    query = str(query or "").strip()
    if not query:
        raise ValueError("A route path, handler name or keyword is required.")
    depth = max(1, min(int(depth or 3), 4))

    words = [w for w in vocab.split_identifier(query) if w not in vocab.STOPWORDS
             and w not in vocab.QUESTION_NOISE] or vocab.split_identifier(query)
    wanted_path = query if query.startswith("/") else None

    scored = []
    for path, route in index.routes:
        effective_path = _effective_route_path(index, path, route)
        score = 0
        route_tokens = set(vocab.split_identifier(route.path + " " + effective_path))
        handler_tokens = set(vocab.split_identifier(route.handler))
        file_tokens = set(vocab.split_identifier(path))
        if wanted_path and effective_path == _norm_route(wanted_path):
            score += 30
        elif wanted_path and _norm_route(wanted_path).strip("/") and (
            _norm_route(wanted_path).strip("/") in effective_path.strip("/")
        ):
            score += 10
        for word in words:
            if word in route_tokens:
                score += 4
            if word in handler_tokens:
                score += 4
            if word in file_tokens:
                score += 1
        if score:
            scored.append((-score, path, route.line, route, effective_path))
    scored.sort(key=lambda item: item[:3])

    results = []
    for _neg, path, _line, route, effective_path in scored[:max(1, min(int(limit or 3), 5))]:
        entry = _handler_symbol(index, path, route.handler)
        chain = [{"role": "route", "path": path, "symbol": route.handler, "kind": "route",
                  "start_line": route.line, "end_line": route.line}]
        edges, files = [], [(path, route.line, route.line)]
        if entry:
            handler_path, handler = entry
            chain[0].update({"path": handler_path, "start_line": handler.start,
                             "end_line": handler.end, "symbol": handler.qualname,
                             "role": _role_label(handler_path, handler.name)
                             if _role_label(handler_path) != "source" else "handler"})
            files = [(handler_path, handler.start, handler.end)]
            nodes, edges = _chain_from(index, handler_path, handler, depth)
            chain.extend(nodes)
            files.extend((n["path"], n["start_line"], n["end_line"]) for n in nodes)

        frontend = []
        frontend_chains = []
        frontend_providers = []
        frontend_ui_callers = []
        frontend_ui_chains = []
        for frontend_path, info in index.files.items():
            # Discover client-side request sites from parsed HTTP calls, not directory names.
            # Projects commonly wrap their source tree in an archive root (or use client/,
            # web/, ui/, etc.); a path-prefix check silently loses those callers.
            if info.lang not in {"javascript", "typescript"} or info.parsed.routes:
                continue
            for api_call in info.parsed.api_calls:
                if api_call.method not in (route.method, "GET" if route.method in ("ANY", "ALL") else route.method):
                    continue
                if not _api_path_matches(effective_path, api_call.path):
                    continue
                enclosing = _enclosing_symbols(index, frontend_path, api_call.line)
                leaf = enclosing[0] if enclosing else None
                caller_record = {"path": frontend_path, "line": api_call.line,
                                 "method": api_call.method, "request_path": api_call.path}
                if leaf:
                    caller_record.update({"symbol": leaf.qualname, "start_line": leaf.start,
                                          "end_line": leaf.end})
                    for parent in enclosing[1:]:
                        if parent.kind == "component":
                            provider = {"path": frontend_path, "symbol": parent.qualname,
                                        "start_line": parent.start, "end_line": parent.end}
                            if provider not in frontend_providers:
                                frontend_providers.append(provider)
                    nodes, frontend_edges = _chain_from(index, frontend_path, leaf, min(depth, 2))
                    frontend_chains.append({"path": frontend_path, "symbol": leaf.qualname,
                                            "start_line": leaf.start, "end_line": leaf.end,
                                            "api_call_line": api_call.line,
                                            "chain": nodes, "edges": frontend_edges})
                    files.append((frontend_path, leaf.start, leaf.end))
                    files.extend((n["path"], n["start_line"], n["end_line"]) for n in nodes)

                    # The page-level submit handler usually calls the context callback. Find
                    # actual frontend callers of that callback and include their enclosing UI.
                    for call_path, call in index.calls_by_callee.get(leaf.name, []):
                        caller_info = index.files.get(call_path)
                        if not caller_info or caller_info.lang != info.lang:
                            continue
                        call_symbols = _enclosing_symbols(index, call_path, call.line)
                        if not call_symbols:
                            continue
                        ui_leaf = call_symbols[0]
                        item = {"path": call_path, "symbol": ui_leaf.qualname,
                                "kind": ui_leaf.kind, "start_line": ui_leaf.start,
                                "end_line": ui_leaf.end, "called_at": call.line,
                                "is_parent": False}
                        if item not in frontend_ui_callers:
                            frontend_ui_callers.append(item)
                        for parent in call_symbols[1:]:
                            if parent.kind == "component":
                                parent_item = {"path": call_path, "symbol": parent.qualname,
                                               "kind": parent.kind, "start_line": parent.start,
                                               "end_line": parent.end, "called_at": call.line,
                                               "is_parent": True}
                                if parent_item not in frontend_ui_callers:
                                    frontend_ui_callers.append(parent_item)
                                ui_nodes, ui_edges = _chain_from(index, call_path, parent, 1)
                                frontend_ui_chains.append({"path": call_path, "symbol": parent.qualname,
                                                           "chain": ui_nodes, "edges": ui_edges})
                        files.append((call_path, ui_leaf.start, ui_leaf.end))
                frontend.append(caller_record)

        results.append({
            "method": route.method, "route": route.path, "effective_route": effective_path,
            "declared_at": f"{path}:{route.line}", "framework": route.framework,
            "handler": route.handler, "handler_resolved": entry is not None,
            "chain": chain, "edges": edges[:30], "frontend_callers": frontend[:6],
            "frontend_chains": frontend_chains[:6],
            "frontend_providers": frontend_providers[:6],
            "frontend_ui_callers": frontend_ui_callers[:8],
            "frontend_ui_chains": frontend_ui_chains[:8],
            "files_to_read": [{"path": p, "start_line": a, "end_line": b} for p, a, b in files],
        })

    fallback = []
    if not results:
        for symbol_name, entries in index.by_name.items():
            tokens = set(vocab.split_identifier(symbol_name))
            if words and all(w in tokens for w in words[:2]):
                fallback.extend(_symbol_dict(p, s, "name") for p, s in entries
                                if s.kind in ("function", "method", "component"))
        fallback = fallback[:10]
    return {"project_id": project_id, "query": query, "routes": results,
            "total_routes_indexed": len(index.routes),
            "matching_routes": len(scored), "related_symbols": fallback,
            "note": None if results else
            ("No declared route matches; related function names are listed instead."
             if fallback else "No route declaration or function matched. Routes are detected for "
             "FastAPI/Flask/Django, Express and Spring; try search_project.")}


def _likely_test_files(index, path):
    stem = PurePosixPath(path).stem
    out = []
    for rel, info in index.files.items():
        if not info.is_test:
            continue
        test_stem = PurePosixPath(rel).stem.lower()
        wanted = stem.lower()
        if test_stem in (f"test_{wanted}", f"{wanted}_test", f"{wanted}.test", f"{wanted}test",
                         f"{wanted}tests", f"{wanted}.spec") or \
                test_stem.replace("test_", "").replace("_test", "") == wanted:
            out.append(rel)
    return sorted(out)


def find_tests_for_symbol(project_id, name, limit=20):
    index = get_index(project_id)
    last = _last_name(str(name or ""))
    if not last:
        raise ValueError("A name is required.")
    regex = _identifier_regex(last)
    definitions = [(p, s) for p, s in index.by_name.get(last, [])]
    references = []
    for rel, text in _iter_text_files(project_id, index, only_tests=True):
        if last not in text:
            continue
        hits = []
        for number, line in enumerate(source_lines(text), start=1):
            if regex.search(line):
                hits.append({"line": number, "content": line.strip()[:200]})
        if hits:
            enclosing = []
            for hit in hits[:5]:
                owner = _enclosing(index, rel, hit["line"])
                if owner:
                    enclosing.append(owner.qualname)
            references.append({"path": rel, "references": len(hits), "examples": hits[:3],
                               "test_functions": sorted(set(enclosing))[:5]})
    references.sort(key=lambda r: (-r["references"], r["path"]))
    likely = sorted({t for p, _ in definitions for t in _likely_test_files(index, p)})
    page, meta = _page(references, limit, 0)
    note = None
    if not references and not likely:
        note = ("No test file mentions this name. Tests may call it indirectly (through an API route "
                "or another function), so this does not prove it is untested.")
    return {"project_id": project_id, "name": last,
            "defined_in": [{"path": p, "start_line": s.start} for p, s in definitions[:5]],
            "test_files": page, "likely_tests_by_file_name": likely[:10], "note": note, **meta}


# --------------------------------------------------------------------------- project map

def _readme_summary(index):
    for path in sorted(index.manifests, key=lambda p: (p.count("/"), p)):
        if path.rsplit("/", 1)[-1].lower().startswith("readme"):
            text = index.manifests[path]
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            for paragraph in paragraphs:
                plain = re.sub(r"[#*`>\[\]!]+", "", paragraph).strip()
                if len(plain) > 30 and not plain.lower().startswith(("badge", "http")):
                    return path, " ".join(plain.split())[:300]
    return None, ""


def _detect_frameworks(index):
    blob = "\n".join(
        f"{p.rsplit('/', 1)[-1].lower()}\n{t.lower()}" for p, t in index.manifests.items()
        if p.rsplit("/", 1)[-1].lower() != "readme.md")
    found = []
    for label, needles in FRAMEWORKS:
        if any(n in blob for n in needles):
            found.append(label)
    for info in index.files.values():
        for imp in info.parsed.imports[:40]:
            head = imp.module.lstrip(".").split(".")[0].split("/")[0].lower()
            for label, needles in FRAMEWORKS:
                if head in needles and label not in found:
                    found.append(label)
    return found[:10]


def project_map_text(project_id, budget=2400):
    """A compact, factual map of the project, built from the index. Shorter than a file list."""
    index = get_index(project_id)
    lines = ["PROJECT MAP (generated automatically from the files; confirm details with the tools):"]
    readme_path, purpose = _readme_summary(index)
    if purpose:
        lines.append(f"Purpose (from {readme_path}): {purpose}")
    langs = sorted(index.languages.items(), key=lambda kv: (-kv[1], kv[0]))
    if langs:
        lines.append("Languages: " + ", ".join(f"{k} ({v} files)" for k, v in langs[:5]))
    frameworks = _detect_frameworks(index)
    if frameworks:
        lines.append("Frameworks/tools: " + ", ".join(frameworks))

    entry = find_entry_points(project_id)["entry_points"][:6]
    if entry:
        lines.append("Entry points: " + "; ".join(
            f"{e['path']}:{e['line']} ({e['kind']})" for e in entry))

    if index.routes:
        shown = []
        for path, route in sorted(index.routes, key=lambda r: (r[0], r[1].line))[:12]:
            shown.append(f"{route.method} {route.path} -> {route.handler} ({path}:{route.line})")
        extra = f" (+{len(index.routes) - 12} more)" if len(index.routes) > 12 else ""
        lines.append("Routes: " + "; ".join(shown) + extra)

    folders = defaultdict(lambda: [0, defaultdict(int)])
    for path in index.all_paths:
        parts = path.split("/")
        key = "/".join(parts[:2]) if len(parts) > 2 else (parts[0] if len(parts) > 1 else ".")
        folders[key][0] += 1
        role = role_of(path)
        if role != "source":
            folders[key][1][role] += 1
    summary = []
    for key, (count, roles) in sorted(folders.items(), key=lambda kv: (-kv[1][0], kv[0]))[:10]:
        top_role = max(roles, key=roles.get) if roles else ""
        summary.append(f"{key}/ ({count} files{', ' + top_role if top_role else ''})")
    if summary:
        lines.append("Folders: " + "; ".join(summary))

    tests = sorted({PurePosixPath(p).parent.as_posix() for p, i in index.files.items() if i.is_test})
    if tests:
        lines.append("Tests: " + ", ".join(f"{t}/" for t in tests[:4]))

    key_files = []
    for path, info in sorted(index.files.items()):
        if info.is_test or info.role == "source" and not info.parsed.symbols:
            continue
        top = [s.name for s in info.parsed.symbols if s.kind in ("class", "function", "component")
               and not s.parent][:5]
        if top and info.role in ("controller", "service", "data"):
            key_files.append(f"{path}: {', '.join(top)}")
    if key_files:
        lines.append("Key modules: " + "; ".join(key_files[:8]))

    manifests = [p for p in sorted(index.manifests) if "readme" not in p.lower()][:8]
    if manifests:
        lines.append("Manifests/config: " + ", ".join(manifests))
    if index.skipped_large or index.unparsed:
        lines.append(f"Not indexed: {index.skipped_large} oversized and {index.unparsed} unreadable source files.")

    text = "\n".join(lines)
    while len(text) > budget and len(lines) > 2:
        lines.pop(-2 if len(lines) > 3 else -1)
        text = "\n".join(lines)
    return text[:budget]


def project_facts(project_id):
    """Names and edges used to check an answer: symbol names, files, call edges, line counts."""
    index = get_index(project_id)
    cached = getattr(index, "_facts", None)
    if cached is not None:
        return cached
    edges = set()
    for callee, entries in index.calls_by_callee.items():
        for _path, call in entries:
            edges.add((_last_name(call.caller), callee))
    symbol_locations = defaultdict(list)
    for path, info in index.files.items():
        for symbol in info.parsed.symbols:
            symbol_locations[symbol.name].append({
                "path": path, "qualname": symbol.qualname, "kind": symbol.kind,
                "start_line": symbol.start, "end_line": symbol.end,
            })
    facts = {
        "symbols": set(index.by_name),
        "symbol_locations": dict(symbol_locations),
        "files": set(index.all_paths),
        "call_edges": edges,
        "line_counts": {p: i.line_count for p, i in index.files.items()},
    }
    index._facts = facts
    return facts

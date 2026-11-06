import io
import os
import re
import shutil
import uuid
import zipfile
from pathlib import Path, PurePosixPath

import config
from services import db
from services.locks import project_lock

WORKSPACE = config.WORKSPACE_DIR
WORKSPACE.mkdir(parents=True, exist_ok=True)

PROJECT_ID_PATTERN = re.compile(r"^[0-9a-fA-F-]{32,36}$")

# Folders that are generated or huge. They are dropped when a ZIP is extracted,
# so a zipped project that still contains .venv or node_modules stays usable.
HEAVY_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "target", "dist",
    "build", "coverage", ".idea", ".mypy_cache", ".pytest_cache", ".gradle",
    ".next", ".cache", "__macosx",
}
SKIP_FILES = {".ds_store", "thumbs.db"}
HEAVY_DIRS |= set(config.EXTRA_IGNORED_DIRS)      # LC_EXTRA_IGNORED_DIRS
HEAVY_DIRS -= set(config.INCLUDE_DIRS)            # LC_INCLUDE_DIRS

TEXT_EXTENSIONS = {
    ".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".cs", ".cpp",
    ".c", ".h", ".hpp", ".php", ".rb", ".swift", ".kt", ".kts", ".scala", ".sql",
    ".html", ".css", ".scss", ".vue", ".svelte", ".md", ".txt", ".json", ".yml",
    ".yaml", ".toml", ".xml", ".gradle", ".properties", ".cfg", ".ini", ".sh",
    ".bat", ".ps1", ".env", ".csv", ".lock",
}
MANIFEST_NAMES = {
    "readme.md", "readme.txt", "pom.xml", "package.json", "requirements.txt",
    "pyproject.toml", "build.gradle", "cargo.toml", "go.mod", "composer.json",
}

# Extensionless names that are plain text and worth searching.
TEXT_NAMES = {
    "dockerfile", "makefile", "readme", "license", "changelog", "contributing", "jenkinsfile",
    "procfile", "gemfile", "rakefile", "vagrantfile", "pipfile", ".gitignore", ".dockerignore",
    ".editorconfig", ".env.example", ".eslintrc", ".prettierrc", ".babelrc", ".npmrc",
}
MAX_SEARCH_FILE_BYTES = 1_000_000
SEARCH_TIME_BUDGET_SECONDS = 15

MB = 1024 * 1024


# --------------------------------------------------------------------------
# Paths and safety helpers
# --------------------------------------------------------------------------

def project_path(project_id: str) -> Path:
    """Resolve a project directory, rejecting anything that is not a real project ID."""
    if not isinstance(project_id, str) or not PROJECT_ID_PATTERN.match(project_id):
        raise ValueError("Invalid project ID.")

    workspace = WORKSPACE.resolve()
    project_dir = (workspace / project_id).resolve()

    if not project_dir.is_relative_to(workspace):
        raise ValueError("Invalid project ID.")

    if not project_dir.is_dir():
        raise FileNotFoundError("Project not found.")

    return project_dir


def clean_relative_path(name: str):
    """Return a safe PurePosixPath for an uploaded/generated file name, or None to skip it."""
    if not isinstance(name, str):
        raise ValueError("Invalid file path.")

    name = name.replace("\\", "/").strip()

    if not name or name.endswith("/"):
        return None

    if (
        name.startswith("/")
        or re.match(r"^[A-Za-z]:", name)
        or "\x00" in name
        or len(name) > 300
    ):
        raise ValueError(f"Invalid file path: {name}")

    parts = [p for p in PurePosixPath(name).parts if p not in ("", ".")]

    if not parts or any(p == ".." for p in parts):
        raise ValueError(f"Invalid file path: {name}")

    path = PurePosixPath(*parts)

    if any(p.lower() in HEAVY_DIRS for p in path.parts[:-1]):
        return None

    if path.name.lower() in SKIP_FILES:
        return None

    return path


def _looks_binary(text: str) -> bool:
    """Decoded bytes that are mostly control characters or replacement marks are not text."""
    sample = text[:4000]
    if not sample:
        return False
    bad = sum(1 for ch in sample if (ord(ch) < 32 and ch not in "\t\n\r\f") or ch == "\ufffd")
    return bad / len(sample) > 0.10


def _read_bytes_as_text(data: bytes) -> str:
    if b"\x00" in data[:8000]:
        raise ValueError("File is not a readable text file.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Common for files saved on Windows.
        text = data.decode("cp1252", errors="replace")
    if _looks_binary(text):
        raise ValueError("File is not a readable text file.")
    return text


def is_text_candidate(filename: str, size: int = 0) -> bool:
    """One policy for 'worth reading as text', used by search, the index and the context map."""
    name = filename.lower()
    if name in SKIP_FILES:
        return False
    suffix = Path(name).suffix
    if suffix:
        return suffix in TEXT_EXTENSIONS
    return name in TEXT_NAMES or size <= 200_000


def read_text_file(path: Path) -> str:
    return _read_bytes_as_text(path.read_bytes())


# --------------------------------------------------------------------------
# Uploading
# --------------------------------------------------------------------------

def _copy_limited(source, target, limit_bytes, message):
    written = 0
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            break
        written += len(chunk)
        if written > limit_bytes:
            raise ValueError(message)
        target.write(chunk)
    return written


def _write_member(source, target_path, budget):
    """Write one file, enforcing the real number of bytes (not the ZIP header's claim)."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as out:
        return _copy_limited(
            source, out, budget,
            f"The project is larger than {config.MAX_UNZIPPED_MB} MB when unzipped.",
        )


def extract_project_with_stats(file):
    filename = file.filename or ""

    if not filename.lower().endswith(".zip"):
        raise ValueError("Only ZIP files are supported.")

    project_id = str(uuid.uuid4())
    project_dir = WORKSPACE / project_id
    zip_path = WORKSPACE / f"{project_id}.zip"

    project_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zip_path.open("wb") as buffer:
            _copy_limited(
                file.file, buffer, config.MAX_UPLOAD_MB * MB,
                f"The ZIP is larger than {config.MAX_UPLOAD_MB} MB.",
            )

        with zipfile.ZipFile(zip_path, "r") as archive:
            stats = _extract_archive(archive, project_dir)

        if stats["files"] == 0:
            raise ValueError("The ZIP contained no usable files.")

    except zipfile.BadZipFile:
        shutil.rmtree(project_dir, ignore_errors=True)
        raise ValueError("The uploaded file is not a valid ZIP archive.")

    except Exception:
        shutil.rmtree(project_dir, ignore_errors=True)
        raise

    finally:
        zip_path.unlink(missing_ok=True)

    stats["project_id"] = project_id
    stats["name"] = Path(filename).stem
    return stats


def extract_project(file):
    """Original helper: extract a ZIP and return the new project ID."""
    return extract_project_with_stats(file)["project_id"]


def _extract_archive(archive, project_dir):
    project_root = project_dir.resolve()
    budget = config.MAX_UNZIPPED_MB * MB
    single_limit = config.MAX_SINGLE_FILE_MB * MB
    files = 0
    skipped = 0

    for member in archive.infolist():
        if member.is_dir():
            continue

        relative = clean_relative_path(member.filename)

        if relative is None:
            skipped += 1
            continue

        target = (project_dir / relative).resolve()

        if not target.is_relative_to(project_root):
            raise ValueError("Invalid ZIP file.")

        if member.file_size > single_limit:
            skipped += 1
            continue

        if files >= config.MAX_PROJECT_FILES:
            raise ValueError(
                f"The project has more than {config.MAX_PROJECT_FILES} files."
            )

        with archive.open(member) as source:
            budget -= _write_member(source, target, budget)

        files += 1

    return {"files": files, "skipped": skipped}


def _project_totals(project_dir: Path):
    """(file count, total bytes) of what is currently in a project."""
    count = total = 0
    for path in _walk_files(project_dir):
        try:
            total += path.stat().st_size
        except OSError:
            continue
        count += 1
    return count, total


def _commit_staged(staged):
    """Move every staged file into place, or put everything back exactly as it was.

    An existing file is first moved aside (not deleted), so if any later step fails it can be
    restored, and files that were newly created are removed again.
    """
    journal = []   # (target, moved-aside original or None)

    try:
        for temporary, target in staged:
            if target.is_dir():
                raise ValueError(f"{target.name} is a folder in this project.")

            original = None
            if target.exists():
                original = target.with_name(f".{target.name}.atlas-old")
                os.replace(target, original)

            try:
                os.replace(temporary, target)
            except Exception:
                if original is not None:
                    os.replace(original, target)
                raise

            journal.append((target, original))

    except Exception:
        for target, original in reversed(journal):
            try:
                target.unlink(missing_ok=True)
                if original is not None:
                    os.replace(original, target)
            except OSError:
                pass
        raise

    for _, original in journal:
        if original is not None:
            original.unlink(missing_ok=True)


def add_uploaded_files(uploads, project_id=None):
    """Store loose files (not a ZIP) in a new project, or add them to an existing one.

    Limits apply to the whole request and to the project as a whole, not just to each file.
    The upload is all-or-nothing: files are staged first, and if any step of putting them in
    place fails, everything already moved is undone. Only one change to a project runs at a time.
    """
    if project_id:
        project_path(project_id)                      # fail fast on a bad ID
        with project_lock(project_id, what="Another upload or a test run"):
            return _add_uploaded_files(uploads, project_id)

    return _add_uploaded_files(uploads, None)


def _add_uploaded_files(uploads, project_id):
    if project_id:
        project_dir = project_path(project_id)
        created = False
        existing_files, existing_bytes = _project_totals(project_dir)
    else:
        project_id = str(uuid.uuid4())
        project_dir = WORKSPACE / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        created = True
        existing_files = existing_bytes = 0

    root = project_dir.resolve()
    per_file = config.MAX_SINGLE_FILE_MB * MB
    batch_limit = config.MAX_BATCH_UPLOAD_MB * MB
    project_room = config.MAX_UNZIPPED_MB * MB - existing_bytes
    allowed = min(batch_limit, project_room)
    cumulative_message = (
        f"These files add up to more than {config.MAX_BATCH_UPLOAD_MB} MB."
        if batch_limit <= project_room
        else f"Adding these files would take the project over {config.MAX_UNZIPPED_MB} MB."
    )

    staged = []   # (temporary path, final path)
    names = set()
    used = 0
    skipped = 0

    try:
        for filename, stream in uploads:
            relative = clean_relative_path(filename or "")

            if relative is None:
                skipped += 1
                continue

            key = relative.as_posix().lower()
            if key in names:
                raise ValueError(
                    f"Two of the files have the same path ({relative}). Nothing was uploaded."
                )
            names.add(key)

            target = (project_dir / relative).resolve()

            if not target.is_relative_to(root):
                raise ValueError("Invalid file path.")

            if existing_files + len(staged) >= config.MAX_PROJECT_FILES:
                raise ValueError(
                    f"A project can hold at most {config.MAX_PROJECT_FILES} files."
                )

            remaining = allowed - used
            if remaining <= 0:
                raise ValueError(cumulative_message)

            limit = min(per_file, remaining)
            message = (
                f"{relative.name} is larger than {config.MAX_SINGLE_FILE_MB} MB."
                if per_file <= remaining
                else cumulative_message
            )

            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.atlas-upload")
            staged.append((temporary, target))

            with temporary.open("wb") as out:
                used += _copy_limited(stream, out, limit, message)

        if not staged:
            raise ValueError("No usable files were uploaded.")

        _commit_staged(staged)

        # New or replaced files invalidate approval for running this project's tests.
        # A newly created project has no previous approval to revoke.
        if not created:
            db.set_tests_approved(project_id, False)

    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        if created:
            shutil.rmtree(project_dir, ignore_errors=True)
        raise

    return {"project_id": project_id, "files": len(staged), "skipped": skipped}


# --------------------------------------------------------------------------
# Listing, reading, searching
# --------------------------------------------------------------------------

def _walk_files(project_dir: Path):
    for root, directories, filenames in os.walk(project_dir):
        directories[:] = sorted(d for d in directories if d.lower() not in HEAVY_DIRS)
        for filename in sorted(filenames):
            yield Path(root) / filename


def list_project_files(project_id: str):
    project_dir = project_path(project_id)

    return [
        path.relative_to(project_dir).as_posix()
        for path in _walk_files(project_dir)
    ]


def read_project_file(project_id: str, path: str):
    project_dir = project_path(project_id)
    file_path = (project_dir / path).resolve()

    if not file_path.is_relative_to(project_dir):
        raise ValueError("Invalid file path.")

    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError("File not found.")

    return read_text_file(file_path)


def _searchable_files(project_dir: Path):
    """Files the search scans: the same files listing shows, minus oversized or non-text ones."""
    for path in _walk_files(project_dir):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_SEARCH_FILE_BYTES or not is_text_candidate(path.name, size):
            continue
        yield path.relative_to(project_dir).as_posix(), path


def _read_for_search(path: Path):
    try:
        return read_text_file(path)
    except (ValueError, OSError):
        return None


def search_project(project_id: str, query: str, limit: int = 20, offset: int = 0, scan_after: str | None = None):
    """Ranked, paginated search of a project.

    Matches exact text and identifiers, split identifiers (login_user / LoginUser / "login user"),
    file names, and, when literal hits are scarce, related code vocabulary. Every file in the
    result carries a score and the reasons for it. The reply states how many files matched in
    total and whether more pages exist, so a partial page is never mistaken for everything.
    """
    from services import search_engine

    project_dir = project_path(project_id)

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    terms = search_engine.parse_query(query)
    if not terms:
        raise ValueError("Search query cannot be empty.")

    try:
        limit = max(1, min(int(limit), 50))
        offset = max(0, int(offset))
    except (TypeError, ValueError):
        raise ValueError("limit and offset must be whole numbers.")

    wants_tests = bool(re.search(r"\b(?:tests?|spec|pytest|junit)\b", query, re.IGNORECASE))
    found = search_engine.run_search(
        sorted(_searchable_files(project_dir), key=lambda item: item[0]), terms, limit, offset,
        SEARCH_TIME_BUDGET_SECONDS, wants_tests, _read_for_search, scan_after,
    )

    page = found["page"]
    total = found["total_files"]
    has_more = found["scan_truncated"] or offset + len(page) < total

    return {
        "project_id": project_id,
        "query": query,
        "terms": [{"term": t.raw, "words": t.words} for t in terms],
        "synonyms_used": found["synonyms_used"],
        "total_files": total,
        "total_matches": total,           # kept for older callers
        "returned_files": len(page),
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
        "next_offset": offset + len(page) if has_more else None,
        "scanned_files": found["scanned_files"],
        "scan_truncated": found["scan_truncated"],
        "scan_after": scan_after,
        "scan_cursor": found.get("scan_cursor"),
        "next_scan_after": found.get("scan_cursor") if found["scan_truncated"] else None,
        "results": page,
    }


# --------------------------------------------------------------------------
# Context shown to the model when a project is attached
# --------------------------------------------------------------------------

def _size_label(size):
    return f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"


def build_project_context(project_id: str, char_budget: int) -> str:
    """What the model is told about an attached project.

    A small project is shown in full. A larger one is shown as a compact PROJECT MAP (purpose,
    languages, frameworks, entry points, routes, folders and key modules, built from the code
    index) plus a short capped file list, never a dump of thousands of names. Everything else
    is fetched on demand with the project tools.
    """
    project_dir = project_path(project_id)

    entries = []
    for path in _walk_files(project_dir):
        try:
            entries.append((path, path.stat().st_size))
        except OSError:
            continue
        if len(entries) >= 5000:
            break

    if not entries:
        return "The project is empty."

    listing_budget = int(char_budget * 0.30)
    lines, used = [], 0

    for path, size in entries:
        line = f"{path.relative_to(project_dir).as_posix()} ({_size_label(size)})"
        if used + len(line) > listing_budget:
            lines.append(
                f"... and {len(entries) - len(lines)} more files "
                "(use search_project or list_project_files)"
            )
            break
        lines.append(line)
        used += len(line) + 1

    text_files = []
    for path, size in entries:
        if path.suffix.lower() in TEXT_EXTENSIONS and size <= 60_000:
            try:
                text_files.append((path, read_text_file(path)))
            except (ValueError, OSError):
                continue

    remaining = char_budget - used
    total_text = sum(len(content) for _, content in text_files)
    parts = [f"PROJECT FILES ({len(entries)} files):", *lines, ""]

    if text_files and total_text <= remaining:
        parts.append("The complete contents of every text file are shown below.")
        for path, content in text_files:
            parts += [f"===== {path.relative_to(project_dir).as_posix()} =====", content, ""]
        return "\n".join(parts)

    project_map = ""
    try:
        from services import code_index
        project_map = code_index.project_map_text(project_id, max(600, int(remaining * 0.7)))
    except Exception:                                # a broken index must never hide the project
        project_map = ""

    if project_map:
        parts += [project_map, ""]
        remaining -= len(project_map)

    parts.append(
        "Only a map and a short file list are shown. Use search_project, find_symbol, "
        "read_symbol and read_project_file to inspect anything before making claims about it."
    )
    if not project_map:
        for path, content in text_files:
            if path.name.lower() in MANIFEST_NAMES and remaining > 600:
                snippet = content[:1500]
                parts += [f"--- {path.relative_to(project_dir).as_posix()} (start) ---", snippet, ""]
                remaining -= len(snippet)

    return "\n".join(parts)


# --------------------------------------------------------------------------
# Bundling generated code blocks for download
# --------------------------------------------------------------------------

def zip_files(files) -> bytes:
    """Zip generated files (a list of {path, content}) in memory, within size limits."""
    files = list(files)

    if len(files) > config.MAX_ZIP_DOWNLOAD_FILES:
        raise ValueError(f"Too many files for one ZIP (the limit is {config.MAX_ZIP_DOWNLOAD_FILES}).")

    buffer = io.BytesIO()
    seen = {}
    names = set()
    total = 0

    for index, item in enumerate(files, start=1):
        relative = clean_relative_path(item.get("path") or f"file-{index}.txt")
        if relative is None:
            relative = PurePosixPath(f"file-{index}.txt")
        if relative.as_posix().lower() in names:
            raise ValueError(f"Two files have the same path ({relative}). Nothing was zipped.")
        names.add(relative.as_posix().lower())
        content = item.get("content", "")
        total += len(content.encode("utf-8"))
        if total > config.MAX_ZIP_DOWNLOAD_MB * MB:
            raise ValueError(f"These files add up to more than {config.MAX_ZIP_DOWNLOAD_MB} MB.")
        seen[relative.as_posix()] = content

    if not seen:
        raise ValueError("There are no files to zip.")

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, content in seen.items():
            archive.writestr(path, content)

    return buffer.getvalue()

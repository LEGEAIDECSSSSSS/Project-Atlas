import collections
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import config
from services.locks import ProjectBusy, project_lock
from services.project_service import project_path

TIMEOUT_SECONDS = config.TEST_TIMEOUT_SECONDS
MAX_OUTPUT_CHARS = config.TEST_OUTPUT_CHARS

IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "target",
    "build",
    "dist",
}

# Variables that look like credentials are not passed to the test process.
SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")

SUMMARY = re.compile(
    r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)(?:,\s*Skipped:\s*(\d+))?"
)


def find_maven_project(project_dir: Path):
    """Find the Maven project root inside an uploaded project."""

    root_pom = project_dir / "pom.xml"

    if root_pom.is_file():
        return project_dir

    pom_files = []

    for pom in project_dir.rglob("pom.xml"):
        relative_parts = pom.relative_to(project_dir).parts

        if any(
            part.lower() in IGNORED_DIRECTORIES
            for part in relative_parts[:-1]
        ):
            continue

        if pom.is_file():
            pom_files.append(pom)

    if not pom_files:
        raise FileNotFoundError(
            "No pom.xml was found. This may not be a Maven project."
        )

    if len(pom_files) > 1:
        raise ValueError(
            "Multiple Maven projects were found. "
            "This version supports one Maven project at a time."
        )

    return pom_files[0].parent


def scrubbed_environment():
    return {
        key: value
        for key, value in os.environ.items()
        if not any(hint in key.upper() for hint in SECRET_HINTS)
    }


def kill_process_tree(process):
    """Stop the process AND everything it started (Maven forks a separate test JVM)."""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=15,
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def run_bounded(command, cwd, timeout, max_chars, env=None):
    """Run a command with a time limit and a hard cap on how much output is kept in memory.

    Returns (exit_code, output_tail, timed_out). Output beyond the cap is discarded as it
    arrives instead of being buffered, so a test that prints gigabytes cannot exhaust memory.
    """
    options = {
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": env,
        "shell": False,
    }

    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True

    process = subprocess.Popen(command, **options)

    chunks = collections.deque()
    size = 0

    def reader():
        nonlocal size
        for chunk in iter(lambda: process.stdout.read(4096), ""):
            chunks.append(chunk)
            size += len(chunk)
            while size > max_chars * 2 and len(chunks) > 1:
                size -= len(chunks.popleft())

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_process_tree(process)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    thread.join(timeout=5)
    output = "".join(chunks).strip()[-max_chars:]
    return process.returncode, output, timed_out


def summarize(output):
    """Pull the last Maven 'Tests run: N, Failures: F, Errors: E' line out of the output."""
    matches = SUMMARY.findall(output or "")
    if not matches:
        return None
    run, failures, errors, skipped = matches[-1]
    return {
        "tests_run": int(run),
        "failures": int(failures),
        "errors": int(errors),
        "skipped": int(skipped or 0),
    }


# --------------------------------------------------------------------------
# Test ecosystems
# --------------------------------------------------------------------------

ECOSYSTEMS = ("python", "node", "maven", "gradle", "go", "rust")
DEFAULT_NPM_TEST = re.compile(r"no test specified", re.IGNORECASE)
MAX_DETECT_DEPTH = 3


def _candidate_dirs(project_dir: Path):
    """The project root and sub-folders up to MAX_DETECT_DEPTH, shallowest first."""
    found = [project_dir]
    frontier = [project_dir]
    for _ in range(MAX_DETECT_DEPTH):
        nxt = []
        for folder in frontier:
            try:
                children = sorted(c for c in folder.iterdir() if c.is_dir())
            except OSError:
                continue
            for child in children:
                if child.name.lower() in IGNORED_DIRECTORIES or child.name.startswith("."):
                    continue
                found.append(child)
                nxt.append(child)
        frontier = nxt
    return found


def _has_python_tests(folder: Path):
    for marker in ("pytest.ini", "tox.ini", "conftest.py"):
        if (folder / marker).is_file():
            return True
    for name in ("setup.cfg", "pyproject.toml"):
        path = folder / name
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "pytest" in text:
                return True
    for sub in ("tests", "test"):
        tests = folder / sub
        if tests.is_dir():
            try:
                if any(tests.rglob("test_*.py")) or any(tests.rglob("*_test.py")):
                    return True
            except OSError:
                pass
    try:
        return any(folder.glob("test_*.py")) or any(folder.glob("*_test.py"))
    except OSError:
        return False


def _node_has_test_script(folder: Path):
    path = folder / "package.json"
    if not path.is_file():
        return False
    try:
        scripts = json.loads(path.read_text(encoding="utf-8", errors="replace")).get("scripts", {})
    except (OSError, ValueError, AttributeError):
        return False
    test = scripts.get("test") if isinstance(scripts, dict) else None
    return isinstance(test, str) and bool(test.strip()) and not DEFAULT_NPM_TEST.search(test)


def detect_ecosystems(project_dir: Path):
    """{ecosystem: [project roots]} for every kind of test setup found in the project."""
    found = {}
    for folder in _candidate_dirs(project_dir):
        checks = {
            "maven": (folder / "pom.xml").is_file(),
            "gradle": (folder / "build.gradle").is_file() or (folder / "build.gradle.kts").is_file(),
            "node": _node_has_test_script(folder),
            "go": (folder / "go.mod").is_file(),
            "rust": (folder / "Cargo.toml").is_file(),
            "python": _has_python_tests(folder),
        }
        for name, present in checks.items():
            if present:
                found.setdefault(name, []).append(folder)
    # a folder nested inside another root of the same kind is part of it (multi-module builds)
    for name, roots in list(found.items()):
        top = []
        for root in roots:
            if not any(parent in root.parents for parent in top):
                top.append(root)
        found[name] = top
    return found


def junit_xml_summary(folders):
    """Totals from JUnit-style XML reports (Maven surefire, Gradle), or None."""
    tests = failures = errors = skipped = files = 0
    for folder in folders:
        if not folder.is_dir():
            continue
        for report in folder.rglob("TEST-*.xml"):
            try:
                root = ET.parse(report).getroot()
            except (ET.ParseError, OSError):
                continue
            suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
            for suite in suites:
                try:
                    tests += int(suite.get("tests", 0))
                    failures += int(suite.get("failures", 0))
                    errors += int(suite.get("errors", 0))
                    skipped += int(suite.get("skipped", 0))
                    files += 1
                except ValueError:
                    continue
    if not files:
        return None
    return {"tests_run": tests, "failures": failures, "errors": errors, "skipped": skipped}


def _num(match, group=1):
    try:
        return int(match.group(group))
    except (AttributeError, TypeError, ValueError):
        return 0


def summarize_pytest(output):
    for line in reversed((output or "").splitlines()):
        if not re.search(r"\bin\s+[\d.]+\s*s\b", line):
            continue
        if not re.search(r"\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed)|no tests ran", line):
            continue
        passed = _num(re.search(r"(\d+)\s+passed", line))
        failed = _num(re.search(r"(\d+)\s+failed", line))
        errors = _num(re.search(r"(\d+)\s+errors?", line))
        skipped = _num(re.search(r"(\d+)\s+skipped", line))
        return {"tests_run": passed + failed + errors + skipped, "failures": failed,
                "errors": errors, "skipped": skipped}
    return None


def summarize_unittest(output):
    ran = re.findall(r"Ran\s+(\d+)\s+tests?\s+in", output or "")
    if not ran:
        return None
    failures = errors = skipped = 0
    for line in reversed((output or "").splitlines()):
        if line.startswith(("OK", "FAILED")):
            failures = _num(re.search(r"failures=(\d+)", line))
            errors = _num(re.search(r"errors=(\d+)", line))
            skipped = _num(re.search(r"skipped=(\d+)", line))
            break
    return {"tests_run": int(ran[-1]), "failures": failures, "errors": errors, "skipped": skipped}


def summarize_node(output):
    text = output or ""
    jest = re.search(r"Tests:\s+(?:(\d+)\s+failed,\s*)?(?:(\d+)\s+skipped,\s*)?"
                     r"(?:(\d+)\s+(?:passed|todo),\s*)*(\d+)\s+total", text)
    if jest:
        return {"tests_run": _num(jest, 4), "failures": _num(jest, 1), "errors": 0,
                "skipped": _num(jest, 2)}
    vitest = re.search(r"Tests\s+(?:(\d+)\s+failed\s*\|\s*)?(\d+)\s+passed(?:\s*\|\s*(\d+)\s+skipped)?", text)
    if vitest:
        failed, passed, skipped = _num(vitest, 1), _num(vitest, 2), _num(vitest, 3)
        return {"tests_run": failed + passed + skipped, "failures": failed, "errors": 0,
                "skipped": skipped}
    tap = re.findall(r"^#\s+(tests|pass|fail|skipped)\s+(\d+)", text, re.MULTILINE)
    if tap:
        counts = {k: int(v) for k, v in tap}
        return {"tests_run": counts.get("tests", 0), "failures": counts.get("fail", 0),
                "errors": 0, "skipped": counts.get("skipped", 0)}
    mocha_pass = re.search(r"(\d+)\s+passing", text)
    if mocha_pass:
        failing = _num(re.search(r"(\d+)\s+failing", text))
        pending = _num(re.search(r"(\d+)\s+pending", text))
        passing = _num(mocha_pass)
        return {"tests_run": passing + failing + pending, "failures": failing, "errors": 0,
                "skipped": pending}
    return None


def summarize_go(output):
    passed = len(re.findall(r"^--- PASS", output or "", re.MULTILINE))
    failed = len(re.findall(r"^--- FAIL", output or "", re.MULTILINE))
    skipped = len(re.findall(r"^--- SKIP", output or "", re.MULTILINE))
    if not (passed or failed or skipped):
        return {"tests_run": 0, "failures": 0, "errors": 0, "skipped": 0} \
            if re.search(r"no test files|\[no tests to run\]", output or "") else None
    return {"tests_run": passed + failed + skipped, "failures": failed, "errors": 0,
            "skipped": skipped}


def summarize_cargo(output):
    results = re.findall(r"test result: \w+\.\s+(\d+) passed;\s+(\d+) failed;\s+(\d+) ignored", output or "")
    if not results:
        return None
    passed = sum(int(r[0]) for r in results)
    failed = sum(int(r[1]) for r in results)
    ignored = sum(int(r[2]) for r in results)
    return {"tests_run": passed + failed + ignored, "failures": failed, "errors": 0,
            "skipped": ignored}


def _package_manager(folder: Path):
    if (folder / "pnpm-lock.yaml").is_file() and shutil.which("pnpm"):
        return ["pnpm", "test"]
    if (folder / "yarn.lock").is_file() and shutil.which("yarn"):
        return ["yarn", "test"]
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    return [npm, "test"] if npm else None


def _command_for(ecosystem: str, root: Path):
    """(command, summarizer, report_folders) or an error string. Fixed argument lists only."""
    if ecosystem == "python":
        if importlib.util.find_spec("pytest") is not None:
            return ([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                     "--maxfail=100"], summarize_pytest, [])
        return ([sys.executable, "-m", "unittest", "discover", "-v"], summarize_unittest, [])

    if ecosystem == "node":
        if not (root / "node_modules").is_dir():
            return ("This project's dependencies are not installed (there is no node_modules "
                    "folder, and uploads never include one). Atlas does not run `npm install`, "
                    "so the tests were NOT run.")
        command = _package_manager(root)
        if not command:
            return "npm (or pnpm/yarn) could not be found in the backend environment."
        return (command, summarize_node, [])

    if ecosystem == "gradle":
        wrapper = root / ("gradlew.bat" if os.name == "nt" else "gradlew")
        runner = str(wrapper) if wrapper.is_file() else (shutil.which("gradle.bat") or shutil.which("gradle"))
        if not runner:
            return "Gradle could not be found in the backend environment."
        return ([runner, "test", "--console=plain", "--no-daemon"], None,
                [root / "build" / "test-results"])

    if ecosystem == "go":
        go = shutil.which("go")
        if not go:
            return "Go could not be found in the backend environment."
        return ([go, "test", "./...", "-count=1", "-v"], summarize_go, [])

    if ecosystem == "rust":
        cargo = shutil.which("cargo")
        if not cargo:
            return "Cargo could not be found in the backend environment."
        return ([cargo, "test", "--no-fail-fast"], summarize_cargo, [])

    return f"Unsupported test ecosystem: {ecosystem}."


def run_project_tests(project_id: str, ecosystem: str = None):
    """Run the project's own tests with a fixed command. Only called after the user approves it.

    Supports Python (pytest or unittest), Node (npm/pnpm/yarn test), Maven, Gradle, Go and Rust.
    Only one test run (or upload) may touch a project at a time; a second request is refused
    straight away rather than queued behind a run that can take minutes.
    """
    project_path(project_id)    # validate the ID before anything else

    try:
        with project_lock(project_id, timeout=0, what="Another test run or upload"):
            return _run_project_tests(project_id, ecosystem)
    except ProjectBusy as error:
        return {"project_id": project_id, "status": "error", "error": str(error)}


def _maven_command(project_id, project_root):
    maven = shutil.which("mvn.cmd") or shutil.which("mvn")
    java = shutil.which("java")

    if not maven:
        return {"project_id": project_id, "status": "error",
                "error": "Maven could not be found in the backend environment."}
    if not java:
        return {"project_id": project_id, "status": "error",
                "error": "Java could not be found in the backend environment."}

    maven_home = Path(maven).resolve().parent.parent
    classworlds_files = list((maven_home / "boot").glob("plexus-classworlds-*.jar"))
    if not classworlds_files:
        return {"project_id": project_id, "status": "error",
                "error": ("Maven's Plexus Classworlds launcher JAR was not found "
                          f"under {maven_home / 'boot'}.")}

    maven_conf = maven_home / "bin" / "m2.conf"
    if not maven_conf.is_file():
        return {"project_id": project_id, "status": "error",
                "error": f"Maven configuration not found: {maven_conf}"}

    return [
        java, "-classpath", str(classworlds_files[0]),
        f"-Dclassworlds.conf={maven_conf}", f"-Dmaven.home={maven_home}",
        f"-Dmaven.multiModuleProjectDirectory={project_root}",
        "org.codehaus.plexus.classworlds.launcher.Launcher", "-B", "-ntp", "test",
    ]


def _run_project_tests(project_id: str, ecosystem: str = None):
    project_dir = project_path(project_id)
    detected = detect_ecosystems(project_dir)

    if ecosystem:
        ecosystem = ecosystem.lower().strip()
        if ecosystem not in ECOSYSTEMS:
            return {"project_id": project_id, "status": "error",
                    "error": f"Unknown ecosystem '{ecosystem}'. Choose one of: {', '.join(ECOSYSTEMS)}."}
    else:
        names = [name for name in ECOSYSTEMS if name in detected]
        if not names:
            return {"project_id": project_id, "status": "error",
                    "error": ("No test setup was found (no pom.xml, build.gradle, package.json test "
                              "script, go.mod, Cargo.toml, or Python tests). Atlas supports Python, "
                              "Node, Maven, Gradle, Go and Rust projects.")}
        if len(names) > 1:
            return {"project_id": project_id, "status": "error",
                    "error": ("This project has more than one kind of tests: "
                              + ", ".join(names) + ". Call run_project_tests again with "
                              "\"ecosystem\" set to the one you want.")}
        ecosystem = names[0]

    roots = detected.get(ecosystem, [])
    if not roots:
        return {"project_id": project_id, "status": "error",
                "error": f"No {ecosystem} test setup was found in this project."}
    if len(roots) > 1:
        shown = ", ".join(r.relative_to(project_dir).as_posix() or "." for r in roots[:4])
        return {"project_id": project_id, "status": "error",
                "error": (f"Several {ecosystem} projects were found ({shown}). "
                          "This version supports one per kind at a time.")}
    root = roots[0]

    if ecosystem == "maven":
        command = _maven_command(project_id, root)
        if isinstance(command, dict):
            return command
        summarizer, reports = None, [root / "target" / "surefire-reports"]
        label = "java Maven launcher -B -ntp test"
    else:
        built = _command_for(ecosystem, root)
        if isinstance(built, str):
            return {"project_id": project_id, "status": "error", "ecosystem": ecosystem,
                    "error": built}
        command, summarizer, reports = built
        label = " ".join(Path(command[0]).name if i == 0 else c for i, c in enumerate(command))

    base = {
        "project_id": project_id,
        "ecosystem": ecosystem,
        "build_system": {"maven": "Maven", "gradle": "Gradle", "node": "Node", "python": "Python",
                         "go": "Go", "rust": "Rust"}[ecosystem],
        "command": label,
        "working_directory": root.relative_to(project_dir).as_posix() or ".",
    }

    env = scrubbed_environment()
    if ecosystem == "node":
        env["CI"] = "true"                         # no interactive watch mode

    try:
        exit_code, output, timed_out = run_bounded(
            command, cwd=root, timeout=TIMEOUT_SECONDS, max_chars=MAX_OUTPUT_CHARS, env=env,
        )
    except OSError as error:
        return {**base, "status": "error", "error": str(error)}

    counts = None
    if ecosystem == "maven":
        counts = summarize(output) or junit_xml_summary(reports)
    elif summarizer:
        counts = summarizer(output)
    if counts is None and reports:
        counts = junit_xml_summary(reports)
    extra = {"summary": counts} if counts else {}

    if timed_out:
        return {**base, "status": "timeout",
                "error": f"Tests exceeded the {TIMEOUT_SECONDS}-second time limit and were stopped.",
                "output": output, **extra}

    # pytest exits 5 when it collected no tests: that is "zero tests ran", not a failure
    zero_tests = ecosystem == "python" and exit_code == 5 and (
        counts is None or counts.get("tests_run") == 0)
    if zero_tests:
        counts = {"tests_run": 0, "failures": 0, "errors": 0, "skipped": 0}
        extra = {"summary": counts}

    if exit_code == 0 or zero_tests:
        # a clean exit with no test counts proves nothing was actually checked
        status = "passed" if counts is not None else "unverified"
    else:
        status = "failed"

    result = {**base, "status": status, "exit_code": exit_code, "output": output, **extra}
    if status == "unverified":
        result["note"] = ("The command exited successfully but no test counts could be read from "
                          "its output, so it is not known that any test ran.")
    return result

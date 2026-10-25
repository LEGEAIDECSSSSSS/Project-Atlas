import collections
import os
import re
import shutil
import signal
import subprocess
import threading
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


def run_project_tests(project_id: str):
    """Run Maven tests using its Java launcher. Only called after the user approves it.

    Only one test run (or upload) may touch a project at a time; a second request is refused
    straight away rather than queued behind a run that can take minutes.
    """
    project_path(project_id)    # validate the ID before anything else

    try:
        with project_lock(project_id, timeout=0, what="Another test run or upload"):
            return _run_project_tests(project_id)
    except ProjectBusy as error:
        return {"project_id": project_id, "status": "error", "error": str(error)}


def _run_project_tests(project_id: str):
    project_dir = project_path(project_id)
    project_root = find_maven_project(project_dir)

    maven = shutil.which("mvn.cmd") or shutil.which("mvn")
    java = shutil.which("java")

    if not maven:
        return {
            "project_id": project_id,
            "status": "error",
            "error": "Maven could not be found in the backend environment.",
        }

    if not java:
        return {
            "project_id": project_id,
            "status": "error",
            "error": "Java could not be found in the backend environment.",
        }

    maven_home = Path(maven).resolve().parent.parent
    classworlds_files = list(
        (maven_home / "boot").glob("plexus-classworlds-*.jar")
    )

    if not classworlds_files:
        return {
            "project_id": project_id,
            "status": "error",
            "error": (
                "Maven's Plexus Classworlds launcher JAR was not found "
                f"under {maven_home / 'boot'}."
            ),
        }

    classworlds_jar = classworlds_files[0]
    maven_conf = maven_home / "bin" / "m2.conf"

    if not maven_conf.is_file():
        return {
            "project_id": project_id,
            "status": "error",
            "error": f"Maven configuration not found: {maven_conf}",
        }

    command = [
        java,
        "-classpath",
        str(classworlds_jar),
        f"-Dclassworlds.conf={maven_conf}",
        f"-Dmaven.home={maven_home}",
        f"-Dmaven.multiModuleProjectDirectory={project_root}",
        "org.codehaus.plexus.classworlds.launcher.Launcher",
        "-B",
        "-ntp",
        "test",
    ]

    base = {
        "project_id": project_id,
        "build_system": "Maven",
        "command": "java Maven launcher -B -ntp test",
    }

    try:
        exit_code, output, timed_out = run_bounded(
            command,
            cwd=project_root,
            timeout=TIMEOUT_SECONDS,
            max_chars=MAX_OUTPUT_CHARS,
            env=scrubbed_environment(),
        )
    except OSError as error:
        return {**base, "status": "error", "error": str(error)}

    counts = summarize(output)

    if timed_out:
        return {
            **base,
            "status": "timeout",
            "error": f"Tests exceeded the {TIMEOUT_SECONDS}-second time limit and were stopped.",
            "output": output,
            **({"summary": counts} if counts else {}),
        }

    return {
        **base,
        "status": "passed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "output": output,
        **({"summary": counts} if counts else {}),
    }

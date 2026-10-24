
import shutil
import subprocess
from pathlib import Path

from services.project_service import WORKSPACE


TIMEOUT_SECONDS = 120
MAX_OUTPUT_CHARS = 12000

IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "target",
    "build",
    "dist",
}


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


def run_project_tests(project_id: str):
    """Run Maven tests using its Java launcher."""

    workspace = WORKSPACE.resolve()
    project_dir = (workspace / project_id).resolve()

    if not project_dir.is_relative_to(workspace):
        raise ValueError("Invalid project ID.")

    if not project_dir.is_dir():
        raise FileNotFoundError("Project not found.")

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

    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
            shell=False,
        )

        output = (
            (result.stdout or "")
            + "\n"
            + (result.stderr or "")
        ).strip()

        return {
            "project_id": project_id,
            "build_system": "Maven",
            "command": "java Maven launcher -B -ntp test",
            "status": (
                "passed" if result.returncode == 0 else "failed"
            ),
            "exit_code": result.returncode,
            "output": output[-MAX_OUTPUT_CHARS:],
        }

    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""

        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")

        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        output = f"{stdout}\n{stderr}".strip()

        return {
            "project_id": project_id,
            "build_system": "Maven",
            "command": "java Maven launcher -B -ntp test",
            "status": "timeout",
            "error": (
                f"Tests exceeded the {TIMEOUT_SECONDS}-second time limit."
            ),
            "output": output[-MAX_OUTPUT_CHARS:],
        }

    except OSError as error:
        return {
            "project_id": project_id,
            "build_system": "Maven",
            "status": "error",
            "error": str(error),
        }

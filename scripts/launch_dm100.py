"""Prepare a local, isolated Python environment before starting DM100."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

PROBE = """
import importlib.metadata
import importlib.util
import json
import sys

for name in ('numpy', 'scipy', 'yaml', 'matplotlib', 'fastapi', 'uvicorn', 'imageio_ffmpeg'):
    __import__(name)
spec = importlib.util.find_spec('powered_landing_guidance')
print(json.dumps({
    'version': list(sys.version_info[:2]),
    'prefix': sys.prefix,
    'source': spec.origin if spec else None,
    'project_version': importlib.metadata.version('powered-landing-guidance'),
    'dependencies': True,
}))
"""


class LauncherError(RuntimeError):
    """An actionable setup failure; the application must not start."""


def _process_env() -> dict[str, str]:
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def _python_path(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _inside_project(project_root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(project_root):
        raise LauncherError(f"Refusing to change a path outside this project: {path}")
    return resolved


def inspect_environment(venv_python: Path, project_root: Path) -> bool:
    """Check the interpreter, isolation, dependencies and editable source location."""
    project_root = project_root.resolve()
    if not venv_python.is_file():
        return False
    try:
        config = dict(
            line.split("=", 1)
            for line in (project_root / ".venv/pyvenv.cfg").read_text(encoding="utf-8").splitlines()
            if "=" in line
        )
        config = {key.strip(): value.strip() for key, value in config.items()}
        if config.get("include-system-site-packages", "").lower() != "false":
            return False
        with (project_root / "pyproject.toml").open("rb") as manifest:
            project_version = tomllib.load(manifest)["project"]["version"]
        result = subprocess.run(
            [str(venv_python), "-I", "-X", "utf8", "-c", PROBE],
            cwd=project_root,
            env=_process_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
        if result.returncode:
            return False
        info = json.loads(result.stdout)
        if not isinstance(info, dict):
            return False
        return (
            info.get("version") == [3, 12]
            and info.get("dependencies") is True
            and info.get("project_version") == project_version
            and Path(info["prefix"]).resolve() == (project_root / ".venv").resolve()
            and Path(info["source"]).resolve()
            == (project_root / "src/powered_landing_guidance/__init__.py").resolve()
        )
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError):
        return False


def _run_setup(command: list[str], project_root: Path, description: str) -> None:
    print(f"DM100: {description}", flush=True)
    try:
        result = subprocess.run(command, cwd=project_root, env=_process_env(), check=False)
    except OSError as exc:
        raise LauncherError(f"{description} failed: {exc}") from exc
    if result.returncode:
        raise LauncherError(
            f"{description} failed (exit {result.returncode}). "
            "Check the messages above and your internet connection, then run the launcher again."
        )


def ensure_environment(project_root: Path, base_python: Path) -> Path:
    """Reuse a healthy environment, or preserve it and create a fresh one."""
    project_root = project_root.resolve()
    venv_dir = _inside_project(project_root, project_root / ".venv")
    if venv_dir != project_root / ".venv":
        raise LauncherError("The .venv directory must not be a link to another directory.")
    venv_python = _python_path(venv_dir)
    if inspect_environment(venv_python, project_root):
        return venv_python
    if venv_dir.exists():
        if not venv_dir.is_dir():
            raise LauncherError(f"The environment path is not a directory: {venv_dir}")
        backup_parent = _inside_project(project_root, project_root / ".local/environment-backups")
        backup_parent.mkdir(parents=True, exist_ok=True)
        backup = _inside_project(
            project_root,
            backup_parent / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}",
        )
        try:
            venv_dir.rename(backup)
        except OSError as exc:
            raise LauncherError(
                "Could not preserve the old environment. Close other DM100/Python windows "
                f"using this project and try again: {exc}"
            ) from exc
        print(f"DM100: Previous environment preserved at {backup}", flush=True)
    _run_setup(
        [str(base_python), "-I", "-X", "utf8", "-m", "venv", str(venv_dir)],
        project_root,
        "Creating an isolated Python 3.12 environment",
    )
    _run_setup(
        [
            str(venv_python),
            "-I",
            "-X",
            "utf8",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            *(
                ["--constraint", str(project_root / "requirements" / "constraints.txt")]
                if (project_root / "requirements" / "constraints.txt").is_file()
                else []
            ),
            "-e",
            f"{project_root}[ui,video]",
        ],
        project_root,
        "Installing DM100 and its dependencies (internet required)",
    )
    if not inspect_environment(venv_python, project_root):
        raise LauncherError(
            "The new environment did not pass its startup checks. "
            "Review the installation output and run the launcher again."
        )
    return venv_python


def main(argv: list[str] | None = None, project_root: Path | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = (Path(__file__).resolve().parents[1] if project_root is None else project_root).resolve()
    if sys.version_info[:2] != (3, 12):
        print("DM100: Python 3.12.x is required. Run launch-dm100.bat.", file=sys.stderr)
        return 1
    base_python = Path(getattr(sys, "_base_executable", None) or sys.executable).resolve()
    try:
        # On Windows the running venv executable cannot be moved. Restart outside it first.
        if sys.prefix != sys.base_prefix:
            result = subprocess.run(
                [str(base_python), "-I", "-X", "utf8", str(Path(__file__).resolve()), *arguments],
                cwd=root,
                env=_process_env(),
                check=False,
            )
            return result.returncode
        setup_only = "--setup-only" in arguments
        arguments = [argument for argument in arguments if argument != "--setup-only"]
        python = ensure_environment(root, base_python)
        if setup_only:
            print("DM100: Environment ready. Run launch-dm100.bat to open the app.", flush=True)
            return 0
        result = subprocess.run(
            [
                str(python),
                "-I",
                "-X",
                "utf8",
                "-m",
                "powered_landing_guidance.local_app",
                *arguments,
            ],
            cwd=root,
            env=_process_env(),
            check=False,
        )
        return result.returncode
    except (LauncherError, OSError) as exc:
        print(f"DM100: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

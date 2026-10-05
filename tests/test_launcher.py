"""Launcher regression checks that never install packages or start the web application."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

LAUNCHER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "launch_dm100.py"
SPEC = importlib.util.spec_from_file_location("dm100_launcher_under_test", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="dm100 launcher 한글 test ")
        self.addCleanup(self.temporary_directory.cleanup)
        self.project_root = Path(self.temporary_directory.name).resolve()
        (self.project_root / "pyproject.toml").write_text(
            '[project]\nname = "powered-landing-guidance"\nversion = "1.0.0"\n',
            encoding="utf-8",
        )
        self.source = self.project_root / "src" / "powered_landing_guidance" / "__init__.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text('__version__ = "1.0.0"\n', encoding="utf-8")
        interpreter = "Scripts/python.exe" if os.name == "nt" else "bin/python"
        self.venv_python = self.project_root / ".venv" / interpreter
        self.make_environment()

    def make_environment(self):
        self.venv_python.parent.mkdir(parents=True, exist_ok=True)
        self.venv_python.touch()
        (self.project_root / ".venv" / "pyvenv.cfg").write_text(
            f"home = {Path(sys.executable).parent}\n"
            "include-system-site-packages = false\n"
            f"executable = {sys.executable}\n"
            "version = 3.12.7\n"
            f"command = {sys.executable} -m venv {self.project_root / '.venv'}\n",
            encoding="utf-8",
        )

    def probe_result(self, **changes):
        payload = {
            "version": [3, 12],
            "prefix": str(self.project_root / ".venv"),
            "source": str(self.source),
            "project_version": "1.0.0",
            "dependencies": True,
        }
        payload.update(changes)
        return subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")

    def test_matching_current_source_environment_is_accepted(self):
        with mock.patch.object(launcher.subprocess, "run", return_value=self.probe_result()):
            self.assertTrue(launcher.inspect_environment(self.venv_python, self.project_root))

    def test_missing_old_base_python_is_rejected(self):
        (self.project_root / ".venv" / "pyvenv.cfg").write_text(
            "home = C:\\Users\\old-owner\\Desktop\\DM100\\python\n"
            "include-system-site-packages = false\n"
            "executable = C:\\Users\\old-owner\\Desktop\\DM100\\python\\python.exe\n",
            encoding="utf-8",
        )
        broken = subprocess.CompletedProcess(
            [], 103, stdout="", stderr="No Python at the previous owner's Desktop path"
        )
        with mock.patch.object(launcher.subprocess, "run", return_value=broken):
            self.assertFalse(launcher.inspect_environment(self.venv_python, self.project_root))

    def test_editable_install_from_another_directory_is_rejected(self):
        old_source = self.project_root.parent / "previous checkout" / "src" / self.source.name
        with mock.patch.object(
            launcher.subprocess, "run", return_value=self.probe_result(source=str(old_source))
        ):
            self.assertFalse(launcher.inspect_environment(self.venv_python, self.project_root))

    def test_unavailable_dependencies_and_wrong_versions_are_rejected(self):
        for changes in (
            {"dependencies": False},
            {"version": [3, 11]},
            {"project_version": "0.9.0"},
            {"prefix": str(self.project_root.parent)},
        ):
            with (
                self.subTest(changes=changes),
                mock.patch.object(
                    launcher.subprocess, "run", return_value=self.probe_result(**changes)
                ),
            ):
                self.assertFalse(launcher.inspect_environment(self.venv_python, self.project_root))

    def test_global_site_packages_are_rejected(self):
        configuration = self.project_root / ".venv" / "pyvenv.cfg"
        configuration.write_text(
            configuration.read_text(encoding="utf-8").replace(
                "include-system-site-packages = false", "include-system-site-packages = true"
            ),
            encoding="utf-8",
        )
        with mock.patch.object(launcher.subprocess, "run", return_value=self.probe_result()):
            self.assertFalse(launcher.inspect_environment(self.venv_python, self.project_root))

    def test_failed_or_malformed_probe_is_rejected(self):
        for result in (
            subprocess.CompletedProcess([], 1, stdout="", stderr="ImportError"),
            subprocess.CompletedProcess([], 0, stdout="not JSON", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="[]", stderr=""),
        ):
            with (
                self.subTest(result=result),
                mock.patch.object(launcher.subprocess, "run", return_value=result),
            ):
                self.assertFalse(launcher.inspect_environment(self.venv_python, self.project_root))

    def test_unresponsive_or_missing_interpreter_is_rejected(self):
        for error in (
            FileNotFoundError("python.exe no longer exists"),
            subprocess.TimeoutExpired([str(self.venv_python)], 20),
        ):
            with (
                self.subTest(error=type(error).__name__),
                mock.patch.object(launcher.subprocess, "run", side_effect=error),
            ):
                self.assertFalse(launcher.inspect_environment(self.venv_python, self.project_root))

    def test_valid_environment_skips_environment_creation_and_installation(self):
        with (
            mock.patch.object(launcher, "inspect_environment", return_value=True),
            mock.patch.object(launcher.subprocess, "run") as run,
        ):
            result = launcher.ensure_environment(self.project_root, Path(sys.executable))
        self.assertEqual(result.resolve(), self.venv_python.resolve())
        run.assert_not_called()

    def test_invalid_environment_is_preserved_and_rebuilt_at_current_location(self):
        old_marker = self.project_root / ".venv" / "previous-owner-files.txt"
        old_marker.write_text("Keep this old environment for recovery.", encoding="utf-8")
        commands = []

        def fake_run(command, **kwargs):
            commands.append(list(command))
            if "venv" in command:
                self.assertFalse(old_marker.exists())
                self.make_environment()
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with (
            mock.patch.object(launcher, "inspect_environment", side_effect=[False, True]),
            mock.patch.object(launcher.subprocess, "run", side_effect=fake_run),
        ):
            result = launcher.ensure_environment(self.project_root, Path(sys.executable))

        self.assertEqual(result.resolve(), self.venv_python.resolve())
        backups = list(
            (self.project_root / ".local" / "environment-backups").glob(
                "*/previous-owner-files.txt"
            )
        )
        self.assertEqual(len(backups), 1)
        self.assertEqual(
            backups[0].read_text(encoding="utf-8"), "Keep this old environment for recovery."
        )
        self.assertTrue(backups[0].resolve().is_relative_to(self.project_root))
        self.assertFalse(old_marker.exists())
        creation_commands = [command for command in commands if "venv" in command]
        install_commands = [command for command in commands if "pip" in command]
        self.assertEqual(len(creation_commands), 1)
        self.assertEqual(Path(creation_commands[0][-1]).resolve(), self.project_root / ".venv")
        self.assertEqual(len(install_commands), 1)
        self.assertIn(str(self.project_root) + "[ui,video]", install_commands[0])
        self.assertIn("-e", install_commands[0])

    def test_creation_or_installation_failure_does_not_return_runnable_environment(self):
        for failed_step in ("venv", "pip"):
            with self.subTest(failed_step=failed_step):

                def fake_run(command, failed_step=failed_step, **kwargs):
                    if "venv" in command and failed_step != "venv":
                        self.make_environment()
                    code = 2 if failed_step in command else 0
                    return subprocess.CompletedProcess(command, code, stdout="", stderr="failed")

                with (
                    mock.patch.object(launcher, "inspect_environment", return_value=False),
                    mock.patch.object(launcher.subprocess, "run", side_effect=fake_run),
                    self.assertRaises(launcher.LauncherError),
                ):
                    launcher.ensure_environment(self.project_root, Path(sys.executable))
                self.make_environment()

    def test_backup_path_resolving_outside_project_is_rejected_before_any_mutation(self):
        old_marker = self.project_root / ".venv" / "keep.txt"
        old_marker.write_text("untouched", encoding="utf-8")
        backup_parent = self.project_root / ".local" / "environment-backups"
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            if path == backup_parent:
                return self.project_root.parent / "outside-backups"
            return original_resolve(path, *args, **kwargs)

        with (
            mock.patch.object(Path, "resolve", resolve),
            mock.patch.object(launcher, "inspect_environment", return_value=False),
            mock.patch.object(launcher.subprocess, "run") as run,
            self.assertRaises(launcher.LauncherError),
        ):
            launcher.ensure_environment(self.project_root, Path(sys.executable))
        self.assertEqual(old_marker.read_text(encoding="utf-8"), "untouched")
        run.assert_not_called()

    def test_linked_environment_is_rejected_before_inspection_or_repair(self):
        environment = self.project_root / ".venv"
        old_marker = environment / "keep.txt"
        old_marker.write_text("untouched", encoding="utf-8")
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            if path == environment:
                return self.project_root / "some-other-environment"
            return original_resolve(path, *args, **kwargs)

        with (
            mock.patch.object(Path, "resolve", resolve),
            mock.patch.object(launcher, "inspect_environment") as inspect_environment,
            mock.patch.object(launcher.subprocess, "run") as run,
            self.assertRaises(launcher.LauncherError),
        ):
            launcher.ensure_environment(self.project_root, Path(sys.executable))
        self.assertEqual(old_marker.read_text(encoding="utf-8"), "untouched")
        inspect_environment.assert_not_called()
        run.assert_not_called()

    def test_failed_post_installation_probe_is_not_treated_as_success(self):
        def fake_run(command, **kwargs):
            if "venv" in command:
                self.make_environment()
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with (
            mock.patch.object(launcher, "inspect_environment", side_effect=[False, False]),
            mock.patch.object(launcher.subprocess, "run", side_effect=fake_run),
            self.assertRaises(launcher.LauncherError),
        ):
            launcher.ensure_environment(self.project_root, Path(sys.executable))

    def test_main_does_not_start_application_after_setup_failure(self):
        with (
            mock.patch.object(launcher.sys, "prefix", launcher.sys.base_prefix),
            mock.patch.object(
                launcher,
                "ensure_environment",
                side_effect=launcher.LauncherError("installation failed"),
            ),
            mock.patch.object(launcher.subprocess, "run") as run,
        ):
            self.assertEqual(launcher.main([], project_root=self.project_root), 1)
        run.assert_not_called()

    def test_application_arguments_and_exit_code_are_preserved(self):
        arguments = [
            "--port",
            "9888",
            "--no-browser",
            "--output-dir",
            str(self.project_root / "results 한글 with spaces"),
        ]
        with (
            mock.patch.object(launcher.sys, "prefix", launcher.sys.base_prefix),
            mock.patch.object(launcher, "ensure_environment", return_value=self.venv_python),
            mock.patch.object(
                launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)
            ) as run,
        ):
            result = launcher.main(arguments, project_root=self.project_root)
        self.assertEqual(result, 7)
        command = run.call_args.args[0]
        self.assertEqual(Path(command[0]).resolve(), self.venv_python.resolve())
        self.assertIn("powered_landing_guidance.local_app", command)
        self.assertEqual(command[-len(arguments) :], arguments)
        self.assertEqual(Path(run.call_args.kwargs["cwd"]).resolve(), self.project_root)

    def test_setup_only_prepares_environment_without_starting_application(self):
        with (
            mock.patch.object(launcher.sys, "prefix", launcher.sys.base_prefix),
            mock.patch.object(launcher, "ensure_environment", return_value=self.venv_python),
            mock.patch.object(launcher.subprocess, "run") as run,
        ):
            self.assertEqual(launcher.main(["--setup-only"], project_root=self.project_root), 0)
        run.assert_not_called()

    def test_bootstrap_started_in_venv_restarts_with_base_interpreter_before_repair(self):
        arguments = ["--setup-only", "--port", "9888"]
        base_interpreter = Path(sys.executable).resolve()
        with (
            mock.patch.object(launcher.sys, "prefix", str(self.project_root / ".venv")),
            mock.patch.object(launcher.sys, "_base_executable", str(base_interpreter)),
            mock.patch.object(launcher, "ensure_environment") as ensure_environment,
            mock.patch.object(
                launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 23)
            ) as run,
        ):
            result = launcher.main(arguments, project_root=self.project_root)
        self.assertEqual(result, 23)
        ensure_environment.assert_not_called()
        command = run.call_args.args[0]
        self.assertEqual(Path(command[0]).resolve(), base_interpreter)
        self.assertEqual(command[1], "-I")
        self.assertEqual(command[2:4], ["-X", "utf8"])
        self.assertEqual(Path(command[4]).resolve(), LAUNCHER_PATH)
        self.assertEqual(command[5:], arguments)


if __name__ == "__main__":
    unittest.main()

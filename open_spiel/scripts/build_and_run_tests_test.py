# Copyright 2026 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise the build script without installing or compiling OpenSpiel."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class BuildAndRunTestsTest(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    scripts = self.root / "open_spiel" / "scripts"
    scripts.mkdir(parents=True)
    for name in ("build_and_run_tests.sh", "argslib.sh", "global_variables.sh"):
      shutil.copyfile(Path(__file__).with_name(name), scripts / name)
    self.script = scripts / "build_and_run_tests.sh"
    self.bin = self.root / "bin"
    self.bin.mkdir()
    self.log = self.root / "commands.log"
    self.real_ctest = shutil.which("ctest")
    self.env = {
        **os.environ,
        "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
        "TEST_COMMAND_LOG": str(self.log),
        "CXX": str(self.bin / "clang++"),
        "VIRTUAL_ENV": "",
        "OPEN_SPIEL_BUILD_WITH_JULIA": "OFF",
        "REAL_CTEST": self.real_ctest or "",
    }
    self._tool("clang++", "exit 0")
    self._tool(
        "which",
        r"""if [[ "$1" == python && "$TEST_NO_PYTHON" == 1 ]]; then
  exit 1
fi
command -v "$1"
""",
    )
    self._tool(
        "python",
        r"""if [[ "$1" == -c ]]; then
  echo 3.12.0
else
  echo "pip $*" >> "$TEST_COMMAND_LOG"
  exit "${TEST_PIP_EXIT:-0}"
fi
""",
    )
    shutil.copyfile(self.bin / "python", self.bin / "python3")
    (self.bin / "python3").chmod(0o755)
    self._tool(
        "cmake",
        r"""echo cmake >> "$TEST_COMMAND_LOG"
exit "${TEST_CMAKE_EXIT:-0}"
""",
    )
    self._tool(
        "make",
        r"""echo "make $*" >> "$TEST_COMMAND_LOG"
exit "${TEST_MAKE_EXIT:-0}"
""",
    )
    self._tool(
        "ctest",
        r"""echo ctest >> "$TEST_COMMAND_LOG"
if [[ "$TEST_NO_TESTS" == 1 ]]; then
  exec "$REAL_CTEST" "$@"
fi
exit "${TEST_CTEST_EXIT:-0}"
""",
    )
    self._tool(
        "virtualenv",
        r"""echo virtualenv >> "$TEST_COMMAND_LOG"
if [[ "${TEST_VIRTUALENV_EXIT:-0}" != 0 ]]; then
  exit "$TEST_VIRTUALENV_EXIT"
fi
mkdir -p venv/bin
cat > venv/bin/activate <<'ACTIVATE'
deactivate() { echo cleanup >> "$TEST_COMMAND_LOG"; }
ACTIVATE
""",
    )

  def _tool(self, name, body):
    path = self.bin / name
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)

  def _run(self, *args, **environment):
    self.log.write_text("")
    result = subprocess.run(
        [
            "/bin/bash",
            str(self.script),
            "--virtualenv=false",
            "--install=false",
            "--num_threads=1",
            *args,
        ],
        cwd=self.root,
        env={**self.env, **environment},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    return result, self.log.read_text().splitlines()

  def test_configuration_failure_stops_before_build_and_tests(self):
    for build_only in ("false", "true"):
      with self.subTest(build_only=build_only):
        result, commands = self._run(
            "--build_only=" + build_only, TEST_CMAKE_EXIT="17"
        )
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
        self.assertEqual(commands, ["cmake"])
        self.assertNotIn("All tests passed", result.stdout)

  def test_build_failure_stops_before_tests_for_each_target(self):
    for target in (
        "all",
        "python_smoke",
        "julia_test",
        "gospiel_test",
        "chess",
    ):
      with self.subTest(target=target):
        result, commands = self._run(
            "--test_only=" + target, TEST_MAKE_EXIT="18"
        )
        self.assertEqual(result.returncode, 18, result.stdout + result.stderr)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0], "cmake")
        self.assertTrue(commands[1].startswith("make "))
        self.assertNotIn("All tests passed", result.stdout)

  def test_failed_build_only_is_not_successful(self):
    result, commands = self._run("--build_only=true", TEST_MAKE_EXIT="18")
    self.assertEqual(result.returncode, 18)
    self.assertNotIn("ctest", commands)
    self.assertNotIn("Skipping runing tests", result.stdout)

  def test_pip_build_failure_stops_before_tests(self):
    result, commands = self._run("--build_with_pip=true", TEST_PIP_EXIT="19")
    self.assertEqual(result.returncode, 19)
    self.assertEqual(commands, ["pip -m pip install ."])
    self.assertNotIn("All tests passed", result.stdout)

  def test_dependency_installation_failure_stops_before_build(self):
    result, commands = self._run("--install=true", TEST_PIP_EXIT="19")
    self.assertEqual(result.returncode, 19)
    self.assertEqual(len(commands), 1)
    self.assertTrue(commands[0].startswith("pip "))

  def test_virtualenv_creation_failure_stops(self):
    result, commands = self._run("--virtualenv=true", TEST_VIRTUALENV_EXIT="20")
    self.assertEqual(result.returncode, 20)
    self.assertEqual(commands, ["virtualenv"])

  def test_cleanup_preserves_configuration_failure_status(self):
    result, commands = self._run("--virtualenv=true", TEST_CMAKE_EXIT="17")
    self.assertEqual(result.returncode, 17)
    self.assertEqual(commands[-2:], ["cmake", "cleanup"])
    self.assertNotIn("ctest", commands)

  def test_invalid_build_directory_stops(self):
    (self.root / "not_a_directory").write_text("file")
    result, commands = self._run("--build_dir=not_a_directory")
    self.assertNotEqual(result.returncode, 0)
    self.assertEqual(commands, [])

  def test_empty_test_selection_is_an_error(self):
    if not self.real_ctest:
      self.skipTest("CTest is needed to verify its no-tests behavior")
    for args in ((), ("--test_only=missing",), ("--build_with_pip=true",)):
      with self.subTest(args=args):
        result, commands = self._run(*args, TEST_NO_TESTS="1")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ctest", commands)
        self.assertNotIn("All tests passed", result.stdout)

  def test_test_failure_remains_an_error(self):
    result, commands = self._run(TEST_CTEST_EXIT="21")
    self.assertNotEqual(result.returncode, 0)
    self.assertEqual(commands[-1], "ctest")
    self.assertIn("At least one test failed", result.stdout)
    self.assertNotIn("All tests passed", result.stdout)

  def test_success_runs_build_and_tests(self):
    for args in (
        (),
        ("--test_only=python_smoke",),
        ("--build_with_pip=true",),
        ("--github_ci=true",),
    ):
      with self.subTest(args=args):
        result, commands = self._run(*args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(commands[-1], "ctest")
        self.assertIn("All tests passed", result.stdout)

  def test_build_only_does_not_run_tests(self):
    result, commands = self._run("--build_only=true")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(commands, ["cmake", "make -j1"])
    self.assertNotIn("All tests passed", result.stdout)

  def test_python3_fallback_is_preserved(self):
    result, commands = self._run(TEST_NO_PYTHON="1")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("trying to find python3", result.stdout)
    self.assertEqual(commands[-1], "ctest")

  def test_ci_failure_keeps_nonzero_status(self):
    result, commands = self._run("--github_ci=true", TEST_CMAKE_EXIT="17")
    self.assertEqual(result.returncode, 17)
    self.assertEqual(commands, ["cmake"])


if __name__ == "__main__":
  unittest.main()

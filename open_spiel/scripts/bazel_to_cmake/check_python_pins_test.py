# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for check_python_pins, on small fake copies of the pin files."""

import contextlib
import io
import os
import textwrap

from absl.testing import absltest

from open_spiel.scripts.bazel_to_cmake import check_python_pins

_EXTRA_DEPS = """\
#!/usr/bin/env bash
PY_VER="3.9"
if [ "$1" != "" ]; then
  PY_VER=`$1 --version | awk '{print $2}'`
fi
verlte() {
  stuff=`echo -e "$1\\n$2" | sort -V | head -n1`
  [ "$1" = "$stuff" ]
}
verlt() {
  [ "$1" = "$2" ] && return 1 || verlte $1 $2
}
echo "Set Python version: $PY_VER"
if verlt $PY_VER 3.12; then
  export OPEN_SPIEL_PYTHON_JAX_DEPS=""
  export OPEN_SPIEL_PYTHON_MISC_DEPS="scipy==1.0.0"
elif verlt $PY_VER 3.13; then
  export OPEN_SPIEL_PYTHON_JAX_DEPS="jax==0.12.0 dm-haiku==0.0.17"
  export OPEN_SPIEL_PYTHON_MISC_DEPS="scipy==1.16.3"
else
  export OPEN_SPIEL_PYTHON_JAX_DEPS="jax==0.13.0 dm-haiku==0.0.17"
  export OPEN_SPIEL_PYTHON_MISC_DEPS="scipy==1.17.0"
fi
export OPEN_SPIEL_PYTHON_TEST_DEPS="pytest==9.1.1"
"""

_REQUIREMENTS_IN = """\
# Comment.
absl-py>=0.10.0
jax==0.12.0
dm_haiku==0.0.17
scipy == 1.16.3
pytest==9.1.1
etils[epath]
"""

_LOCK = """\
absl-py==2.3.1 \\
    --hash=sha256:aaaa
    # via -r requirements.in
dm-haiku==0.0.17 \\
    --hash=sha256:bbbb
etils==1.14.0 \\
    --hash=sha256:cccc
jax==0.12.0 \\
    --hash=sha256:dddd
pytest==9.1.1 \\
    --hash=sha256:eeee
scipy==1.16.3 ; sys_platform == 'linux' \\
    --hash=sha256:ffff
"""


class CheckPythonPinsTest(absltest.TestCase):
  """Runs the checker on small copies of the pin files."""

  def setUp(self):
    super().setUp()
    self.open_spiel = self.create_tempdir("repo/open_spiel").full_path
    self.bazel_python = self.create_tempdir("repo/bazel/python").full_path
    self.write("scripts/python_extra_deps.sh", _EXTRA_DEPS)
    self.write("requirements.in", _REQUIREMENTS_IN, self.bazel_python)
    self.write("requirements_lock.txt", _LOCK, self.bazel_python)
    self.write("python.MODULE.bazel", '_PYTHON_VERSION = "3.12"\n',
               self.bazel_python)
    self.write("update_requirements_lock.sh",
               'readonly PYTHON_VERSION="3.12"\n', self.bazel_python)
    self.files = check_python_pins.PinFiles.find(self.open_spiel)

  def write(self, rel: str, text: str, root: str = "") -> None:
    path = os.path.join(root or self.open_spiel, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
      f.write(text)

  def edit(self, path: str, old: str, new: str) -> None:
    with open(path) as f:
      text = f.read()
    self.assertIn(old, text)
    with open(path, "w") as f:
      f.write(text.replace(old, new))

  def test_finds_exported_layout(self):
    self.assertEqual(self.files.requirements_in,
                     os.path.join(self.bazel_python, "requirements.in"))
    self.assertEqual(
        self.files.extra_deps,
        os.path.join(self.open_spiel, "scripts", "python_extra_deps.sh"))

  def test_finds_source_layout(self):
    source = self.create_tempdir("source").full_path
    os.makedirs(os.path.join(source, "opensource", "bazel", "python"))
    files = check_python_pins.PinFiles.find(source)
    self.assertEqual(
        files.lock,
        os.path.join(source, "opensource", "bazel", "python",
                     "requirements_lock.txt"))

  def test_parse_requirements(self):
    self.assertEqual(
        check_python_pins.parse_requirements(_REQUIREMENTS_IN), {
            "absl_py": ">=0.10.0",
            "jax": "==0.12.0",
            "dm_haiku": "==0.0.17",
            "scipy": "==1.16.3",
            "pytest": "==9.1.1",
            "etils": "",
        })
    self.assertEqual(
        check_python_pins.parse_requirements(_LOCK)["scipy"], "==1.16.3")

  def test_extra_deps_selects_the_python_version(self):
    values = check_python_pins.extra_deps(self.files.extra_deps, "3.12")
    self.assertEqual(values["OPEN_SPIEL_PYTHON_MISC_DEPS"], "scipy==1.16.3")
    self.assertEqual(values["OPEN_SPIEL_PYTHON_TEST_DEPS"], "pytest==9.1.1")
    values = check_python_pins.extra_deps(self.files.extra_deps, "3.11")
    self.assertEqual(values["OPEN_SPIEL_PYTHON_JAX_DEPS"], "")
    self.assertEqual(values["OPEN_SPIEL_PYTHON_MISC_DEPS"], "scipy==1.0.0")

  def test_extra_deps_pins(self):
    self.assertEqual(
        check_python_pins.extra_deps_pins({
            "OPEN_SPIEL_PYTHON_JAX_DEPS": "dm-haiku==0.0.17",
            "OPEN_SPIEL_PYTHON_PYTORCH_DEPS": "",
        }), {"dm_haiku": ("==0.0.17", "OPEN_SPIEL_PYTHON_JAX_DEPS")})
    with self.assertRaisesRegex(ValueError, "torch>=2"):
      check_python_pins.extra_deps_pins(
          {"OPEN_SPIEL_PYTHON_PYTORCH_DEPS": "torch>=2"})

  def test_agreeing_pins(self):
    self.assertEqual(check_python_pins.problems(self.files), [])

  def test_extra_deps_disagree_with_requirements_in(self):
    self.edit(self.files.extra_deps, 'DEPS="scipy==1.16.3"',
              'DEPS="scipy==1.16.2"')
    self.assertEqual(check_python_pins.problems(self.files), [
        "scipy: OPEN_SPIEL_PYTHON_MISC_DEPS in python_extra_deps.sh pins "
        "==1.16.2 for Python 3.12, requirements.in has ==1.16.3"
    ])

  def test_test_deps_are_checked(self):
    self.edit(self.files.extra_deps, "pytest==9.1.1", "pytest==9.1.2")
    (problem,) = check_python_pins.problems(self.files)
    self.assertStartsWith(problem, "pytest: OPEN_SPIEL_PYTHON_TEST_DEPS")

  def test_unpinned_extra_dep(self):
    self.edit(self.files.requirements_in, "dm_haiku==0.0.17\n", "dm_haiku\n")
    (problem,) = check_python_pins.problems(self.files)
    self.assertEqual(
        problem, "dm_haiku: OPEN_SPIEL_PYTHON_JAX_DEPS in python_extra_deps.sh"
        " pins ==0.0.17 for Python 3.12, requirements.in has no pin")

  def test_other_python_versions_are_ignored(self):
    self.edit(self.files.extra_deps, "jax==0.13.0", "jax==0.14.0")
    self.assertEqual(check_python_pins.problems(self.files), [])

  def test_lock_disagrees_with_requirements_in(self):
    self.edit(self.files.lock, "jax==0.12.0", "jax==0.12.1")
    self.assertEqual(check_python_pins.problems(self.files), [
        "jax: requirements.in pins ==0.12.0, requirements_lock.txt has "
        "==0.12.1"
    ])

  def test_requirement_missing_from_lock(self):
    self.edit(self.files.lock, "etils==", "etilz==")
    self.assertEqual(check_python_pins.problems(self.files), [
        "etils: in requirements.in but not in requirements_lock.txt"
    ])

  def test_python_versions_must_agree(self):
    self.edit(self.files.update_script, "3.12", "3.13")
    (problem,) = check_python_pins.problems(self.files)
    self.assertIn("locks for Python 3.13", problem)

  def test_main(self):
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
      self.assertEqual(
          check_python_pins.main(["--open_spiel_dir", self.open_spiel]), 0)
      self.edit(self.files.lock, "jax==0.12.0", "jax==0.12.1")
      self.assertEqual(
          check_python_pins.main(["--open_spiel_dir", self.open_spiel]), 1)
    self.assertEqual(stdout.getvalue(), "Python pins agree.\n")
    self.assertEqual(
        stderr.getvalue().splitlines()[0],
        textwrap.dedent("""\
        error: jax: requirements.in pins ==0.12.0, requirements_lock.txt has \
==0.12.1"""))


if __name__ == "__main__":
  absltest.main()

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


"""Tests for the server's health report."""

import pathlib
import subprocess
from unittest import mock

from absl.testing import absltest

from open_spiel.python.gui import contract
from open_spiel.python.gui import health_report

_COMMIT = "0123456789abcdef0123456789abcdef01234567"


class HealthReportTest(absltest.TestCase):

  def test_reports_the_code_and_its_capabilities(self) -> None:
    report = health_report.build()
    self.assertTrue(report.version)
    self.assertTrue(report.revision is None or len(report.revision) >= 40)
    self.assertEqual(report.capabilities.controllers[0].name, contract.HUMAN)
    self.assertEqual(
        set(report.capabilities.chance_policies), set(contract.ChancePolicy)
    )

  def test_is_stable_across_calls(self) -> None:
    self.assertEqual(health_report.build(), health_report.build())


class RevisionTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.root = pathlib.Path(self.create_tempdir().full_path).resolve()
    self.package = self.root / "open_spiel" / "python" / "gui"

  def revision(self, git_output: str) -> str | None:
    """Returns the revision read from git's output."""
    result = subprocess.CompletedProcess([], 0, stdout=git_output)
    with mock.patch.object(subprocess, "run", return_value=result):
      return health_report.revision(self.package)

  def test_reports_the_commit_of_an_open_spiel_tree(self) -> None:
    self.assertEqual(self.revision(f"{self.root}\n{_COMMIT}\n"), _COMMIT)

  def test_ignores_a_repository_the_package_is_merely_inside(self) -> None:
    self.assertIsNone(self.revision(f"{self.package.parent}\n{_COMMIT}\n"))
    self.assertIsNone(self.revision(f"/elsewhere\n{_COMMIT}\n"))

  def test_reports_nothing_without_git(self) -> None:
    with mock.patch.object(subprocess, "run", side_effect=OSError("no git")):
      self.assertIsNone(health_report.revision(self.package))


if __name__ == "__main__":
  absltest.main()

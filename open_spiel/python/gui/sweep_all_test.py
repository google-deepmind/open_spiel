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

"""Tests for the all-games sweep's command line, on named games only."""

import contextlib
import io

from absl.testing import absltest

from open_spiel.python.gui import sweep_all


class MainTest(absltest.TestCase):

  def test_named_games_that_agree_with_the_list_pass(self) -> None:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      status = sweep_all.main(["tic_tac_toe", "add_noise"])
    lines = stdout.getvalue().splitlines()
    self.assertEqual(status, 0, stdout.getvalue())
    self.assertRegex(lines[0], r"^ok    tic_tac_toe: passed after \d+ plies$")
    self.assertRegex(lines[1], r"^ok    add_noise: refused after 0 plies")
    self.assertEqual(lines[-1], "PASSED: 2 swept, 0 problems, 0 not judged")

  def test_unregistered_game_is_refused(self) -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
      with self.assertRaises(SystemExit) as raised:
        sweep_all.main(["no_such_game"])
    self.assertEqual(raised.exception.code, 2)
    self.assertIn("not registered: no_such_game", stderr.getvalue())


if __name__ == "__main__":
  absltest.main()

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


"""Tests for the check that C++ search can run on Python games."""

import subprocess
from unittest import mock

from absl.testing import absltest

from open_spiel.python.games import iterated_prisoners_dilemma  # pylint: disable=unused-import
from open_spiel.python.games import tic_tac_toe  # pylint: disable=unused-import
from open_spiel.python.gui import gil_safety
import pyspiel


class UsesPythonTest(absltest.TestCase):

  def test_a_game_implemented_in_cpp_does_not(self) -> None:
    self.assertFalse(gil_safety.uses_python(pyspiel.load_game("chess")))

  def test_a_game_implemented_in_python_does(self) -> None:
    game = pyspiel.load_game("python_tic_tac_toe")
    self.assertTrue(gil_safety.uses_python(game))

  def test_a_cpp_wrapper_around_a_cpp_game_does_not(self) -> None:
    game = pyspiel.convert_to_turn_based(pyspiel.load_game("matrix_pd"))
    self.assertFalse(gil_safety.uses_python(game))

  def test_a_cpp_wrapper_around_a_python_game_does(self) -> None:
    game = pyspiel.convert_to_turn_based(
        pyspiel.load_game("python_iterated_prisoners_dilemma")
    )
    self.assertEqual(type(game).__module__, pyspiel.__name__)
    self.assertTrue(gil_safety.uses_python(game))


def _exit(returncode: int) -> subprocess.CompletedProcess[str]:
  """Returns a child interpreter's result with `returncode`."""
  return subprocess.CompletedProcess(
      args=[], returncode=returncode, stdout="", stderr="the child's stderr"
  )


class PythonGamesAreSafeTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    gil_safety._probe.cache_clear()  # pylint: disable=protected-access
    self.addCleanup(gil_safety._probe.cache_clear)  # pylint: disable=protected-access

  def _run(
      self,
      return_value: object = None,
      side_effect: BaseException | None = None,
  ) -> mock.MagicMock:
    """Replaces the child interpreter for the rest of the test."""
    patcher = mock.patch.object(
        subprocess,
        "run",
        autospec=True,
        return_value=return_value,
        side_effect=side_effect,
    )
    self.addCleanup(patcher.stop)
    run: mock.MagicMock = patcher.start()
    return run

  def test_a_search_that_completes_is_safe(self) -> None:
    self._run(return_value=_exit(0))
    self.assertTrue(gil_safety.python_games_are_safe())

  def test_a_search_that_crashes_the_interpreter_is_unsafe(self) -> None:
    self._run(return_value=_exit(-11))
    self.assertFalse(gil_safety.python_games_are_safe())

  def test_a_search_that_does_not_finish_is_unsafe_and_cached(self) -> None:
    run = self._run(
        side_effect=subprocess.TimeoutExpired(cmd="probe", timeout=1)
    )
    self.assertFalse(gil_safety.python_games_are_safe())
    self.assertFalse(gil_safety.python_games_are_safe())
    run.assert_called_once()

  def test_a_child_that_fails_otherwise_is_an_error(self) -> None:
    self._run(return_value=_exit(1))
    with self.assertRaisesRegex(RuntimeError, "the child's stderr"):
      gil_safety.python_games_are_safe()

  def test_the_probe_runs_once_per_process(self) -> None:
    run = self._run(return_value=_exit(0))
    gil_safety.python_games_are_safe()
    gil_safety.python_games_are_safe()
    run.assert_called_once()

  def test_the_working_directory_is_not_on_the_childs_path(self) -> None:
    run = self._run(return_value=_exit(0))
    gil_safety.python_games_are_safe()
    self.assertIn("-P", run.call_args.args[0])


if __name__ == "__main__":
  absltest.main()

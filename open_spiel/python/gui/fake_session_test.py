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

"""Tests for the stand-in session."""

from absl.testing import absltest

from open_spiel.python.gui import contract
from open_spiel.python.gui import fake_session

_ORACLE = contract.Viewer(contract.ViewerKind.ORACLE)


class FakeSessionTest(absltest.TestCase):

  def test_refuses_a_simultaneous_game(self) -> None:
    with self.assertRaisesRegex(ValueError, "not a sequential game"):
      fake_session.FakeSession("matrix_rps")

  def test_plays_a_sequential_game_to_its_end(self) -> None:
    session = fake_session.FakeSession("tic_tac_toe", seed=5)
    session.advance(_ORACLE)
    self.assertTrue(session.is_terminal)


if __name__ == "__main__":
  absltest.main()

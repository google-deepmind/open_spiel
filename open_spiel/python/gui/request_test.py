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

"""Tests for the invariants of the request bodies."""

from absl.testing import absltest

from open_spiel.python.gui import contract
from open_spiel.python.gui import request


class RequestTest(absltest.TestCase):

  def test_a_human_seat_cannot_autoplay(self) -> None:
    with self.assertRaisesRegex(ValueError, "human seat"):
      request.SeatConfig(contract.HUMAN, {}, True)
    self.assertTrue(request.SeatConfig("mcts", {}, True).autoplay)

  def test_a_seed_is_not_negative(self) -> None:
    with self.assertRaisesRegex(ValueError, "seed -1"):
      request.CreateSession(
          "kuhn_poker", {}, (), contract.ChancePolicy.AUTO, -1
      )
    request.CreateSession("kuhn_poker", {}, (), contract.ChancePolicy.AUTO, 0)

  def test_an_advance_applies_a_bounded_number_of_actions(self) -> None:
    with self.assertRaisesRegex(ValueError, "at most 0"):
      request.Advance(max_plies=0)
    with self.assertRaisesRegex(ValueError, "from 1 to 1000"):
      request.Advance(max_plies=request.MAX_ADVANCE_PLIES + 1)
    self.assertEqual(request.Advance(max_plies=1).limit, 1)
    self.assertEqual(
        request.Advance(max_plies=None).limit, request.MAX_ADVANCE_PLIES
    )


if __name__ == "__main__":
  absltest.main()

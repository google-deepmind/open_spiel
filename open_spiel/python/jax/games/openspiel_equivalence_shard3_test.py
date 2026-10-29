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

"""Shard 3 of openspiel_equivalence's move-for-move playouts."""

from absl.testing import absltest

from open_spiel.python.jax.games import openspiel_equivalence as equivalence


class PyspielEquivalenceShard3Test(absltest.TestCase):
  """Shard 3 of the move-for-move playouts."""

  def test_move_for_move_equivalence(self):
    """This shard's move-for-move playouts match OpenSpiel."""
    for game_str, count in equivalence.run_playouts(
        self,
        equivalence.move_for_move_configs(),
        file_shard=(3, equivalence.MOVE_SHARDS),
    ).items():
      if count:
        print(f"openspiel_equivalence: game={game_str} truncated={count}")

  @equivalence.needs_universal_poker
  def test_move_for_move_equivalence_no_limit_holdem(self):
    """This shard's no_limit_holdem playouts match OpenSpiel."""
    equivalence.run_playouts(
        self,
        [equivalence.NO_LIMIT_HOLDEM_CONFIG],
        file_shard=(3, equivalence.MOVE_SHARDS),
    )


if __name__ == "__main__":
  absltest.main()

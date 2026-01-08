# Copyright 2019 DeepMind Technologies Limited
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

"""Tests for the game-specific functions for bargaining."""


from absl.testing import absltest

import pyspiel
barg = pyspiel.bargaining


class GamesBargainingTest(absltest.TestCase):

  def test_constants(self):
    self.assertEqual(barg.NumItemTypes, 3)
    self.assertEqual(barg.PoolMinNumItems, 5)
    self.assertEqual(barg.PoolMaxNumItems, 7)
    self.assertEqual(barg.TotalValueAllItems, 10)

  def test_game_specific_constants(self):
    game0 = pyspiel.load_game("bargaining")
    self.assertEqual(game0.max_turns(), 10)
    self.assertEqual(game0.discount(), 1.0)
    self.assertEqual(game0.prob_end(), 0.0)

    game1 = pyspiel.load_game(
        "bargaining(max_turns=15,discount=0.9,prob_end=0.1)"
    )
    self.assertEqual(game1.max_turns(), 15)
    self.assertEqual(game1.discount(), 0.9)
    self.assertEqual(game1.prob_end(), 0.1)

  def test_game_mechanism(self):
    game = pyspiel.load_game("bargaining")
    state = game.new_initial_state()

    # first check the instance matches the true instance
    true_instance = [(1, 2, 3), (8, 1, 0), (4, 0, 2)]
    state.apply_action(0)
    cur_instance = state.instance()
    cur_instance = [
        tuple(cur_instance.pool),
        tuple(cur_instance.values[0]),
        tuple(cur_instance.values[1])
    ]

    for item1, item2 in zip(true_instance, cur_instance):
      self.assertEqual(item1, item2)

    # then set a new instance and check it works
    all_instances = game.all_instances()
    new_instance = all_instances[2]
    state.set_instance(new_instance)
    new_instance = [
        tuple(new_instance.pool),
        tuple(new_instance.values[0]),
        tuple(new_instance.values[1])
    ]
    cur_instance = state.instance()
    cur_instance = [
        tuple(cur_instance.pool),
        tuple(cur_instance.values[0]),
        tuple(cur_instance.values[1])
    ]
    for item1, item2 in zip(cur_instance, new_instance):
      self.assertEqual(item1, item2)

  def test_offer_and_instance_map(self):
    game = pyspiel.load_game("bargaining")
    all_offers = game.all_offers()
    for i, offer in enumerate(all_offers):
      self.assertEqual(game.get_offer_index(offer), i)
    for i, instance in enumerate(game.all_instances()):
      self.assertEqual(game.get_instance_index(instance), i)

  def test_get_possible_opponent_values(self):
    game = pyspiel.load_game("bargaining")
    self.assertEqual(
        game.get_possible_opponent_values(0, [1, 2, 3], [8, 1, 0]),
        [
            [4, 0, 2],
            [7, 0, 1],
            [1, 3, 1],
        ],
    )


if __name__ == "__main__":
  absltest.main()

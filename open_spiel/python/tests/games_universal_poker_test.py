# Copyright 2025 DeepMind Technologies Limited
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

"""Tests for Python bindings of universal poker."""

import pickle
from absl.testing import absltest
import numpy as np

import pyspiel


class GamesUniversalPokerTest(absltest.TestCase):

  def test_load_game(self):
    game = pyspiel.load_game("universal_poker")
    state = game.new_initial_state()
    self.assertFalse(state.is_terminal())

  def test_state_struct(self):
    game = pyspiel.load_game(
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,"
        "blind=100 50,firstPlayer=2 1 1 1,numSuits=4,numRanks=13,"
        "numHoleCards=2,numBoardCards=0 3 1 1,stack=20000 20000,"
        "bettingAbstraction=fullgame,calcOddsNumSims=100000)")
    state = game.new_initial_state()
    state_struct = state.to_struct()
    self.assertEqual(state_struct.current_player, pyspiel.PlayerId.CHANCE)
    self.assertEqual(state_struct.blinds, [100, 50])
    self.assertEqual(state_struct.betting_history, "")
    self.assertEqual(state_struct.pot_size, 150)
    self.assertEqual(state_struct.starting_stacks, [20000, 20000])
    self.assertEmpty(state_struct.board_cards)
    self.assertLen(state_struct.player_hands, 2)
    self.assertEmpty(state_struct.player_hands[0])
    self.assertEmpty(state_struct.player_hands[1])
    self.assertEqual(state_struct.player_contributions, [100, 50])
    self.assertEqual(state_struct.best_hand_rank_types,
                     ["High Card", "High Card"])
    self.assertEmpty(state_struct.best_five_card_hands[0])
    self.assertEmpty(state_struct.best_five_card_hands[1])
    self.assertLen(state_struct.odds, 4)

    state.apply_action(state.string_to_action("player=-1 move=Deal As"))
    state.apply_action(state.string_to_action("player=-1 move=Deal Ac"))
    state.apply_action(state.string_to_action("player=-1 move=Deal Ks"))
    state.apply_action(state.string_to_action("player=-1 move=Deal Kc"))
    state.apply_action(state.string_to_action("player=1 move=Bet300"))
    state.apply_action(state.string_to_action("player=0 move=Bet1000"))
    state.apply_action(state.string_to_action("player=1 move=Call"))
    state.apply_action(state.string_to_action("player=-1 move=Deal Qc"))
    state.apply_action(state.string_to_action("player=-1 move=Deal Js"))
    state.apply_action(state.string_to_action("player=-1 move=Deal Tc"))
    state.apply_action(state.string_to_action("player=0 move=Call"))
    state_struct = state.to_struct()
    self.assertEqual(state_struct.board_cards, "QcJsTc")
    self.assertEqual(state_struct.player_hands[0], "AsAc")
    self.assertEqual(state_struct.player_hands[1], "KsKc")
    self.assertEqual(state_struct.player_contributions, [1000, 1000])
    self.assertEqual(state_struct.best_hand_rank_types, ["Pair", "Pair"])
    self.assertEqual(state_struct.best_five_card_hands[0], "AsAcQcJsTc")
    self.assertEqual(state_struct.best_five_card_hands[1], "KsKcQcJsTc")
    state.apply_action(state.string_to_action("player=1 move=Call"))
    state.apply_action(state.string_to_action("player=-1 move=Deal 9d"))
    state_struct = state.to_struct()
    self.assertEqual(state_struct.best_hand_rank_types, ["Pair", "Straight"])
    self.assertLess(state_struct.odds[0], state_struct.odds[2])
    self.assertEqual(state_struct.odds[1], state_struct.odds[3])
    self.assertEqual(state_struct.odds[1], 0.0)

  def test_random_game(self):
    game = pyspiel.load_game("universal_poker")
    state = game.new_initial_state()
    rng = np.random.RandomState(0)
    for _ in range(10):
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes = state.chance_outcomes()
          action_list, prob_list = zip(*outcomes)
          action = rng.choice(action_list, p=prob_list)
          state.apply_action(action)
        else:
          state.apply_action(rng.choice(state.legal_actions()))
      state.returns()
      state = game.new_initial_state()

  def test_pickle(self):
    game = pyspiel.load_game("universal_poker")
    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(1)
    pickled_state = pickle.dumps(state)
    unpickled_state = pickle.loads(pickled_state)
    self.assertEqual(str(state), str(unpickled_state))


if __name__ == "__main__":
  if "universal_poker" in pyspiel.registered_names():
    absltest.main()
  else:
    print("universal_poker is not registered. Skipping test.")

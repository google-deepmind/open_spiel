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

import json
from absl.testing import absltest
import numpy as np

import pyspiel


class GamesRepeatedPokerTest(absltest.TestCase):

  def test_bindings(self):
    acpc_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 50,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,"
        "numBoardCards=0 3 1 1,stack=20000 20000)"
    )
    game_string = (
        f"repeated_poker(universal_poker_game_string={acpc_game_string},"
        "max_num_hands=3,reset_stacks=True,rotate_dealer=True)"
    )
    game = pyspiel.load_game(game_string)
    state = game.new_initial_state()

    self.assertEqual(state.dealer(), 1)
    self.assertEqual(state.small_blind(), 50)
    self.assertEqual(state.big_blind(), 100)
    self.assertEqual(state.stacks(), [20000, 20000])
    self.assertEqual(state.player_to_seat(0), 0)
    self.assertEqual(state.player_to_seat(1), 1)
    self.assertEqual(state.seat_to_player(0), 0)
    self.assertEqual(state.seat_to_player(1), 1)
    self.assertEqual(state.dealer_seat(), 1)
    self.assertEqual(state.small_blind_seat(), 1)
    self.assertEqual(state.big_blind_seat(), 0)
    self.assertEqual(state.acpc_hand_histories(), [])

    rng = np.random.RandomState(0)
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = rng.choice(action_list, p=prob_list)
        state.apply_action(action)
      else:
        state.apply_action(rng.choice(state.legal_actions()))
    self.assertLen(state.acpc_hand_histories(), 3)
    for history in state.acpc_hand_histories():
      self.assertIsInstance(history, str)

  def test_state_struct(self):
    acpc_game_string = pyspiel.hunl_game_string("fullgame")
    game_string = (
        f"repeated_poker(universal_poker_game_string={acpc_game_string},"
        "max_num_hands=100,reset_stacks=True,rotate_dealer=True)")
    game = pyspiel.load_game(game_string)
    state = game.new_initial_state()
    state_struct = state.to_struct()
    self.assertEqual(state_struct.hand_number, 0)
    self.assertEqual(state_struct.max_num_hands, 100)
    self.assertEqual(state_struct.stacks, [20000, 20000])
    self.assertEqual(state_struct.dealer, 1)
    self.assertEqual(state_struct.small_blind, 50)
    self.assertEqual(state_struct.big_blind, 100)
    self.assertLen(state_struct.hand_returns, 1)
    self.assertEqual(state_struct.hand_returns[0], [0.0, 0.0])
    up_json = json.loads(state_struct.current_universal_poker_json)
    self.assertEqual(up_json["current_player"], pyspiel.PlayerId.CHANCE)
    self.assertEqual(state_struct.prev_universal_poker_json, "")
    state.apply_action(state.string_to_action("player=-1 move=Deal 2c"))
    state.apply_action(state.string_to_action("player=-1 move=Deal 2d"))
    state.apply_action(state.string_to_action("player=-1 move=Deal Tc"))
    state.apply_action(state.string_to_action("player=-1 move=Deal 2s"))
    state.apply_action(state.string_to_action("player=1 move=Bet300"))
    state.apply_action(state.string_to_action("player=0 move=Fold"))
    self.assertEqual(state.returns(), [0.0, 0.0])
    state_struct = state.to_struct()
    prev_up_json = json.loads(state_struct.prev_universal_poker_json)
    self.assertEqual(prev_up_json["player_hands"], ["2d2c", "Tc2s"])


if __name__ == "__main__":
  if "universal_poker" in pyspiel.registered_names():
    absltest.main()
  else:
    print("universal_poker is not registered. Skipping test.")

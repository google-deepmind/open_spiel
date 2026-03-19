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

# Lint as python3
"""Tests for basic heuristics."""

import functools
from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import minimax
from open_spiel.python.games import heuristics
import pyspiel


def simulate_random_game(game: pyspiel.Game):
  """Simulates a game."""
  state = game.new_initial_state()
  while not state.is_terminal():
    # The state can be three different types: chance node,
    # simultaneous node, or decision node
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(action_list, p=prob_list)
      state.apply_action(action)
    elif state.is_simultaneous_node():
      heuristic_value = heuristics.evaluate_state(state, state.current_player())
      print(f"---\nState is: \n{state}\nHeuristic value: {heuristic_value}\n")
      random_choice = lambda a: np.random.choice(a) if a else [0]
      chosen_actions = [
          random_choice(state.legal_actions(pid))
          for pid in range(game.num_players())
      ]
      state.apply_actions(chosen_actions)
    else:
      heuristic_value = heuristics.evaluate_state(state, state.current_player())
      print(f"---\nState is: \n{state}\nHeuristic value: {heuristic_value}\n")
      action = np.random.choice(state.legal_actions(state.current_player()))
      state.apply_action(action)


class HeuristicsTest(absltest.TestCase):

  def test_can_call_heuristic_callback(self):
    """Checks we can create the game and a state."""
    for game_str in heuristics.HEURISTIC_CALLBACKS:
      game = pyspiel.load_game(game_str)
      simulate_random_game(game)

  def test_minimax_chess(self):
    """Tests minimax for chess."""
    game = pyspiel.load_game("chess")
    state = game.new_initial_state()
    while not state.is_terminal():
      player = state.current_player()
      if player == 0:
        action = np.random.choice(state.legal_actions(state.current_player()))
      else:
        _, action = minimax.alpha_beta_search(
            game=game,
            state=state,
            value_function=functools.partial(heuristics.evaluate_state,
                                             player=player),
            maximum_depth=2,
            maximizing_player_id=player)
      print(f"State is: \n{state.debug_string()}\n")
      print(f"Taking action: {state.action_to_string(player, action)}\n")
      state.apply_action(action)


if __name__ == "__main__":
  absltest.main()

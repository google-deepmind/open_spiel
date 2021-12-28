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

"""Tests for open_spiel.python.algorithms.get_all_states."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from open_spiel.python.algorithms import get_all_states
import pyspiel


class GetAllStatesTest(absltest.TestCase):

  def test_tic_tac_toe_number_histories(self):
    game = pyspiel.load_game("tic_tac_toe")
    states = get_all_states.get_all_states(
        game,
        depth_limit=-1,
        include_terminals=True,
        include_chance_states=False,
        to_string=lambda s: s.history_str())
    self.assertLen(states, 549946)
    states = get_all_states.get_all_states(
        game,
        depth_limit=-1,
        include_terminals=True,
        include_chance_states=False,
        to_string=str)
    self.assertLen(states, 5478)

  def test_simultaneous_python_game_get_all_state(self):
    game = pyspiel.load_game(
        "python_iterated_prisoners_dilemma(max_game_length=6)")
    states = get_all_states.get_all_states(
        game,
        depth_limit=-1,
        include_terminals=True,
        include_chance_states=False,
        to_string=lambda s: s.history_str())
    self.assertLen(states, 10921)
    states = get_all_states.get_all_states(
        game,
        depth_limit=-1,
        include_terminals=True,
        include_chance_states=False,
        to_string=str)
    self.assertLen(states, 5461)

  def test_simultaneous_game_get_all_state(self):
    game = game = pyspiel.load_game("goofspiel", {"num_cards": 3})
    states = get_all_states.get_all_states(
        game,
        depth_limit=-1,
        include_terminals=True,
        include_chance_states=False,
        to_string=lambda s: s.history_str())
    self.assertLen(states, 273)


if __name__ == "__main__":
  absltest.main()

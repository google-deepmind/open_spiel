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

"""Tests for the extra_info game transformation."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import pyspiel


class GamesExtraInfoTest(parameterized.TestCase):
    def test_bindings(self):
        game_name = 'matrix_pd'
        game = pyspiel.create_repeated_game(game_name, {
           "num_repetitions": 8,"recall": 4})

        state = game.new_initial_state()
        old_ot_len = len(state.observation_tensor(0))
        old_it_len = len(state.information_state_tensor(0))
        extra_info_game = pyspiel.create_extra_info_game(game,
          {"extra_observation_tensor_size":1,
          "extra_information_tensor_size":1} )
        state = extra_info_game.new_initial_state()
        assert len(state.observation_tensor(0)) == old_ot_len + 1
        assert len(state.information_state_tensor(0)) == old_it_len + 1
        assert state.observation_tensor(0)[-1] == 0.0 # initial values
        assert state.information_state_tensor(0)[-1] == 0
        # setting value withot a player id affects all players
        old_is0 = state.information_state_string(0)
        old_is1 = state.information_state_string(1)
        old_os0 = state.observation_string(0)
        old_os1 = state.observation_string(1)
        state.set_extra_observation_tensor(values=[0.2])
        state.set_extra_information_state_tensor(values=[0.4])
        state.set_extra_observation_string("green")
        state.set_extra_information_state_string("Monoecious")
        assert state.observation_string(0) == old_os0 + "green"
        assert state.observation_string(1) == old_os1 + "green"
        assert state.information_state_string(0) == old_is0 + "Monoecious"
        assert state.information_state_string(1) == old_is1 + "Monoecious"
        self.assertAlmostEqual(state.observation_tensor(0)[-1], 0.2)
        self.assertAlmostEqual(state.observation_tensor(1)[-1], 0.2)
        self.assertAlmostEqual(state.information_state_tensor(0)[-1], 0.4)
        self.assertAlmostEqual(state.information_state_tensor(1)[-1], 0.4)
        # check individual seters and getters
        state.set_extra_observation_string(0, "yellow")
        state.set_extra_observation_string(1, "blue")
        state.set_extra_information_state_string(0, "Gynomonoecious")
        state.set_extra_information_state_string(1, "Andromonoecious")
        state.set_extra_observation_tensor(player=0, values=[0.22])
        state.set_extra_observation_tensor(player=1, values=[0.26])
        state.set_extra_information_state_tensor(0,[0.44])
        state.set_extra_information_state_tensor(1,[0.46])
        assert state.observation_string(0) == old_os0 + "yellow"
        assert state.observation_string(1) == old_os1 + "blue"
        assert state.information_state_string(0) == old_is0 + "Gynomonoecious"
        assert state.information_state_string(1) == old_is1 + "Andromonoecious"
        self.assertAlmostEqual(state.observation_tensor(0)[-1], 0.22)
        self.assertAlmostEqual(state.observation_tensor(1)[-1], 0.26)
        self.assertAlmostEqual(state.information_state_tensor(0)[-1], 0.44)
        self.assertAlmostEqual(state.information_state_tensor(1)[-1], 0.46)
        # wrong size should raise error
        with self.assertRaises(pyspiel.SpielError):
           state.set_extra_observation_tensor(values=[0.2, 0.3])
        with self.assertRaises(pyspiel.SpielError):
           state.set_extra_information_state_tensor(player=0, values=[])


if __name__ == "__main__":
  absltest.main()

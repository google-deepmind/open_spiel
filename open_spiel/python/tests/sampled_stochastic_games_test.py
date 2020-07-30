# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python.tests.test_utils import random_playout

import pyspiel

# All games with kSampledStochastic chance mode.
SPIEL_SAMPLED_STOCHASTIC_GAMES_LIST = [
    g for g in pyspiel.registered_games() if g.default_loadable
    and g.chance_mode == pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC
]
assert len(SPIEL_SAMPLED_STOCHASTIC_GAMES_LIST) >= 2


class SampledStochasticGamesTest(parameterized.TestCase):
    @parameterized.parameters(*SPIEL_SAMPLED_STOCHASTIC_GAMES_LIST)
    def test_stateful_game_serialization(self, game_info):
        game = pyspiel.load_game(game_info.short_name,
                                 {"rng_seed": pyspiel.GameParameter(0)})
        # Mutate game's internal RNG state.
        state = random_playout(game.new_initial_state())
        deserialized_game = pickle.loads(pickle.dumps(game))

        # Make sure initial states are the same after game deserialization.
        state1 = random_playout(game.new_initial_state())
        d_state1 = random_playout(deserialized_game.new_initial_state())
        # TODO: If game with default parameters doesn't use the RNG
        # below assertion fails.
        # self.assertNotEqual(str(state1), str(state))
        self.assertEqual(str(state1), str(d_state1))
        # try one more time
        state2 = random_playout(game.new_initial_state())
        d_state2 = random_playout(deserialized_game.new_initial_state())
        # TODO: If game with default parameters doesn't use the RNG
        # below assertion fails.
        # self.assertNotEqual(str(state2), str(state1))
        self.assertEqual(str(state2), str(d_state2))


if __name__ == "__main__":
    absltest.main()

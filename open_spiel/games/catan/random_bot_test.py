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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from open_spiel.games.catan import py_catan  # pylint: disable=unused-import
from open_spiel.python.bots import uniform_random
import pyspiel


class RandomBotTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(seed=34239871, max_turns=int(1e2)),
  ])
  def test_random_bot_plays(self, seed, max_turns):
    np.random.seed(seed)

    game = pyspiel.load_game('catan')
    state = game.new_initial_state()

    # Load bot
    bots = [uniform_random.UniformRandomBot(player_id=i, rng=np.random)
            for i in range(4)]

    while not state.is_terminal() and state.player_turns() < max_turns:
      if state.is_chance_node():
        # Chance node: sample an outcome
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = np.random.choice(action_list, p=prob_list)
        state.apply_action(action)
      else:
        # Decision node: sample action for the single current player
        action = bots[state.current_player()].step(state)
        state.apply_action(action)

    valid_exit = state.is_terminal() or (state.player_turns() >= max_turns)
    self.assertTrue(valid_exit, 'game loop exited prematurely')


if __name__ == '__main__':
  absltest.main()

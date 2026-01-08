# Copyright 2022 DeepMind Technologies Limited
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
"""Test the MCTS Agent."""

from absl.testing import absltest
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import mcts_agent


class MCTSAgentTest(absltest.TestCase):

  def test_tic_tac_toe_episode(self):
    env = rl_environment.Environment("tic_tac_toe", include_full_state=True)
    num_players = env.num_players
    num_actions = env.action_spec()["num_actions"]

    # Create the MCTS bot. Both agents can share the same bot in this case since
    # there is no state kept between searches. See mcts.py for more info about
    # the arguments.
    mcts_bot = mcts.MCTSBot(env.game, 1.5, 100, mcts.RandomRolloutEvaluator())

    agents = [
        mcts_agent.MCTSAgent(player_id=idx, num_actions=num_actions,
                             mcts_bot=mcts_bot)
        for idx in range(num_players)
    ]

    time_step = env.reset()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      agent_output = agents[player_id].step(time_step)
      time_step = env.step([agent_output.action])
    for agent in agents:
      agent.step(time_step)


if __name__ == "__main__":
  absltest.main()

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
"""Test the IS-MCTS Agent."""

from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import ismcts
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import mcts_agent


class MCTSAgentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="tic_tac_toe", game_string="kuhn_poker"),
      dict(testcase_name="leduc_poker", game_string="leduc_poker"),
  )
  def test_self_play_episode(self, game_string: str):
    env = rl_environment.Environment(game_string, include_full_state=True)
    num_players = env.num_players
    num_actions = env.action_spec()["num_actions"]

    # Create the MCTS bot. Both agents can share the same bot in this case since
    # there is no state kept between searches. See mcts.py for more info about
    # the arguments.
    ismcts_bot = ismcts.ISMCTSBot(
        game=env.game,
        uct_c=1.5,
        max_simulations=100,
        evaluator=mcts.RandomRolloutEvaluator())

    agents = [
        mcts_agent.MCTSAgent(
            player_id=idx, num_actions=num_actions, mcts_bot=ismcts_bot)
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest, parameterized
import numpy as np

from open_spiel.python.algorithms.ars import ARS
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.psro_v2.rl_policy import rl_policy_factory
import pyspiel

class ARSTest(absltest.TestCase):

    def test_run_game(self, game_name="kuhn_poker"):
        env = rl_environment.Environment(game_name)
        info_state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]

        agents = [ARS(session=None,
                      player_id=player_id,
                      info_state_size=info_state_size,
                      num_actions=num_actions) for player_id in [0,1]]
        for _ in range(2):
            time_step = env.reset()
            while not time_step.last():
                current_player = time_step.observations["current_player"]
                current_agent = agents[current_player]
                agent_output = current_agent.step(time_step)
                time_step = env.step([agent_output.action])

            for agent in agents:
                agent.step(time_step)

        for agent in agents:
            print(agent.theta)

    def test_ars_policy(self):
        ARSPolicy = rl_policy_factory(ARS)

if __name__ == "__main__":
    absltest.main()


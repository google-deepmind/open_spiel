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

from open_spiel.python import rl_environment
from open_spiel.python.jax import nfsp


class NFSPTest(absltest.TestCase):

  def test_run_kuhn(self):
    env = rl_environment.Environment("kuhn_poker")
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agents = [
        nfsp.NFSP(  # pylint: disable=g-complex-comprehension
            player_id,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=[16],
            reservoir_buffer_capacity=10,
            anticipatory_param=0.1,
            allow_checkpointing=False,
        )
        for player_id in [0, 1]
    ]
    for _ in range(10):
      time_step = env.reset()
      while not time_step.last():
        current_player = time_step.observations["current_player"]
        current_agent = agents[current_player]
        agent_output = current_agent.step(time_step)
        time_step = env.step([agent_output.action])
      for agent in agents:
        agent.step(time_step)

if __name__ == "__main__":
  absltest.main()

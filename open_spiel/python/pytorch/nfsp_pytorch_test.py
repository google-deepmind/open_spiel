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

import os

from absl.testing import absltest
import torch

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nfsp


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
        )
        for player_id in [0, 1]
    ]
    for unused_ep in range(10):
      time_step = env.reset()
      while not time_step.last():
        current_player = time_step.observations["current_player"]
        current_agent = agents[current_player]
        agent_output = current_agent.step(time_step)
        time_step = env.step([agent_output.action])
      for agent in agents:
        agent.step(time_step)

  def test_save_and_restore(self):
    env = rl_environment.Environment("kuhn_poker")
    agent_kwargs = dict(
        state_representation_size=env.observation_spec()["info_state"][0],
        num_actions=env.action_spec()["num_actions"],
        hidden_layers_sizes=[16],
        reservoir_buffer_capacity=100,
        anticipatory_param=0.5,
        min_buffer_size_to_learn=4,
        batch_size=4,
        learn_every=10,
        optimizer_str="adam",
    )
    agents = [
        nfsp.NFSP(player_id, seed=0, **agent_kwargs) for player_id in [0, 1]
    ]
    for unused_ep in range(50):
      time_step = env.reset()
      while not time_step.last():
        current_player = time_step.observations["current_player"]
        agent_output = agents[current_player].step(time_step)
        time_step = env.step([agent_output.action])
      for agent in agents:
        agent.step(time_step)
    agent = agents[0]
    self.assertNotIn(None, agent.loss)

    # A missing directory given as a str, to check both are handled.
    checkpoint_dir = os.path.join(self.create_tempdir().full_path, "agent_0")
    agent.save(checkpoint_dir)
    restored_agent = nfsp.NFSP(0, seed=1, **agent_kwargs)
    restored_agent.restore(checkpoint_dir)

    # pylint: disable=protected-access
    for restored, original in [
        (restored_agent._avg_network, agent._avg_network),
        (restored_agent._optimizer, agent._optimizer),
        (restored_agent._rl_agent._q_network, agent._rl_agent._q_network),
        (restored_agent._rl_agent._optimizer, agent._rl_agent._optimizer),
    ]:
      torch.testing.assert_close(restored.state_dict(), original.state_dict())
    # pylint: enable=protected-access
    self.assertEqual(restored_agent.step_counter, agent.step_counter)
    self.assertEqual(restored_agent.loss, agent.loss)


if __name__ == "__main__":
  absltest.main()

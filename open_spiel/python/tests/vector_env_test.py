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

import collections

from absl.testing import absltest
from open_spiel.python import rl_environment
from open_spiel.python import vector_env


class VectorEnvTest(absltest.TestCase):

  def test_async_vector_env_tic_tac_toe(self):
    num_envs = 4

    def env_fn():
      return rl_environment.Environment("tic_tac_toe")

    envs = [env_fn for _ in range(num_envs)]
    venv = vector_env.AsyncVectorEnv(envs)

    self.assertEqual(len(venv), num_envs)
    self.assertEqual(venv.num_players, 2)

    # Initial reset
    time_steps = venv.reset()
    self.assertLen(time_steps, num_envs)

    for ts in time_steps:
      self.assertTrue(ts.step_type.first())

    # Step through the environment
    StepOutput = collections.namedtuple("StepOutput", ["action"])
    # Action 0 is valid for the first turn in Tic-Tac-Toe
    step_outputs = [StepOutput(action=0) for _ in range(num_envs)]

    time_steps, rewards, dones, unreset_time_steps = venv.step(
        step_outputs, reset_if_done=False)

    self.assertLen(time_steps, num_envs)
    self.assertLen(rewards, num_envs)
    self.assertLen(dones, num_envs)
    self.assertLen(unreset_time_steps, num_envs)

    for ts in time_steps:
      self.assertTrue(ts.step_type.mid())
      # After player 0 plays, it should be player 1's turn
      self.assertEqual(ts.observations["current_player"], 1)

    # Clean exit
    venv.close()


if __name__ == "__main__":
  absltest.main()

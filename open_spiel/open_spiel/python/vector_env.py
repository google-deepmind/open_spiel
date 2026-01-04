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
"""A vectorized RL Environment."""


class SyncVectorEnv(object):
  """A vectorized RL Environment.

  This environment is synchronized - games do not execute in parallel. Speedups
  are realized by calling models on many game states simultaneously.
  """

  def __init__(self, envs):
    if not isinstance(envs, list):
      raise ValueError(
          "Need to call this with a list of rl_environment.Environment objects")
    self.envs = envs

  def __len__(self):
    return len(self.envs)

  def observation_spec(self):
    return self.envs[0].observation_spec()

  @property
  def num_players(self):
    return self.envs[0].num_players

  def step(self, step_outputs, reset_if_done=False):
    """Apply one step.

    Args:
      step_outputs: the step outputs
      reset_if_done: if True, automatically reset the environment
          when the epsiode ends

    Returns:
      time_steps: the time steps,
      reward: the reward
      done: done flag
      unreset_time_steps: unreset time steps
    """
    time_steps = [
        self.envs[i].step([step_outputs[i].action])
        for i in range(len(self.envs))
    ]
    reward = [step.rewards for step in time_steps]
    done = [step.last() for step in time_steps]
    unreset_time_steps = time_steps  # Copy these because you may want to look
                                     # at the unreset versions to extract
                                     # information from them

    if reset_if_done:
      time_steps = self.reset(envs_to_reset=done)

    return time_steps, reward, done, unreset_time_steps

  def reset(self, envs_to_reset=None):
    if envs_to_reset is None:
      envs_to_reset = [True for _ in range(len(self.envs))]

    time_steps = [
        self.envs[i].reset()
        if envs_to_reset[i] else self.envs[i].get_time_step()
        for i in range(len(self.envs))
    ]
    return time_steps

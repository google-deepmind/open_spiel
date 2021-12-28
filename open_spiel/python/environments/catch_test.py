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

"""Tests for open_spiel.python.environment.catch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from absl.testing import absltest

from open_spiel.python import rl_environment
from open_spiel.python.environments import catch


def _select_random_legal_action(time_step):
  cur_legal_actions = time_step.observations["legal_actions"][0]
  action = random.choice(cur_legal_actions)
  return action


class CatchEnvTest(absltest.TestCase):

  def test_obs_spec(self):
    env = catch.Environment()
    obs_specs = env.observation_spec()
    self.assertLen(obs_specs, 3)
    self.assertCountEqual(obs_specs.keys(),
                          ["current_player", "info_state", "legal_actions"])

  def test_action_spec(self):
    env = catch.Environment()
    action_spec = env.action_spec()
    self.assertLen(action_spec, 4)
    self.assertCountEqual(action_spec.keys(),
                          ["dtype", "max", "min", "num_actions"])
    self.assertEqual(action_spec["num_actions"], 3)
    self.assertEqual(action_spec["dtype"], int)

  def test_action_interfaces(self):
    env = catch.Environment(height=2)
    time_step = env.reset()

    # Singleton list works
    action_list = [0]
    time_step = env.step(action_list)
    self.assertEqual(time_step.step_type, rl_environment.StepType.MID)

    # Integer works
    action_int = 0
    time_step = env.step(action_int)
    self.assertEqual(time_step.step_type, rl_environment.StepType.LAST)

  def test_many_runs(self):
    random.seed(123)
    for _ in range(20):
      height = random.randint(2, 10)
      env = catch.Environment(height=height)

      time_step = env.reset()
      self.assertEqual(time_step.step_type, rl_environment.StepType.FIRST)
      self.assertIsNone(time_step.rewards)

      action_int = _select_random_legal_action(time_step)
      time_step = env.step(action_int)
      self.assertEqual(time_step.step_type, rl_environment.StepType.MID)
      self.assertEqual(time_step.rewards, [0])

      for _ in range(1, height):
        action_int = _select_random_legal_action(time_step)
        time_step = env.step(action_int)
      self.assertEqual(time_step.step_type, rl_environment.StepType.LAST)
      self.assertIn(time_step.rewards[0], [-1, 0, 1])


if __name__ == "__main__":
  absltest.main()

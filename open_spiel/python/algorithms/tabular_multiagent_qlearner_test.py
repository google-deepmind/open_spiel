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
"""Tests for open_spiel.python.algorithms.tabular_multiagent_qlearner."""

from absl.testing import absltest
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.algorithms.tabular_qlearner import QLearner
from open_spiel.python.algorithms.tabular_multiagent_qlearner import MAQLearner, TwoPlayerNashSolver, CorrelatedEqSolver


class MultiagentQTest(absltest.TestCase):

  def test_simple_run(self):
    env = rl_environment.Environment(
        "pathfinding", grid="B.A\n...\na.b", players=2, step_reward=-1.)

    qlearner = QLearner(0, env.game.num_distinct_actions())
    nashqlearner = MAQLearner(
        1, 2, [env.game.num_distinct_actions()] * 2, TwoPlayerNashSolver())

    time_step = env.reset()
    actions = [None, None]
    step_cnt = 0

    while not time_step.last():
      actions = [qlearner.step(time_step).action,
                 nashqlearner.step(time_step, actions).action]
      time_step = env.step(actions)
      step_cnt += 1

    with self.subTest("nash_q"):
      self.assertLess(step_cnt, 500)

    qlearner = QLearner(0, env.game.num_distinct_actions())
    ceqlearner = MAQLearner(
        1, 2, [env.game.num_distinct_actions()] * 2, CorrelatedEqSolver(is_CCE=False))

    time_step = env.reset()
    actions = [None, None]
    step_cnt = 0

    while not time_step.last():
      actions = [qlearner.step(time_step).action,
                 ceqlearner.step(time_step, actions).action]
      time_step = env.step(actions)
      step_cnt += 1

    with self.subTest("ce_q"):
      self.assertLess(step_cnt, 500)

    qlearner = QLearner(0, env.game.num_distinct_actions())
    cceqlearner = MAQLearner(
        1, 2, [env.game.num_distinct_actions()] * 2, CorrelatedEqSolver(is_CCE=True))

    time_step = env.reset()
    actions = [None, None]
    step_cnt = 0

    while not time_step.last():
      actions = [qlearner.step(time_step).action,
                 cceqlearner.step(time_step, actions).action]
      time_step = env.step(actions)
      step_cnt += 1

    with self.subTest("cce_q"):
      self.assertLess(step_cnt, 500)


if __name__ == "__main__":
  absltest.main()

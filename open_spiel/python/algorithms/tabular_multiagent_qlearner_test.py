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
from open_spiel.python.algorithms.tabular_multiagent_qlearner import CorrelatedEqSolver
from open_spiel.python.algorithms.tabular_multiagent_qlearner import MultiagentQLearner
from open_spiel.python.algorithms.tabular_multiagent_qlearner import StackelbergEqSolver
from open_spiel.python.algorithms.tabular_multiagent_qlearner import TwoPlayerNashSolver
from open_spiel.python.algorithms.tabular_qlearner import QLearner
from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel

SEED = 18763511


class MultiagentQTest(absltest.TestCase):

  def test_simple_pathfinding_run(self):
    env = rl_environment.Environment(
        "pathfinding", grid="B.A\n...\na.b", players=2, step_reward=-1.)

    with self.subTest("nash_q"):
      qlearner = QLearner(0, env.game.num_distinct_actions())
      nashqlearner = MultiagentQLearner(1, 2,
                                        [env.game.num_distinct_actions()] * 2,
                                        TwoPlayerNashSolver())

      time_step = env.reset()
      actions = [None, None]
      step_cnt = 0

      while not time_step.last():
        actions = [
            qlearner.step(time_step).action,
            nashqlearner.step(time_step, actions).action
        ]
        time_step = env.step(actions)
        step_cnt += 1
      self.assertLess(step_cnt, 500)

    with self.subTest("ce_q"):
      qlearner = QLearner(0, env.game.num_distinct_actions())
      ceqlearner = MultiagentQLearner(1, 2,
                                      [env.game.num_distinct_actions()] * 2,
                                      CorrelatedEqSolver(is_cce=False))

      time_step = env.reset()
      actions = [None, None]
      step_cnt = 0

      while not time_step.last():
        actions = [
            qlearner.step(time_step).action,
            ceqlearner.step(time_step, actions).action
        ]
        time_step = env.step(actions)
        step_cnt += 1

      self.assertLess(step_cnt, 500)

    with self.subTest("cce_q"):
      qlearner = QLearner(0, env.game.num_distinct_actions())
      cceqlearner = MultiagentQLearner(1, 2,
                                       [env.game.num_distinct_actions()] * 2,
                                       CorrelatedEqSolver(is_cce=True))

      time_step = env.reset()
      actions = [None, None]
      step_cnt = 0

      while not time_step.last():
        actions = [
            qlearner.step(time_step).action,
            cceqlearner.step(time_step, actions).action
        ]
        time_step = env.step(actions)
        step_cnt += 1

      self.assertLess(step_cnt, 500)

    with self.subTest("asym_q"):
      qlearner = QLearner(0, env.game.num_distinct_actions())
      asymqlearner = MultiagentQLearner(1, 2,
                                        [env.game.num_distinct_actions()] * 2,
                                        StackelbergEqSolver())

      time_step = env.reset()
      actions = [None, None]
      step_cnt = 0

      while not time_step.last():
        actions = [
            qlearner.step(time_step).action,
            asymqlearner.step(time_step, actions).action
        ]
        time_step = env.step(actions)
        step_cnt += 1

      self.assertLess(step_cnt, 500)

  def test_rps_run(self):
    env = rl_environment.Environment("matrix_rps")
    nashqlearner0 = MultiagentQLearner(0, 2,
                                       [env.game.num_distinct_actions()] * 2,
                                       TwoPlayerNashSolver())

    nashqlearner1 = MultiagentQLearner(1, 2,
                                       [env.game.num_distinct_actions()] * 2,
                                       TwoPlayerNashSolver())

    for _ in range(1000):
      time_step = env.reset()
      actions = [None, None]
      actions = [
          nashqlearner0.step(time_step, actions).action,
          nashqlearner1.step(time_step, actions).action
      ]
      time_step = env.step(actions)
      nashqlearner0.step(time_step, actions)
      nashqlearner1.step(time_step, actions)

    with self.subTest("correct_rps_strategy"):
      time_step = env.reset()
      actions = [None, None]
      learner0_strategy, learner1_strategy = nashqlearner0.step(
          time_step, actions).probs, nashqlearner1.step(time_step,
                                                        actions).probs
      np.testing.assert_array_almost_equal(
          np.asarray([1 / 3, 1 / 3, 1 / 3]),
          learner0_strategy.reshape(-1),
          decimal=4)
      np.testing.assert_array_almost_equal(
          np.asarray([1 / 3, 1 / 3, 1 / 3]),
          learner1_strategy.reshape(-1),
          decimal=4)

    with self.subTest("correct_rps_value"):
      time_step = env.reset()
      ground_truth_values = game_payoffs_array(
          pyspiel.load_matrix_game("matrix_rps"))
      info_state = str(time_step.observations["info_state"])
      learner0_values, learner1_values = nashqlearner0._get_payoffs_array(
          info_state), nashqlearner1._get_payoffs_array(info_state)
      np.testing.assert_array_almost_equal(
          ground_truth_values, learner0_values, decimal=4)
      np.testing.assert_array_almost_equal(
          ground_truth_values, learner1_values, decimal=4)


if __name__ == "__main__":
  np.random.seed(SEED)
  absltest.main()

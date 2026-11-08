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
from open_spiel.python import rl_tools
from open_spiel.python.algorithms.tabular_qlearner import QLearner
from open_spiel.python.algorithms.wolf_phc import WoLFPHC
import pyspiel

SEED = 18763511


class WoLFTest(absltest.TestCase):

  def test_terminal_transition_updates_policy(self):
    game = pyspiel.create_matrix_game([[1, 1], [1, 1]], [[0, 0], [0, 0]])
    env = rl_environment.Environment(game=game)
    agent = WoLFPHC(
        0,
        game.num_distinct_actions(),
        step_size=rl_tools.ConstantSchedule(1.0),
        epsilon_schedule=rl_tools.ConstantSchedule(0.0),
        delta_w=rl_tools.ConstantSchedule(0.05),
        delta_l=rl_tools.ConstantSchedule(0.1),
    )

    first = agent.step(env.reset())
    np.testing.assert_allclose(first.probs, [0.5, 0.5])
    terminal = env.step([first.action, 0])
    self.assertTrue(terminal.last())
    agent.step(terminal)

    updated = agent.step(env.reset(), is_evaluation=True)
    self.assertAlmostEqual(updated.probs[first.action], 0.6)
    self.assertAlmostEqual(updated.probs[1 - first.action], 0.4)

  def test_simple_pathfinding_run(self):
    env = rl_environment.Environment(
        "pathfinding", grid="B.A\n...\na.b", players=2, step_reward=-1.)

    with self.subTest("wolf_phc"):
      qlearner = QLearner(0, env.game.num_distinct_actions())
      wolflearner = WoLFPHC(1, env.game.num_distinct_actions())
      time_step = env.reset()
      step_cnt = 0

      while not time_step.last():
        actions = [
            qlearner.step(time_step).action,
            wolflearner.step(time_step).action
        ]
        time_step = env.step(actions)
        step_cnt += 1

      self.assertLess(step_cnt, 500)

  def test_rps_run(self):
    np.random.seed(SEED)
    env = rl_environment.Environment("matrix_rps")
    wolf0 = WoLFPHC(0, env.game.num_distinct_actions())
    wolf1 = WoLFPHC(1, env.game.num_distinct_actions())

    for _ in range(1000):
      time_step = env.reset()
      actions = [wolf0.step(time_step).action, wolf1.step(time_step).action]
      time_step = env.step(actions)
      wolf0.step(time_step)
      wolf1.step(time_step)

    with self.subTest("near_uniform_rps_strategy"):
      time_step = env.reset()
      for learner in (wolf0, wolf1):
        strategy = learner.step(time_step, is_evaluation=True).probs
        # A finite self-play run keeps the policy near the equilibrium.
        np.testing.assert_allclose(strategy, [1 / 3] * 3, atol=0.05)


if __name__ == "__main__":
  np.random.seed(SEED)
  absltest.main()

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
"""Python example of multiagent Nash Q-learners."""

import enum
import logging
from absl import app

from open_spiel.python import rl_environment
from open_spiel.python.algorithms.tabular_multiagent_qlearner import MAQLearner
from open_spiel.python.algorithms.tabular_multiagent_qlearner import TwoPlayerNashSolver
from open_spiel.python.algorithms.tabular_qlearner import QLearner


class Action(enum.IntEnum):
  STAY = 0
  LEFT = 1
  UP = 2
  RIGHT = 3
  DOWN = 4


def print_iteration(actions, state):
  """Print actions and state."""
  logging.info("Action taken by agent 0: %s", Action(actions[0]).name)
  logging.info("Action taken by agent 1: %s", Action(actions[1]).name)
  logging.info("Board state:\n %s", state)
  logging.info("-" * 80)


def marl_path_finding_example(_):
  """Example usage of multiagent Nash Q-learner.

  Based on https://www.jmlr.org/papers/volume4/hu03a/hu03a.pdf
  """

  logging.info("Creating the Grid Game")
  env = rl_environment.Environment(
      "pathfinding", grid="B.A\n...\na.b", players=2, step_reward=-1.)

  qlearner = QLearner(0, env.game.num_distinct_actions())
  nashqlearner = MAQLearner(1, 2, [env.game.num_distinct_actions()] * 2,
                            TwoPlayerNashSolver())

  time_step = env.reset()
  actions = [None, None]

  while not time_step.last():
    actions = [
        qlearner.step(time_step).action,
        nashqlearner.step(time_step, actions).action
    ]
    time_step = env.step(actions)
    print_iteration(actions, env.get_state)


if __name__ == "__main__":
  app.run(marl_path_finding_example)

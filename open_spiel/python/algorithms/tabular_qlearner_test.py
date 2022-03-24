"""Tests for open_spiel.python.algorithms.tabular_qlearner"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner
import pyspiel

# A simple two-action game encoded as an EFG game. Going left gets -1, going
# right gets a +1.
SIMPLE_EFG_DATA = """
  EFG 2 R "Simple single-agent problem" { "Player 1" } ""
  p "ROOT" 1 1 "ROOT" { "L" "R" } 0
    t "L" 1 "Outcome L" { -1.0 }
    t "R" 2 "Outcome R" { 1.0 }
"""


class QlearnerTest(absltest.TestCase):

    def test_simple_game(self):
        game = pyspiel.load_efg_game(SIMPLE_EFG_DATA)
        env = rl_environment.Environment(game=game)

        agent = tabular_qlearner.QLearner(0, game.num_distinct_actions())
        total_reward = 0

        for _ in range(100):
            time_step = env.reset()
            while not time_step.last():
                agent_output = agent.step(time_step)
                time_step = env.step([agent_output.action])
                total_reward += time_step.rewards[0]
            agent.step(time_step)
        self.assertGreaterEqual(total_reward, 75)


if __name__ == "__main__":
    absltest.main()

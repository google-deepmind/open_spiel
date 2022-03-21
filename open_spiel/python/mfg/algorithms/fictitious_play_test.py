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
"""Tests for fictitious play."""

from absl.testing import absltest

from open_spiel.python import policy
from open_spiel.python import rl_agent_policy
from open_spiel.python import rl_environment
from open_spiel.python.jax import dqn
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import fictitious_play
from open_spiel.python.mfg.algorithms import greedy_policy
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import crowd_modelling
import pyspiel


class FictitiousPlayTest(absltest.TestCase):

  def test_fp_python_game(self):
    """Checks if fictitious play works."""
    game = crowd_modelling.MFGCrowdModellingGame()
    fp = fictitious_play.FictitiousPlay(game)
    for _ in range(10):
      fp.iteration()
    fp_policy = fp.get_policy()
    nash_conv_fp = nash_conv.NashConv(game, fp_policy)

    self.assertAlmostEqual(nash_conv_fp.nash_conv(), 0.9908032626911343)

  def test_dqn_fp_python_game(self):
    """Checks if fictitious play with DQN-based value function works."""
    game = crowd_modelling.MFGCrowdModellingGame()
    dfp = fictitious_play.FictitiousPlay(game)

    uniform_policy = policy.UniformRandomPolicy(game)
    dist = distribution.DistributionPolicy(game, uniform_policy)
    envs = [
        rl_environment.Environment(
            game, mfg_distribution=dist, mfg_population=p)
        for p in range(game.num_players())
    ]
    dqn_agent = dqn.DQN(
        0,
        state_representation_size=envs[0].observation_spec()["info_state"][0],
        num_actions=envs[0].action_spec()["num_actions"],
        hidden_layers_sizes=[256, 128, 64],
        replay_buffer_capacity=100,
        batch_size=5,
        epsilon_start=0.02,
        epsilon_end=0.01)

    br_policy = rl_agent_policy.RLAgentPolicy(
        game, dqn_agent, 0, use_observation=True)
    for _ in range(10):
      dfp.iteration(br_policy=br_policy)

    dfp_policy = dfp.get_policy()
    nash_conv_dfp = nash_conv.NashConv(game, dfp_policy)

    self.assertAlmostEqual(nash_conv_dfp.nash_conv(), 1.0558451955622807)

  def test_average(self):
    """Test the average of policies.

    Here we test that the average of values is the value of the average policy.
    """
    game = crowd_modelling.MFGCrowdModellingGame()
    uniform_policy = policy.UniformRandomPolicy(game)
    mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
    br_value = best_response_value.BestResponse(
        game, mfg_dist, value.TabularValueFunction(game))
    py_value = policy_value.PolicyValue(game, mfg_dist, uniform_policy,
                                        value.TabularValueFunction(game))
    greedy_pi = greedy_policy.GreedyPolicy(game, None, br_value)
    greedy_pi = greedy_pi.to_tabular()
    merged_pi = fictitious_play.MergedPolicy(
        game, list(range(game.num_players())), [uniform_policy, greedy_pi],
        [mfg_dist, distribution.DistributionPolicy(game, greedy_pi)],
        [0.5, 0.5])
    merged_pi_value = policy_value.PolicyValue(game, mfg_dist, merged_pi,
                                               value.TabularValueFunction(game))

    self.assertAlmostEqual(
        merged_pi_value(game.new_initial_state()),
        (br_value(game.new_initial_state()) +
         py_value(game.new_initial_state())) / 2)

  def test_fp_cpp_game(self):
    """Checks if fictitious play works."""
    game = pyspiel.load_game("mfg_crowd_modelling")
    fp = fictitious_play.FictitiousPlay(game)
    for _ in range(10):
      fp.iteration()
    fp_policy = fp.get_policy()
    nash_conv_fp = nash_conv.NashConv(game, fp_policy)

    self.assertAlmostEqual(nash_conv_fp.nash_conv(), 0.9908032626911343)

  def test_dqn_fp_cpp_game(self):
    """Checks if fictitious play with DQN-based value function works."""
    game = pyspiel.load_game("mfg_crowd_modelling")
    dfp = fictitious_play.FictitiousPlay(game)

    uniform_policy = policy.UniformRandomPolicy(game)
    dist = distribution.DistributionPolicy(game, uniform_policy)
    envs = [
        rl_environment.Environment(
            game, mfg_distribution=dist, mfg_population=p)
        for p in range(game.num_players())
    ]
    dqn_agent = dqn.DQN(
        0,
        state_representation_size=envs[0].observation_spec()["info_state"][0],
        num_actions=envs[0].action_spec()["num_actions"],
        hidden_layers_sizes=[256, 128, 64],
        replay_buffer_capacity=100,
        batch_size=5,
        epsilon_start=0.02,
        epsilon_end=0.01)

    br_policy = rl_agent_policy.RLAgentPolicy(
        game, dqn_agent, 0, use_observation=True)
    for _ in range(10):
      dfp.iteration(br_policy=br_policy)

    dfp_policy = dfp.get_policy()
    nash_conv_dfp = nash_conv.NashConv(game, dfp_policy)

    self.assertAlmostEqual(nash_conv_dfp.nash_conv(), 1.0558451955622807)


if __name__ == "__main__":
  absltest.main()

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

"""Tests for softmax_policy."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python import policy
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.algorithms import softmax_policy
from open_spiel.python.mfg.games import crowd_modelling  # pylint: disable=unused-import
import pyspiel


class SoftmaxPolicyTest(parameterized.TestCase):

  @parameterized.named_parameters(('python', 'python_mfg_crowd_modelling'),
                                  ('cpp', 'mfg_crowd_modelling'))
  def test_softmax(self, name):
    """Check if the softmax policy works as expected.

    The test checks that:
    - uniform prior policy gives the same results than no prior.
    - very high temperature gives almost a uniform policy.
    - very low temperature gives almost a deterministic policy for the best
    action.

    Args:
      name: Name of the game.
    """

    game = pyspiel.load_game(name)
    uniform_policy = policy.UniformRandomPolicy(game)
    dist = distribution.DistributionPolicy(game, uniform_policy)
    br_value = best_response_value.BestResponse(
        game, dist, value.TabularValueFunction(game))
    br_init_val = br_value(game.new_initial_state())

    # uniform prior policy gives the same results than no prior.
    softmax_pi_uniform_prior = softmax_policy.SoftmaxPolicy(
        game, None, 1.0, br_value, uniform_policy).to_tabular()
    softmax_pi_uniform_prior_value = policy_value.PolicyValue(
        game, dist, softmax_pi_uniform_prior, value.TabularValueFunction(game))
    softmax_pi_uniform_prior_init_val = softmax_pi_uniform_prior_value(
        game.new_initial_state())
    softmax_pi_no_prior = softmax_policy.SoftmaxPolicy(game, None, 1.0,
                                                       br_value, None)
    softmax_pi_no_prior_value = policy_value.PolicyValue(
        game, dist, softmax_pi_no_prior, value.TabularValueFunction(game))
    softmax_pi_no_prior_init_val = softmax_pi_no_prior_value(
        game.new_initial_state())

    self.assertAlmostEqual(softmax_pi_uniform_prior_init_val,
                           softmax_pi_no_prior_init_val)

    # very high temperature gives almost a uniform policy.
    uniform_policy = uniform_policy.to_tabular()
    uniform_value = policy_value.PolicyValue(game, dist, uniform_policy,
                                             value.TabularValueFunction(game))
    uniform_init_val = uniform_value(game.new_initial_state())

    softmax_pi_no_prior = softmax_policy.SoftmaxPolicy(game, None, 100000000,
                                                       br_value, None)
    softmax_pi_no_prior_value = policy_value.PolicyValue(
        game, dist, softmax_pi_no_prior, value.TabularValueFunction(game))
    softmax_pi_no_prior_init_val = softmax_pi_no_prior_value(
        game.new_initial_state())

    self.assertAlmostEqual(uniform_init_val, softmax_pi_no_prior_init_val)

    # very low temperature gives almost a best response policy.
    softmax_pi_no_prior = softmax_policy.SoftmaxPolicy(game, None, 0.0001,
                                                       br_value, None)
    softmax_pi_no_prior_value = policy_value.PolicyValue(
        game, dist, softmax_pi_no_prior, value.TabularValueFunction(game))
    softmax_pi_no_prior_init_val = softmax_pi_no_prior_value(
        game.new_initial_state())

    self.assertAlmostEqual(br_init_val, softmax_pi_no_prior_init_val)


if __name__ == '__main__':
  absltest.main()

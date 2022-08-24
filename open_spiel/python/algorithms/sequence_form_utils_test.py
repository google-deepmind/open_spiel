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

"""Tests for open_spiel.python.sequence_form_utils.py."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import sequence_form_utils
from open_spiel.python.algorithms.expected_game_score import policy_value
import pyspiel

_KUHN_GAME = pyspiel.load_game('kuhn_poker')
_LEDUC_GAME = pyspiel.load_game('leduc_poker')


class SequenceFormTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'game': _KUHN_GAME,
          'cfr_iters': 100
      },
      {
          'game': _LEDUC_GAME,
          'cfr_iters': 10
      },
  )
  def test_sequence_to_policy(self, game, cfr_iters):

    cfr_solver = cfr.CFRSolver(game)

    for _ in range(cfr_iters):
      cfr_solver.evaluate_and_update_policy()

    (_, infoset_actions_to_seq, infoset_action_maps, _, _,
     _) = sequence_form_utils.construct_vars(game)

    policies = cfr_solver.average_policy()
    sequences = sequence_form_utils.policy_to_sequence(game, policies,
                                                       infoset_actions_to_seq)
    converted_policies = sequence_form_utils.sequence_to_policy(
        sequences, game, infoset_actions_to_seq, infoset_action_maps)
    np.testing.assert_allclose(
        policies.action_probability_array,
        converted_policies.action_probability_array,
        rtol=1e-10)

  @parameterized.parameters(
      {
          'game': _KUHN_GAME,
          'cfr_iters': 100
      },
      {
          'game': _LEDUC_GAME,
          'cfr_iters': 10
      },
  )
  def test_sequence_payoff(self, game, cfr_iters):
    (_, infoset_actions_to_seq, _, _, payoff_mat,
     _) = sequence_form_utils.construct_vars(game)

    uniform_policies = policy.TabularPolicy(game)
    uniform_value = policy_value(game.new_initial_state(),
                                 [uniform_policies, uniform_policies])
    sequences = sequence_form_utils.policy_to_sequence(game, uniform_policies,
                                                       infoset_actions_to_seq)
    np.testing.assert_allclose(
        uniform_value[0],
        -sequences[0].T @ payoff_mat @ sequences[1],
        rtol=1e-10)

    # use cfr iterations to construct new policy
    cfr_solver = cfr.CFRSolver(game)
    for _ in range(cfr_iters):
      cfr_solver.evaluate_and_update_policy()

    policies = cfr_solver.average_policy()
    cfr_value = policy_value(game.new_initial_state(), [policies, policies])
    sequences = sequence_form_utils.policy_to_sequence(game, policies,
                                                       infoset_actions_to_seq)
    np.testing.assert_allclose(
        cfr_value[0], -sequences[0].T @ payoff_mat @ sequences[1], rtol=1e-10)


if __name__ == '__main__':
  absltest.main()

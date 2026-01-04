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

"""Tests several sequence form utilities."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import expected_game_score as egs
from open_spiel.python.algorithms import sequence_form_utils
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
    uniform_value = egs.policy_value(
        game.new_initial_state(), [uniform_policies, uniform_policies]
    )
    sequences = sequence_form_utils.policy_to_sequence(game, uniform_policies,
                                                       infoset_actions_to_seq)
    min_mat = -payoff_mat[0]
    np.testing.assert_allclose(
        uniform_value[0],
        -sequences[0].T @ min_mat @ sequences[1],
        rtol=1e-10)

    # use cfr iterations to construct new policy
    cfr_solver = cfr.CFRSolver(game)
    for _ in range(cfr_iters):
      cfr_solver.evaluate_and_update_policy()

    policies = cfr_solver.average_policy()
    cfr_value = egs.policy_value(game.new_initial_state(), [policies, policies])
    sequences = sequence_form_utils.policy_to_sequence(game, policies,
                                                       infoset_actions_to_seq)
    np.testing.assert_allclose(
        cfr_value[0], -sequences[0].T @ min_mat @ sequences[1], rtol=1e-10)

  @parameterized.parameters(
      {
          'game': _KUHN_GAME,
          'seed': 12345
      },
      {
          'game': _LEDUC_GAME,
          'seed': 12345
      },
  )
  def test_sequence_tangent_projection(self, game, seed, step_size=1e-2):
    (_, infoset_actions_to_seq, infoset_action_maps, infoset_parent_map, _,
     _) = sequence_form_utils.construct_vars(game)

    uniform_policies = policy.TabularPolicy(game)
    sequences = sequence_form_utils.policy_to_sequence(game, uniform_policies,
                                                       infoset_actions_to_seq)

    constraints = sequence_form_utils.construct_constraint_vars(
        infoset_parent_map, infoset_actions_to_seq, infoset_action_maps)

    rnd = np.random.RandomState(seed)

    con_errs = []
    for p, (con_mat, b) in constraints.items():
      seq_p = sequences[p]
      # generate random sequence-form "gradient" direction
      grad = rnd.randn(len(seq_p))
      grad = grad / np.linalg.norm(grad)
      # construct tangent projection from sequence-form constraints
      pinv_con_mat = np.linalg.pinv(con_mat)
      proj = np.eye(con_mat.shape[1]) - pinv_con_mat.dot(con_mat)
      # project gradient onto tangent space
      grad_proj = proj.dot(grad)
      # take gradient step (should remain on treeplex w/ small enough step_size)
      seq_p_new = seq_p + step_size * grad_proj
      # measure constraint violation error after gradient step (should be ~0)
      con_err = np.linalg.norm(con_mat.dot(seq_p_new) - b)
      con_errs.append(con_err)

    np.testing.assert_allclose(
        con_errs,
        np.zeros(len(con_errs)),
        atol=1e-10)


if __name__ == '__main__':
  absltest.main()

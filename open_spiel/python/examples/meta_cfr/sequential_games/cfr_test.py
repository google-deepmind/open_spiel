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

"""Tests counterfactual regret minimization."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.examples.meta_cfr.sequential_games import cfr
from open_spiel.python.examples.meta_cfr.sequential_games import game_tree_utils as trees
from open_spiel.python.examples.meta_cfr.sequential_games import openspiel_api


def _uniform_policy(size):
  if size > 0:
    return [1./size]*size
  return []


class CfrTest(parameterized.TestCase):

  @parameterized.named_parameters(('kuhn_poker_test', 'kuhn_poker'),
                                  ('leduc_poker_test', 'leduc_poker'))
  def test_zero_policy_is_uniform(self, game):
    config = {'players': 2}
    cfr_game_tree = trees.build_game_tree(
        openspiel_api.WorldState(
            game_name=game, config=config, perturbation=False))
    cfr.compute_cfr_values(cfr_game_tree, 1)
    infostates_p1 = list(cfr_game_tree.all_infostates_map[1].values())
    infostates_p2 = list(cfr_game_tree.all_infostates_map[2].values())
    with self.subTest('player_1_initial_policy'):
      for i in range(len(infostates_p1)):
        self.assertListEqual(
            list(infostates_p1[i].policy.values()),
            _uniform_policy(len(infostates_p1[i].policy.values())))
    with self.subTest('player_2_initial_policy'):
      for i in range(len(infostates_p2)):
        self.assertListEqual(
            list(infostates_p2[i].policy.values()),
            _uniform_policy(len(infostates_p2[i].policy.values())))

  def test_cfr_leduc_poker(self):
    config = {'players': 2}
    exploitability_error = 0.2
    cfr_game_tree = trees.build_game_tree(
        openspiel_api.WorldState(
            game_name='leduc_poker', config=config, perturbation=False))
    best_response_value_p1, best_response_value_p2 = cfr.compute_cfr_values(
        cfr_game_tree, 20)
    last_best_response_value_player_1 = best_response_value_p1[-1]
    last_best_response_value_player_2 = best_response_value_p2[-1]
    exploitability = (last_best_response_value_player_1 +
                      last_best_response_value_player_2) / 2
    # Exploitability values are computed using OpenSpiel cfr
    self.assertLessEqual(exploitability, 0.59 + exploitability_error)

  def test_cfr_kuhn_poker(self):
    config = {'players': 2}
    exploitability_error = 0.2
    cfr_game_tree = trees.build_game_tree(
        openspiel_api.WorldState(
            game_name='kuhn_poker', config=config, perturbation=False))
    best_response_value_p1, best_response_value_p2 = cfr.compute_cfr_values(
        cfr_game_tree, 20)
    last_best_response_value_player_1 = best_response_value_p1[-1]
    last_best_response_value_player_2 = best_response_value_p2[-1]
    exploitability = (last_best_response_value_player_1 +
                      last_best_response_value_player_2) / 2
    # Exploitability values are computed using OpenSpiel cfr
    self.assertLessEqual(exploitability, 0.06 + exploitability_error)


if __name__ == '__main__':
  absltest.main()

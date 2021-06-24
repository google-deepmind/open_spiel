# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a dummy trajectory and compute the distribution of a policy."""
# pylint: disable=unused-import
from typing import Sequence

from absl import app
from absl import flags
import numpy as np

from open_spiel.python import policy
from open_spiel.python.mfg import games
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import fictitious_play
from open_spiel.python.mfg.algorithms import greedy_policy
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.algorithms import policy_value
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_string('game', 'mfg_crowd_modelling_2d', 'Game to use.')

# Maps game names to the settings to use.
# Empty settings will be used for games that are not in this mapping.
GAME_SETTINGS = {
    'mfg_crowd_modelling_2d': {
        'only_distribution_reward': True,
        'forbidden_states': '[0|0;0|1]',
        'initial_distribution': '[0|2;0|3]',
        'initial_distribution_value': '[0.5;0.5]',
    }
}


def main(argv: Sequence[str]) -> None:
  # TODO(perolat): move to an example directory.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  mfg_game = pyspiel.load_game(FLAGS.game, GAME_SETTINGS.get(FLAGS.game, {}))
  mfg_state = mfg_game.new_initial_state()
  print('Playing a single arbitrary trajectory')
  while not mfg_state.is_terminal():
    print('State obs string:', mfg_state.observation_string(0))
    if mfg_state.current_player() == pyspiel.PlayerId.CHANCE:
      action_list, prob_list = zip(*mfg_state.chance_outcomes())
      action = np.random.choice(action_list, p=prob_list)
      mfg_state.apply_action(action)
    elif mfg_state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
      dist_to_register = mfg_state.distribution_support()
      n_states = len(dist_to_register)
      dist = [1.0 / n_states for _ in range(n_states)]
      mfg_state.update_distribution(dist)
    else:
      legal_list = mfg_state.legal_actions()
      action = np.random.choice(legal_list)
      mfg_state.apply_action(action)

  print('compute nashconv')
  uniform_policy = policy.UniformRandomPolicy(mfg_game)
  nash_conv_fp = nash_conv.NashConv(mfg_game, uniform_policy)
  print('Nashconv:', nash_conv_fp.nash_conv())

  print('compute distribution')
  mfg_dist = distribution.DistributionPolicy(mfg_game, uniform_policy)
  br_value = best_response_value.BestResponse(mfg_game, mfg_dist)
  py_value = policy_value.PolicyValue(mfg_game, mfg_dist, uniform_policy)
  print(
      'Value of a best response policy to a uniform policy '
      '(computed with best_response_value)',
      br_value(mfg_game.new_initial_state()))
  print('Value of the uniform policy:', py_value(mfg_game.new_initial_state()))
  greedy_pi = greedy_policy.GreedyPolicy(mfg_game, None, br_value)
  greedy_pi = greedy_pi.to_tabular()
  pybr_value = policy_value.PolicyValue(mfg_game, mfg_dist, greedy_pi)
  print(
      'Value of a best response policy to a uniform policy (computed at the '
      'value of the greedy policy of the best response value)',
      pybr_value(mfg_game.new_initial_state()))
  print('merge')
  merged_pi = fictitious_play.MergedPolicy(
      mfg_game, list(range(mfg_game.num_players())),
      [uniform_policy, greedy_pi],
      [mfg_dist, distribution.DistributionPolicy(mfg_game, greedy_pi)],
      [0.5, 0.5])

  merged_pi_value = policy_value.PolicyValue(mfg_game, mfg_dist, merged_pi)
  print(br_value(mfg_game.new_initial_state()))
  print(py_value(mfg_game.new_initial_state()))
  print(merged_pi_value(mfg_game.new_initial_state()))
  print((br_value(mfg_game.new_initial_state()) +
         py_value(mfg_game.new_initial_state())) / 2)
  print('fp')
  fp = fictitious_play.FictitiousPlay(mfg_game)
  for j in range(100):
    print('Iteration', j, 'of fictitious play')
    fp.iteration()
    fp_policy = fp.get_policy()
    nash_conv_fp = nash_conv.NashConv(mfg_game, fp_policy)
    print('Nashconv of the current FP policy', nash_conv_fp.nash_conv())


if __name__ == '__main__':
  app.run(main)

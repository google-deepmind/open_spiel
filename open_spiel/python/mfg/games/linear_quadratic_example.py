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
from open_spiel.python.mfg.algorithms import mirror_descent
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import linear_quadratic
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_string('game', 'mean_field_lin_quad', 'Game to use.')
flags.DEFINE_integer('size', 10, 'Number of states.')
flags.DEFINE_integer('horizon', 5, 'Horizon size.')
flags.DEFINE_float('dt', 1.0, 'Delta t.')
flags.DEFINE_integer('n_actions_per_side', 3,
                     'Number actions per side (Total num actions = 2*x+1).')
flags.DEFINE_float('volatility', 1.0, 'Action noise.')
flags.DEFINE_float('learning_rate', 0.01, 'OMD learning rate.')


def get_l1_distribution_dist(mu1, mu2):
  mu1d = mu1.distribution
  mu2d = mu2.distribution
  states = set(list(mu1d.keys()) + list(mu2d.keys()))
  return sum([abs(mu1d.get(a, 0.0) - mu2d.get(a, 0.0)) for a in states
             ]) * FLAGS.dt / FLAGS.horizon


class LinearPolicy(policy.Policy):
  """Project values on the policy simplex."""

  def __init__(self, game, player_ids):  # pylint:disable=useless-super-delegation
    """Initializes the projected policy.

    Args:
      game: The game to analyze.
      player_ids: list of player ids for which this policy applies; each should
        be in the range 0..game.num_players()-1.
    """
    super(LinearPolicy, self).__init__(game, player_ids)

  def action_probabilities(self, state, player_id=None):
    mu_bar_t = state.distribution_average()
    x_t = state.x
    q = state.cross_q
    n_actions_per_side = state.n_actions_per_side
    lin_action = (q + state.eta_t()) * (mu_bar_t - x_t)
    action = n_actions_per_side + min(
        n_actions_per_side, max(round(lin_action), -n_actions_per_side))
    action_prob = [(a, 0.0) for a in state.legal_actions()]
    action_prob[action] = (action, 1.0)
    return dict(action_prob)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  mfg_game = pyspiel.load_game(
      FLAGS.game, {
          'dt': FLAGS.dt,
          'size': FLAGS.size,
          'horizon': FLAGS.horizon,
          'n_actions_per_side': FLAGS.n_actions_per_side,
          'volatility': FLAGS.volatility
      })

  uniform_policy = policy.UniformRandomPolicy(mfg_game)
  nash_conv_fp = nash_conv.NashConv(mfg_game, uniform_policy)
  print('Uniform Policy Nashconv:', nash_conv_fp.nash_conv())

  # Optimal control in the continuous setting.
  theoretical_control = LinearPolicy(mfg_game,
                                     list(range(mfg_game.num_players())))
  theoretical_distribution = distribution.DistributionPolicy(
      mfg_game, theoretical_control)
  discretized_optimal_value = policy_value.PolicyValue(
      mfg_game, theoretical_distribution,
      theoretical_control).eval_state(mfg_game.new_initial_state())

  th_expl = nash_conv.NashConv(mfg_game, theoretical_control).nash_conv()
  print('Theoretical policy NashConv : {}'.format(th_expl))
  print('Theoretical policy Value : {}'.format(discretized_optimal_value))

  fp = fictitious_play.FictitiousPlay(mfg_game)
  md = mirror_descent.MirrorDescent(mfg_game)
  for j in range(1000):
    print('\n\nIteration', j, '\n')
    fp.iteration()
    fp_policy = fp.get_policy()
    nash_conv_fp = nash_conv.NashConv(mfg_game, fp_policy)
    print('Nashconv of the current FP policy', nash_conv_fp.nash_conv())
    fp_current_distribution = distribution.DistributionPolicy(
        mfg_game, fp.get_policy())
    fp_l1_dist = get_l1_distribution_dist(fp_current_distribution,
                                          theoretical_distribution)
    print(
        'L1 distance between FP and theoretical policy : {}'.format(fp_l1_dist))
    md.iteration()
    md_policy = md.get_policy()
    nash_conv_md = nash_conv.NashConv(mfg_game, md_policy)

    print('')

    print('Nashconv of the current MD policy', nash_conv_md.nash_conv())
    md_current_distribution = md._distribution  # pylint:disable=protected-access
    md_l1_dist = get_l1_distribution_dist(md_current_distribution,
                                          theoretical_distribution)
    print('L1 distance between OMD and theoretical policy : {}'.format(
        md_l1_dist))


if __name__ == '__main__':
  app.run(main)

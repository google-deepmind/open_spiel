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
"""Fictitious play on an MFG game."""
import os
from typing import Sequence

from absl import flags

from open_spiel.python.mfg import utils
from open_spiel.python.mfg.algorithms import fictitious_play
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.games import factory
from open_spiel.python.utils import app
from open_spiel.python.utils import metrics

FLAGS = flags.FLAGS

flags.DEFINE_string('game_name', 'mfg_crowd_modelling_2d', 'Name of the game.')
flags.DEFINE_string(
    'setting', None,
    'Name of the game settings. If None, the game name will be used.')
flags.DEFINE_integer('num_iterations', 100,
                     'Number of fictitious play iterations.')

flags.DEFINE_float('learning_rate', None,
                   'Learning rate. If not, it will be set to 1/iteration.')
_LOGDIR = flags.DEFINE_string(
    'logdir', None,
    'Logging dir to use for TF summary files. If None, the metrics will only '
    'be logged to stderr.')
_LOG_DISTRIBUTION = flags.DEFINE_bool('log_distribution', False,
                                      'Enables logging of the distribution.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  game = factory.create_game_with_setting(FLAGS.game_name, FLAGS.setting)

  # Metrics writer will also log the metrics to stderr.
  just_logging = _LOGDIR.value is None
  writer = metrics.create_default_writer(
      logdir=_LOGDIR.value, just_logging=just_logging)

  # Save the parameters.
  learning_rate = FLAGS.learning_rate
  writer.write_hparams({'learning_rate': learning_rate})

  fp = fictitious_play.FictitiousPlay(game)

  for it in range(FLAGS.num_iterations):
    fp.iteration(learning_rate=learning_rate)
    fp_policy = fp.get_policy()
    nash_conv_fp = nash_conv.NashConv(game, fp_policy)
    exploitability = nash_conv_fp.nash_conv()
    writer.write_scalars(it, {'exploitability': exploitability})
    if _LOG_DISTRIBUTION.value and not just_logging:
      filename = os.path.join(_LOGDIR.value, f'distribution_{it}.pkl')
      utils.save_parametric_distribution(nash_conv_fp.distribution, filename)

  writer.flush()


if __name__ == '__main__':
  app.run(main)

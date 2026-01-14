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

"""Simple benchmark for MFG algorithms and environments."""

import itertools
import time
from typing import Sequence

from absl import app
from absl import flags

from open_spiel.python.mfg import games  # pylint: disable=unused-import
from open_spiel.python.mfg.algorithms import fictitious_play
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_list('games',
                  ['python_mfg_crowd_modelling', 'mfg_crowd_modelling'],
                  'List of games to benchmark.')
flags.DEFINE_list(
    'parameters', ['size:10;100', 'horizon:10;100'],
    'List of parameters to sweep on (see default flag value for '
    'syntax).')


def convert_param_spec(param_spec):
  """Converts 'size:10;200' into ('size', [10, 200])."""
  split = param_spec.split(':', 2)
  return split[0], [int(v) for v in split[1].split(';')]


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  param_names, param_values = zip(
      *[convert_param_spec(spec) for spec in FLAGS.parameters])
  header = (['game_name'] + list(param_names) +
            ['fictitious_play_iteration_time'])
  timing_results = []
  for game_name in FLAGS.games:
    for param_tuple in itertools.product(*param_values):
      result_line = [game_name] + [str(p) for p in param_tuple]
      print('Computing timings for:', ' '.join(result_line))
      param_dict = dict(zip(param_names, param_tuple))
      game = pyspiel.load_game(game_name, param_dict)
      t0 = time.time()
      fp = fictitious_play.FictitiousPlay(game)
      fp.iteration()
      elapsed = time.time() - t0
      result_line.append(f'{elapsed:.4f}s')
      print(' '.join(result_line))
      timing_results.append(result_line)

  print('\nRESULTS:')
  print(' '.join(header))
  for line in timing_results:
    print(' '.join([str(v) for v in line]))


if __name__ == '__main__':
  app.run(main)

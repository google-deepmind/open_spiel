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

"""Tests for open_spiel.python.mmd_dilated.py."""
import copy

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from open_spiel.python.algorithms import mmd_dilated
import pyspiel


_DATA = [
    {
        'game':
            pyspiel.load_game('kuhn_poker'),
        'inverse_alpha':
            10,
        'gambit_qre_sol': [
            np.array([
                1., 0.75364232, 0.64695966, 0.10668266, 0.24635768, 0.70309809,
                0.25609184, 0.44700625, 0.29690191, 0.47546799, 0.01290797,
                0.46256001, 0.52453201
            ]),
            np.array([
                1., 0.63415944, 0.36584056, 0.41154828, 0.58845172, 0.28438486,
                0.71561514, 0.0620185, 0.9379815, 0.65005434, 0.34994566,
                0.79722767, 0.20277233
            ])
        ]
    },
    {
        'game':
            pyspiel.load_game('dark_hex(board_size=2,gameversion=adh)'),
        'inverse_alpha':
            2,
        'gambit_qre_sol': [
            np.array([
                1., 0.1997415, 0.0630504, 0.0320848, 0.0309656, 0.0320848,
                0.0309656, 0.0696913, 0.0669998, 0.0334999, 0.0334999,
                0.0334999, 0.0334999, 0.0377519, 0.0252985, 0.0252985,
                0.0252985, 0.0347624, 0.0347624, 0.0349289, 0.0349289, 0.0273,
                0.0273, 0.0396998, 0.0273, 0.3002587, 0.0832425, 0.0414444,
                0.0417981, 0.0414444, 0.0417981, 0.0983483, 0.1186679,
                0.0423458, 0.0408967, 0.0423458, 0.0408967, 0.0397914,
                0.0397914, 0.0585569, 0.0397914, 0.047948, 0.047948, 0.0707199,
                0.047948, 0.3002587, 0.1186679, 0.0707199, 0.047948, 0.047948,
                0.047948, 0.0983483, 0.0832425, 0.0408967, 0.0408967, 0.0423458,
                0.0585569, 0.0397914, 0.0397914, 0.0397914, 0.0423458,
                0.0417981, 0.0417981, 0.0414444, 0.0414444, 0.1997415,
                0.0669998, 0.0396998, 0.0273, 0.0273, 0.0273, 0.0696913,
                0.0630504, 0.0309656, 0.0309656, 0.0320848, 0.0334999,
                0.0334999, 0.0334999, 0.0349289, 0.0349289, 0.0347624,
                0.0347624, 0.0320848, 0.0334999, 0.0252985, 0.0252985,
                0.0377519, 0.0252985
            ]),
            np.array([
                1., 0.22738648, 0.07434555, 0.0790954, 0.03965962, 0.03943577,
                0.07394554, 0.03468592, 0.03925961, 0.03965962, 0.03468592,
                0.27261352, 0.10172918, 0.06014879, 0.04158039, 0.08865251,
                0.08223183, 0.04230736, 0.03992446, 0.04171322, 0.0405186,
                0.27261352, 0.08223183, 0.0405186, 0.04171322, 0.08865251,
                0.03437272, 0.05427979, 0.10172918, 0.04158039, 0.06014879,
                0.22738648, 0.08605167, 0.0346029, 0.05144877, 0.08678769,
                0.03319034, 0.05359735, 0.05454711, 0.04462109, 0.0421666,
                0.05454711, 0.08678769, 0.0421666, 0.04462109, 0.08605167,
                0.04355502, 0.04249665, 0.05083895, 0.11106131, 0.05083895,
                0.06022236, 0.11071326, 0.05083895, 0.05987431, 0.03992446,
                0.04230736, 0.04249665, 0.04355502, 0.05359735, 0.03319034,
                0.05144877, 0.0346029, 0.05427979, 0.03437272, 0.11071326,
                0.05987431, 0.05083895, 0.11106131, 0.06022236, 0.05083895,
                0.05083895, 0.07394554, 0.0790954, 0.03943577, 0.03965962,
                0.07434555, 0.03468592, 0.03965962, 0.03925961, 0.03468592
            ])
        ]
    },
]


class MMDDilatedTest(parameterized.TestCase):

  @parameterized.parameters(*_DATA)
  def test_solution_fixed_point(self, game, inverse_alpha, gambit_qre_sol):
    # Check if a QRE solution is a fixed point of MMD
    mmd = mmd_dilated.MMDDilatedEnt(game, 1. / inverse_alpha)
    mmd.sequences = copy.deepcopy(gambit_qre_sol)
    mmd.update_sequences()
    np.testing.assert_allclose(
        mmd.current_sequences()[0], gambit_qre_sol[0], rtol=1e-6)
    np.testing.assert_allclose(
        mmd.current_sequences()[1], gambit_qre_sol[1], rtol=1e-6)

  @parameterized.parameters(*_DATA)
  def test_gap(self, game, inverse_alpha, gambit_qre_sol):
    mmd = mmd_dilated.MMDDilatedEnt(game, 1. / inverse_alpha)
    mmd.sequences = copy.deepcopy(gambit_qre_sol)
    np.testing.assert_allclose(mmd.get_gap(), 0., atol=1e-6)

  @parameterized.parameters((0.), (0.5), (1.), (1.5))
  def test_rps_update(self, alpha):
    game = pyspiel.load_game_as_turn_based('matrix_rps')
    start_sequences = [
        np.array([1, 0.2, 0.2, 0.6]),
        np.array([1, 0.5, 0.2, 0.3])
    ]
    mmd = mmd_dilated.MMDDilatedEnt(game, alpha)
    mmd.sequences = copy.deepcopy(start_sequences)

    mmd.update_sequences()
    updated_sequences = copy.deepcopy(start_sequences)
    # manually perform update for p1
    updated_sequences[0][1:] = updated_sequences[0][1:] * np.exp(
        mmd.stepsize * -mmd.payoff_mat[1:, 1:] @ start_sequences[1][1:])
    updated_sequences[0][1:] = updated_sequences[0][1:]**(
        1. / (1 + mmd.stepsize * alpha))
    updated_sequences[0][1:] = updated_sequences[0][1:] / np.sum(
        updated_sequences[0][1:])
    np.testing.assert_allclose(mmd.current_sequences()[0], updated_sequences[0])

    # manually perform update for p2
    updated_sequences[1][1:] = updated_sequences[1][1:] * np.exp(
        mmd.stepsize * mmd.payoff_mat[1:, 1:].T @ start_sequences[0][1:])
    updated_sequences[1][1:] = updated_sequences[1][1:]**(
        1. / (1 + mmd.stepsize * alpha))
    updated_sequences[1][1:] = updated_sequences[1][1:] / np.sum(
        updated_sequences[1][1:])
    np.testing.assert_allclose(mmd.current_sequences()[1], updated_sequences[1])

    if alpha > 0:
      # gap cannot be computed for a value of alpha = 0
      # check that uniform random has a gap of zero
      mmd.sequences = [
          np.array([1, 0.33333333, 0.33333333, 0.33333333]),
          np.array([1, 0.33333333, 0.33333333, 0.33333333])
      ]
      np.testing.assert_allclose(mmd.get_gap(), 0.)


if __name__ == '__main__':
  absltest.main()

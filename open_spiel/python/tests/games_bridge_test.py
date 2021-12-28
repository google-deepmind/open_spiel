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

# Lint as: python3
"""Tests for the game-specific functions for bridge."""

import random
import timeit

from absl.testing import absltest
import numpy as np

import pyspiel


class GamesBridgeTest(absltest.TestCase):

  def test_contract_names(self):
    game = pyspiel.load_game('bridge(use_double_dummy_result=false)')
    self.assertEqual(game.contract_string(0), 'Passed Out')
    self.assertEqual(game.contract_string(38), '1SX N')

  def test_possible_contracts(self):
    game = pyspiel.load_game('bridge(use_double_dummy_result=false)')
    state = game.new_initial_state()
    for a in range(52):
      state.apply_action(a)
    state.apply_action(59)  # 1NT - now South cannot declare notrump
    state.apply_action(67)  # 3H - now West cannot declare hearts
    state.apply_action(86)  # 7D
    state.apply_action(53)  # Dbl
    possible_contracts = [
        game.contract_string(i)
        for i, v in enumerate(state.possible_contracts())
        if v
    ]
    self.assertCountEqual(possible_contracts, [
        '7DX S', '7DXX S', '7H N', '7HX N', '7HXX N', '7H E', '7HX E', '7HXX E',
        '7H S', '7HX S', '7HXX S', '7S N', '7SX N', '7SXX N', '7S E', '7SX E',
        '7SXX E', '7S S', '7SX S', '7SXX S', '7S W', '7SX W', '7SXX W', '7N N',
        '7NX N', '7NXX N', '7N E', '7NX E', '7NXX E', '7N W', '7NX W', '7NXX W'
    ])

  def test_scoring(self):
    game = pyspiel.load_game('bridge')
    state = game.new_initial_state()
    #         S J9873
    #         H A7
    #         D KT74
    #         C KT
    # S AKQT          S 42
    # H T852          H K63
    # D AQ            D 52
    # C Q64           C A98732
    #         S 65
    #         H QJ94
    #         D J9863
    #         C J5
    for a in [
        7, 28, 37, 2, 45, 3, 25, 51, 27, 48, 5, 43, 23, 13, 12, 8, 22, 46, 38,
        26, 9, 20, 36, 34, 32, 11, 29, 35, 44, 1, 10, 14, 39, 4, 19, 40, 50, 6,
        17, 41, 33, 0, 42, 16, 21, 18, 30, 49, 31, 24, 15, 47
    ]:
      state.apply_action(a)
    score = {
        game.contract_string(i): s
        for i, s in enumerate(state.score_by_contract())
    }
    self.assertEqual(score['3N E'], 100)
    self.assertEqual(score['3N W'], -460)
    self.assertEqual(score['1N W'], -210)
    self.assertEqual(score['3DX S'], -100)
    self.assertEqual(score['1CXX E'], -830)
    self.assertEqual(score['1CXX W'], -1030)

  def test_score_single_contract(self):
    game = pyspiel.load_game('bridge(use_double_dummy_result=false)')
    state = game.new_initial_state()
    #         S T3
    #         H QT42
    #         D A82
    #         C A632
    # S KJ5           S Q7
    # H A965          H KJ8
    # D Q43           D KJT5
    # C T87           C Q954
    #         S A98642
    #         H 73
    #         D 976
    #         C KJ
    for a in [
        49, 45, 31, 5, 10, 40, 27, 47, 35, 38, 17, 14, 0, 33, 21, 39, 34, 12,
        22, 41, 1, 13, 36, 9, 4, 46, 11, 32, 2, 37, 29, 30, 7, 8, 19, 24, 16,
        43, 51, 15, 48, 23, 6, 20, 42, 26, 44, 50, 25, 28, 3, 18
    ]:
      state.apply_action(a)
    cid = {
        game.contract_string(i): i for i in range(game.num_possible_contracts())
    }
    self.assertEqual(state.score_for_contracts(0, [cid['1H E']]), [-110])
    self.assertEqual(
        state.score_for_contracts(1, [cid['1H E'], cid['1H W']]), [110, 80])
    self.assertEqual(
        state.score_for_contracts(2, [cid['1H E'], cid['2H E'], cid['3H E']]),
        [-110, -110, 50])
    self.assertEqual(
        state.score_for_contracts(3, [cid['1H W'], cid['3N W']]), [80, -50])
    self.assertEqual(state.score_for_contracts(0, [cid['1DX N']]), [-300])
    self.assertEqual(state.score_for_contracts(1, [cid['1CXX W']]), [430])

  def test_benchmark_score_single(self):
    game = pyspiel.load_game('bridge(use_double_dummy_result=false)')
    state = game.new_initial_state()
    for a in [
        49, 45, 31, 5, 10, 40, 27, 47, 35, 38, 17, 14, 0, 33, 21, 39, 34, 12,
        22, 41, 1, 13, 36, 9, 4, 46, 11, 32, 2, 37, 29, 30, 7, 8, 19, 24, 16,
        43, 51, 15, 48, 23, 6, 20, 42, 26, 44, 50, 25, 28, 3, 18
    ]:
      state.apply_action(a)
    cid = {
        game.contract_string(i): i for i in range(game.num_possible_contracts())
    }

    for contracts in (
        ['1H E'],
        ['1H E', '1H W'],
        ['1H E', '2H E', '3H E'],
        ['1H E', '1CXX W'],
        list(cid),
        ):
      cids = [cid[c] for c in contracts]
      def benchmark(cids=cids):
        working_state = state.clone()
        _ = working_state.score_for_contracts(0, cids)
      repeat = 1
      times = np.array(timeit.repeat(benchmark, number=1, repeat=repeat))
      print(f'{contracts} mean {times.mean():.4}s, min {times.min():.4}s')

  def test_public_observation(self):
    game = pyspiel.load_game('bridge(use_double_dummy_result=false)')
    state = game.new_initial_state()
    for a in range(52):
      state.apply_action(a)
    state.apply_action(52)  # Pass
    state.apply_action(59)  # 1NT
    obs = state.public_observation_tensor()
    self.assertLen(obs, game.public_observation_tensor_size())

  def test_private_observation(self):
    game = pyspiel.load_game('bridge(use_double_dummy_result=false)')
    state = game.new_initial_state()
    #         S T3
    #         H QT42
    #         D A82
    #         C A632
    # S KJ5           S Q7
    # H A965          H KJ8
    # D Q43           D KJT5
    # C T87           C Q954
    #         S A98642
    #         H 73
    #         D 976
    #         C KJ
    for a in [
        49, 45, 31, 5, 10, 40, 27, 47, 35, 38, 17, 14, 0, 33, 21, 39, 34, 12,
        22, 41, 1, 13, 36, 9, 4, 46, 11, 32, 2, 37, 29, 30, 7, 8, 19, 24, 16,
        43, 51, 15, 48, 23, 6, 20, 42, 26, 44, 50, 25, 28, 3, 18
    ]:
      state.apply_action(a)
    obs = state.private_observation_tensor(0)
    self.assertLen(obs, game.private_observation_tensor_size())
    self.assertEqual(obs, [
        1.0, 1.0, 1.0, 0.0,  # C2, D2, H2
        1.0, 0.0, 0.0, 1.0,  # C3, S3
        0.0, 0.0, 1.0, 0.0,  # H4
        0.0, 0.0, 0.0, 0.0,  # No 5s
        1.0, 0.0, 0.0, 0.0,  # C6
        0.0, 0.0, 0.0, 0.0,  # No 7s
        0.0, 1.0, 0.0, 0.0,  # D8
        0.0, 0.0, 0.0, 0.0,  # No 9s
        0.0, 0.0, 1.0, 1.0,  # H10, S10
        0.0, 0.0, 0.0, 0.0,  # No Jacks
        0.0, 0.0, 1.0, 0.0,  # HQ
        0.0, 0.0, 0.0, 0.0,  # No kings
        1.0, 1.0, 0.0, 0.0   # CA, DA
    ])

  def test_benchmark_observation(self):
    game = pyspiel.load_game('bridge(use_double_dummy_result=false)')

    def make_state():
      state = game.new_initial_state()
      for _ in range(60):
        a = random.choice(state.legal_actions())
        state.apply_action(a)
        if state.is_terminal(): break
      return state

    batch_size = 16
    obs_shape = [batch_size] + game.observation_tensor_shape()
    states = [make_state() for _ in range(batch_size)]

    def make_obs_copy():
      inputs = np.zeros(obs_shape)
      for i in range(batch_size):
        inputs[i, :] = states[i].observation_tensor()
      return inputs

    def make_obs_inplace():
      inputs = np.zeros(obs_shape, np.float32)
      for i in range(batch_size):
        states[i].write_observation_tensor(inputs[i])
      return inputs

    repeat = 2
    number = 2
    times = np.array(timeit.repeat(make_obs_copy, number=number, repeat=repeat))
    print(f'OpenSpiel {times.mean():.4}s, min {times.min():.4}s')
    times = np.array(
        timeit.repeat(make_obs_inplace, number=number, repeat=repeat))
    print(f'In-place {times.mean():.4}s, min {times.min():.4}s')

    np.testing.assert_array_equal(make_obs_copy(), make_obs_inplace())


if __name__ == '__main__':
  absltest.main()

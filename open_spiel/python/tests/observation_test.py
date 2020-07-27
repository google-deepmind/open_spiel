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

"""Tests for third_party.open_spiel.python.observation."""

import random
import time

from absl.testing import absltest
import numpy as np

from open_spiel.python.observation import make_observation
import pyspiel


class ObservationTest(absltest.TestCase):

  def test_leduc_observation(self):
    game = pyspiel.load_game("leduc_poker")
    observation = make_observation(game)
    state = game.new_initial_state()
    state.apply_action(1)  # Deal 1
    state.apply_action(2)  # Deal 2
    state.apply_action(2)  # Bet
    state.apply_action(1)  # Call
    state.apply_action(3)  # Deal 3
    observation.set_from(state, player=0)
    np.testing.assert_array_equal(
        observation.tensor, [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 3])
    self.assertEqual(list(observation.dict), ["observation"])
    np.testing.assert_array_equal(
        observation.dict["observation"],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 3])
    self.assertEqual(
        observation.string_from(state, 0),
        "[Round 2][Player: 0][Pot: 6][Money: 97 97[Private: 1]"
        "[Ante: 3 3][Public: 3]")

  def test_leduc_info_state(self):
    game = pyspiel.load_game("leduc_poker")
    observation = make_observation(
        game, pyspiel.IIGObservationType(perfect_recall=True))
    state = game.new_initial_state()
    state.apply_action(1)  # Deal 1
    state.apply_action(2)  # Deal 2
    state.apply_action(2)  # Bet
    state.apply_action(1)  # Call
    state.apply_action(3)  # Deal 3
    observation.set_from(state, player=0)
    np.testing.assert_array_equal(observation.tensor, [
        1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0
    ])
    self.assertEqual(list(observation.dict), ["info_state"])
    np.testing.assert_array_equal(observation.dict["info_state"], [
        1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0
    ])
    self.assertEqual(
        observation.string_from(state, 0),
        "[Round 2][Player: 0][Pot: 6][Money: 97 97[Private: 1]]"
        "[Round1]: 2 1[Public: 3]\nRound 2 sequence: ")

  def test_leduc_unsupported(self):
    game = pyspiel.load_game("leduc_poker")
    with self.assertRaises(RuntimeError):
      unused_observation = make_observation(
          game,
          pyspiel.IIGObservationType(
              perfect_recall=True,
              private_info=pyspiel.PrivateInfoType.ALL_PLAYERS))

  def test_benchmark_state_generation(self):
    # Generate trajectories to test on
    game = pyspiel.load_game("chess")
    trajectories = []
    for _ in range(20):
      state = game.new_initial_state()
      while not state.is_terminal():
        state.apply_action(random.choice(state.legal_actions()))
      trajectories.append(state.history())

    # New API
    total = 0
    observation = make_observation(game)
    start = time.time()
    for trajectory in trajectories:
      state = game.new_initial_state()
      for action in trajectory:
        state.apply_action(action)
        observation.set_from(state, 0)
        total += np.mean(observation.tensor)
    end = time.time()
    print("New API time per iteration "
          f"{1000*(end-start)/len(trajectories)}msec")

    # Old API
    total = 0
    start = time.time()
    for trajectory in trajectories:
      state = game.new_initial_state()
      for action in trajectory:
        state.apply_action(action)
        obs = state.observation_tensor(0)
        tensor = np.asarray(obs)
        total += np.mean(tensor)
    end = time.time()
    print("Old API time per iteration "
          f"{1000*(end-start)/len(trajectories)}msec")


if __name__ == "__main__":
  absltest.main()

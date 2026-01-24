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

"""Tests for param_social_dilemma.py."""

from absl.testing import absltest
import numpy as np

from open_spiel.python.games import param_social_dilemma
import pyspiel


class ParamSocialDilemmaTest(absltest.TestCase):

    def test_default_params(self):
        game = pyspiel.load_game("python_param_social_dilemma")
        self.assertEqual(game._num_players, 3)
        self.assertEqual(game._num_actions, 2)
        self.assertEqual(game._max_game_length, 10)
        self.assertEqual(game._reward_noise_std, 0.0)
        self.assertFalse(game._dynamic_payoffs)

    def test_two_player_game(self):
        game = pyspiel.load_game("python_param_social_dilemma", {"num_players": 2})
        self.assertEqual(game._num_players, 2)
        self.assertEqual(game.num_players(), 2)

    def test_five_player_game(self):
        game = pyspiel.load_game("python_param_social_dilemma", {"num_players": 5})
        self.assertEqual(game._num_players, 5)
        state = game.new_initial_state()
        self.assertEqual(state.current_player(), pyspiel.PlayerId.SIMULTANEOUS)

    def test_custom_game_length(self):
        game = pyspiel.load_game("python_param_social_dilemma", 
                                {"max_game_length": 20})
        self.assertEqual(game._max_game_length, 20)
        state = game.new_initial_state()
        for _ in range(19):
            state.apply_actions([0] * game._num_players)
            self.assertFalse(state.is_terminal())
        state.apply_actions([0] * game._num_players)
        self.assertTrue(state.is_terminal())

    def test_stochastic_rewards(self):
        game = pyspiel.load_game("python_param_social_dilemma",
                                {"reward_noise_std": 0.5, "num_players": 2})
        self.assertEqual(game._reward_noise_std, 0.5)
        
        state = game.new_initial_state()
        state.apply_actions([0, 0])
        rewards1 = state.rewards()
        
        state = game.new_initial_state()
        state.apply_actions([0, 0])
        rewards2 = state.rewards()
        
        self.assertEqual(len(rewards1), 2)
        self.assertEqual(len(rewards2), 2)

    def test_dynamic_payoffs(self):
        game = pyspiel.load_game("python_param_social_dilemma",
                                {"dynamic_payoffs": True, 
                                 "payoff_change_prob": 0.5,
                                 "num_players": 2})
        self.assertTrue(game._dynamic_payoffs)
        self.assertEqual(game._payoff_change_prob, 0.5)

    def test_custom_payoff_matrix(self):
        custom_payoff = np.zeros((2, 2, 2))
        custom_payoff[0, 0] = [3, 3]
        custom_payoff[0, 1] = [0, 5]
        custom_payoff[1, 0] = [5, 0]
        custom_payoff[1, 1] = [1, 1]
        
        game = pyspiel.load_game("python_param_social_dilemma",
                                {"num_players": 2,
                                 "payoff_matrix": custom_payoff.tolist()})
        state = game.new_initial_state()
        state.apply_actions([0, 0])
        rewards = state.rewards()
        
        self.assertEqual(rewards[0], 3)
        self.assertEqual(rewards[1], 3)

    def test_game_progression(self):
        game = pyspiel.load_game("python_param_social_dilemma",
                                {"num_players": 3, "max_game_length": 5})
        state = game.new_initial_state()
        
        for step in range(5):
            self.assertFalse(state.is_terminal())
            self.assertEqual(state.current_player(), pyspiel.PlayerId.SIMULTANEOUS)
            
            actions = [0, 1, 0]
            state.apply_actions(actions)
        
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.current_player(), pyspiel.PlayerId.TERMINAL)

    def test_returns_accumulation(self):
        game = pyspiel.load_game("python_param_social_dilemma",
                                {"num_players": 2, "max_game_length": 3})
        state = game.new_initial_state()
        
        initial_returns = state.returns()
        self.assertTrue(np.all(initial_returns == 0))
        
        state.apply_actions([0, 0])
        returns_step1 = state.returns()
        
        state.apply_actions([0, 0])
        returns_step2 = state.returns()
        
        self.assertTrue(np.all(returns_step2 >= returns_step1))

    def test_legal_actions(self):
        game = pyspiel.load_game("python_param_social_dilemma",
                                {"num_players": 3, "num_actions": 3})
        state = game.new_initial_state()
        
        for player in range(game._num_players):
            legal_actions = state.legal_actions(player)
            self.assertEqual(len(legal_actions), 3)
            self.assertEqual(legal_actions, [0, 1, 2])

    def test_action_to_string(self):
        game = pyspiel.load_game("python_param_social_dilemma")
        state = game.new_initial_state()
        
        self.assertEqual(state._action_to_string(0, 0), "C")
        self.assertEqual(state._action_to_string(0, 1), "D")

    def test_random_simulation(self):
        game = pyspiel.load_game("python_param_social_dilemma",
                                {"num_players": 3, "max_game_length": 10})
        pyspiel.random_sim_test(game, num_sims=5, serialize=False, verbose=False)

    def test_turn_based_conversion(self):
        game = pyspiel.load_game("python_param_social_dilemma")
        turn_based = pyspiel.convert_to_turn_based(game)
        pyspiel.random_sim_test(turn_based, num_sims=5, serialize=False, verbose=False)

    def test_observer(self):
        game = pyspiel.load_game("python_param_social_dilemma",
                                {"num_players": 2})
        state = game.new_initial_state()
        observer = game.make_py_observer()
        
        obs_string = observer.string_from(state, 0)
        self.assertIsNotNone(obs_string)


if __name__ == "__main__":
    absltest.main()

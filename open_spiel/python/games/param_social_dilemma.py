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

"""Python implementation of parameterized social dilemma game.

A flexible N-player simultaneous-move game that supports:
- Variable number of agents (N >= 2)
- Dynamic payoff matrices that can change over timesteps
- Optional stochastic rewards with configurable noise
"""

import ast
import enum
import numpy as np
import pyspiel

_DEFAULT_PARAMS = {
    "num_players": 3,
    "num_actions": 2,
    "max_game_length": 10,
    "payoff_matrix": "default",
    "reward_noise_std": 0.0,
    "dynamic_payoffs": False,
    "payoff_change_prob": 0.0
}

_GAME_TYPE = pyspiel.GameType(
    short_name="python_param_social_dilemma",
    long_name="Python Parameterized Social Dilemma",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=10,
    min_num_players=2,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS)


class Action(enum.IntEnum):
    COOPERATE = 0
    DEFECT = 1


class ParamSocialDilemmaGame(pyspiel.Game):
    
    def __init__(self, params=_DEFAULT_PARAMS):
        self._num_players = params.get("num_players", 3)
        self._num_actions = params.get("num_actions", 2)
        self._max_game_length = params.get("max_game_length", 10)
        self._reward_noise_std = params.get("reward_noise_std", 0.0)
        self._dynamic_payoffs = params.get("dynamic_payoffs", False)
        self._payoff_change_prob = params.get("payoff_change_prob", 0.0)
        
        payoff_matrix_str = params.get("payoff_matrix", "default")
        if payoff_matrix_str == "default":
            self._payoff_matrix = self._create_default_payoff_matrix()
        else:
            try:
                parsed = ast.literal_eval(payoff_matrix_str)
                self._payoff_matrix = np.array(parsed, dtype=np.float64)
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f"Invalid payoff_matrix format: {payoff_matrix_str}. "
                    f"Expected nested list as string. Error: {e}")
        
        if self._reward_noise_std > 0:
            game_type = pyspiel.GameType(
                short_name="python_param_social_dilemma",
                long_name="Python Parameterized Social Dilemma",
                dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
                chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
                information=pyspiel.GameType.Information.PERFECT_INFORMATION,
                utility=pyspiel.GameType.Utility.GENERAL_SUM,
                reward_model=pyspiel.GameType.RewardModel.REWARDS,
                max_num_players=10,
                min_num_players=2,
                provides_information_state_string=False,
                provides_information_state_tensor=False,
                provides_observation_string=True,
                provides_observation_tensor=False,
                provides_factored_observation_string=False,
                parameter_specification=_DEFAULT_PARAMS)
        else:
            game_type = _GAME_TYPE
        
        min_utility = np.min(self._payoff_matrix) * self._max_game_length
        max_utility = np.max(self._payoff_matrix) * self._max_game_length
        
        if self._reward_noise_std > 0:
            min_utility -= 3 * self._reward_noise_std * self._max_game_length
            max_utility += 3 * self._reward_noise_std * self._max_game_length
        
        super().__init__(
            game_type,
            pyspiel.GameInfo(
                num_distinct_actions=self._num_actions,
                max_chance_outcomes=0,
                num_players=self._num_players,
                min_utility=min_utility,
                max_utility=max_utility,
                utility_sum=None,
                max_game_length=self._max_game_length,
            ),
            params,
        )
    
    def _create_default_payoff_matrix(self):
        shape = tuple([self._num_actions] * self._num_players + [self._num_players])
        payoff_matrix = np.zeros(shape)
        
        for idx in np.ndindex(shape[:-1]):
            num_cooperators = sum(1 for action in idx if action == Action.COOPERATE)
            for player in range(self._num_players):
                if idx[player] == Action.COOPERATE:
                    payoff_matrix[idx][player] = 3.0 * num_cooperators / self._num_players
                else:
                    payoff_matrix[idx][player] = 5.0 * num_cooperators / self._num_players
        
        return payoff_matrix
    
    def new_initial_state(self):
        return ParamSocialDilemmaState(self)
    
    def make_py_observer(self, iig_obs_type=None, params=None):
        return ParamSocialDilemmaObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params)
    
    def get_payoff(self, actions, timestep=0):
        actions_tuple = tuple(actions)
        payoff = self._payoff_matrix[actions_tuple].copy()
        
        if self._dynamic_payoffs and np.random.random() < self._payoff_change_prob:
            perturbation = np.random.randn(self._num_players) * 0.5
            payoff += perturbation
        
        return payoff


class ParamSocialDilemmaState(pyspiel.State):
    
    def __init__(self, game):
        super().__init__(game)
        self._game_over = False
        self._timestep = 0
        self._rewards = np.zeros(game._num_players)
        self._returns = np.zeros(game._num_players)
    
    def current_player(self):
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        else:
            return pyspiel.PlayerId.SIMULTANEOUS
    
    def _legal_actions(self, player):
        assert player >= 0
        return list(range(self.get_game()._num_actions))
    
    def _apply_actions(self, actions):
        assert not self._game_over
        game = self.get_game()
        
        base_payoff = game.get_payoff(actions, self._timestep)
        
        if game._reward_noise_std > 0:
            noise = np.random.randn(game._num_players) * game._reward_noise_std
            self._rewards = base_payoff + noise
        else:
            self._rewards = base_payoff
        
        self._returns += self._rewards
        self._timestep += 1
        
        if self._timestep >= game._max_game_length:
            self._game_over = True
    
    def _action_to_string(self, player, action):
        if action == Action.COOPERATE:
            return "C"
        elif action == Action.DEFECT:
            return "D"
        else:
            return f"A{action}"
    
    def is_terminal(self):
        return self._game_over
    
    def rewards(self):
        return self._rewards
    
    def returns(self):
        return self._returns
    
    def __str__(self):
        history = self.full_history()
        timesteps = {}
        
        for action_entry in history:
            player = action_entry.player
            action = action_entry.action
            
            if player != pyspiel.PlayerId.SIMULTANEOUS:
                t = len([h for h in history[:history.index(action_entry)] 
                        if h.player == player])
                if t not in timesteps:
                    timesteps[t] = {}
                timesteps[t][player] = self._action_to_string(player, action)
        
        result = []
        for t in sorted(timesteps.keys()):
            actions_str = ",".join([timesteps[t].get(p, "?") 
                                   for p in range(self.get_game()._num_players)])
            result.append(f"t{t}:[{actions_str}]")
        
        return " ".join(result) if result else "initial"


class ParamSocialDilemmaObserver:
    
    def __init__(self, iig_obs_type, params):
        assert not bool(params)
        self.iig_obs_type = iig_obs_type
        self.tensor = None
        self.dict = {}
    
    def set_from(self, state, player):
        pass
    
    def string_from(self, state, player):
        if self.iig_obs_type.public_info:
            return str(state)
        else:
            return None


pyspiel.register_game(_GAME_TYPE, ParamSocialDilemmaGame)

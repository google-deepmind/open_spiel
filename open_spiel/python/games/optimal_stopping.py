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

"""Python implementation of iterated prisoner's dilemma.

This is primarily here to demonstrate simultaneous-move games in Python.
"""

import enum

import numpy as np

import pyspiel

_NUM_PLAYERS = 2
_DEFAULT_PARAMS = {"termination_probability": 0.001, "max_game_length": 5, "L": 3, "R_ST": 20.0,
                   "R_SLA": 5.0, "R_COST": -5.0, "R_INT": -10.0}

# _PAYOFF = [[5, 0], [10, 1]]

_GAME_TYPE = pyspiel.GameType(
    short_name="python_optimal_stopping",
    long_name="Python Optimal Stopping",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True,
    parameter_specification=_DEFAULT_PARAMS)


def next_state(s, a1, a2, l):
    # Terminal state already
    if s == 2:
        return 2

    # Attacker aborts
    if s == 1 and a2 == 1:
        return 2

    # Defender final stop
    if a1 == 1 and l == 1:
        return 2

    # Intrusion starts
    if s == 0 and a2 == 1:
        return 1

    # Stay in the current state
    return s


def reward_function(s, a1, a2, R_SLA, R_ST, R_COST, L, R_INT, l):
    # Terminal state
    if s == 2:
        return 0

    # No intrusion state
    if s == 0:
        # Continue and Wait
        if a1 == 0 and a2 == 0:
            return R_SLA
        # Continue and Attack
        if a1 == 0 and a2 == 1:
            return R_SLA + R_ST / l
        # Stop and Wait
        if a1 == 1 and a2 == 0:
            return R_COST / L
        # Stop and Attack
        if a1 == 1 and a2 == 1:
            return R_COST / L + R_ST / L

    # Intrusion state
    if s == 1:
        # Continue and Continue
        if a1 == 0 and a2 == 0:
            return R_SLA + R_INT
        # Continue and Stop
        if a1 == 0 and a2 == 1:
            return R_SLA
        # Stop and Continue
        if a1 == 1 and a2 == 0:
            return R_COST / L + R_ST / l
        # Stop and Stop
        if a1 == 1 and a2 == 1:
            return R_COST / L

    raise ValueError("Invalid input, s:{}, a1:{}, a2:{}".format(s, a1, a2))


class Action(enum.IntEnum):
    CONTINUE = 0
    STOP = 1


class Chance(enum.IntEnum):
    OBS_0 = 0
    OBS_1 = 1
    OBS_2 = 2
    OBS_3 = 3
    OBS_4 = 4
    OBS_5 = 5
    OBS_6 = 6
    OBS_7 = 7
    OBS_8 = 8
    OBS_9 = 9
    TERMINAL = 10


class OptimalStoppingGame(pyspiel.Game):
    """The game, from which states and observers can be made."""

    def __init__(self, params=_DEFAULT_PARAMS):
        max_game_length = params["max_game_length"]
        super().__init__(
            _GAME_TYPE,
            pyspiel.GameInfo(
                num_distinct_actions=2,
                max_chance_outcomes=11,
                num_players=2,
                min_utility=100,
                max_utility=100,
                utility_sum=0.0,
                max_game_length=max_game_length), params)
        self._termination_probability = params["termination_probability"]
        self.L = params["L"]
        self.R_ST = params["R_ST"]
        self.R_SLA = params["R_SLA"]
        self.R_COST = params["R_COST"]
        self.R_INT = params["R_INT"]

    def observation_tensor_size(self):
        return 1

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return OptimalStoppingState(self, self._termination_probability, self.L, R_SLA=self.R_SLA,
                                    R_COST=self.R_COST, R_INT=self.R_INT, R_ST=self.R_ST)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return OptimalStoppingGameObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params)


class OptimalStoppingState(pyspiel.State):
    """Current state of the game."""

    def __init__(self, game, termination_probability, L, R_SLA, R_COST, R_INT, R_ST):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self._current_iteration = 1
        self._termination_probability = termination_probability
        self._is_chance = False
        self._game_over = False
        self._rewards = np.zeros(_NUM_PLAYERS)
        self._returns = np.zeros(_NUM_PLAYERS)
        self.intrusion = 0
        self.L = L
        self.l = L
        self.R_SLA = R_SLA
        self.R_COST = R_COST
        self.R_INT = R_INT
        self.R_ST = R_ST
        self.latest_obs = Chance.OBS_0

    def observation_tensor(self, player):
        return [self.latest_obs]

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every simultaneous-move game with chance.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        elif self._is_chance:
            return pyspiel.PlayerId.CHANCE
        else:
            return pyspiel.PlayerId.SIMULTANEOUS

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        return [Action.CONTINUE, Action.STOP]

    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities."""
        assert self._is_chance
        if self.intrusion == 0:
            return [(Chance.TERMINAL, self._termination_probability),
                    (Chance.OBS_0, 4 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_1, 4 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_2, 4 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_3, 2 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_4, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_5, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_6, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_7, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_8, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_9, 1 * (1 - self._termination_probability) / 20)
                    ]
        else:
            return [(Chance.TERMINAL, self._termination_probability),
                    (Chance.OBS_0, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_1, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_2, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_3, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_4, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_5, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_6, 2 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_7, 4 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_8, 4 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_9, 4 * (1 - self._termination_probability) / 20)
                    ]

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        # This is not called at simultaneous-move states.
        assert self._is_chance and not self._game_over
        self._current_iteration += 1
        self._is_chance = False
        self._game_over = (action == Chance.TERMINAL)
        if self._current_iteration > self.get_game().max_game_length():
            self._game_over = True

    def _apply_actions(self, actions):
        """Applies the specified actions (per player) to the state."""
        assert not self._is_chance and not self._game_over
        self._is_chance = True
        self._current_iteration += 1
        if self._current_iteration > self.get_game().max_game_length():
            self._game_over = True

        r = reward_function(s=self.intrusion, a1=actions[0], a2=actions[1], R_SLA=self.R_SLA,
                            R_ST=self.R_ST, R_COST=self.R_COST, L=self.L, R_INT=self.R_INT, l=self.l)
        self._rewards[0] = r
        self._rewards[1] = -r
        self._returns += self._rewards

        s_prime = next_state(s=self.intrusion, a1=actions[0], a2=actions[1], l=self.l)

        # Game ended
        if s_prime == 2:
            self._game_over = True
        else:
            self.intrusion = s_prime

        # Decrement stops left
        if actions[0] == 1:
            self.l -= 1

    def _action_to_string(self, player, action):
        """Action -> string."""
        if player == pyspiel.PlayerId.CHANCE:
            return Chance(action).name
        else:
            return Action(action).name

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._game_over

    def rewards(self):
        """Reward at the previous step."""
        return self._rewards

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return self._returns

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return (f"p0:{self.action_history_string(0)} "
                f"p1:{self.action_history_string(1)}")

    def action_history_string(self, player):
        return "".join(
            self._action_to_string(pa.player, pa.action)[0]
            for pa in self.full_history()
            if pa.player == player)


class OptimalStoppingGameObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        assert not bool(params)
        self.iig_obs_type = iig_obs_type
        self.tensor = None
        self.dict = {}

    def set_from(self, state, player):
        pass

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        if self.iig_obs_type.public_info:
            return (f"us:{state.action_history_string(player)} "
                    f"op:{state.action_history_string(1 - player)}")
        else:
            return None


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, OptimalStoppingGame)

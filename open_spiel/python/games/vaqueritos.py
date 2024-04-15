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
_DEFAULT_PARAMS = {"max_game_length": 9999} # {"termination_probability": 0.125, "max_game_length": 9999}
# _PAYOFF = [[5, 0], [10, 1]]

_GAME_TYPE = pyspiel.GameType(
    short_name="python_vaqueritos",
    long_name="Python Iterated Vaqueritos",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC, # TODO : CHANGE TO DETERMINISTIC
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM, # TODO : CHANGE TO ZERO_SUM
    reward_model=pyspiel.GameType.RewardModel.TERMINAL, # as in block_dominoes
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=False,
    provides_observation_tensor=False,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS)


class Action(enum.IntEnum):
  DEFEND = 0 
  LOAD = 1
  SHOOT = 2


# class Chance(enum.IntEnum):
#   CONTINUE = 0
#   STOP = 1


class VaqueritosGame(pyspiel.Game):
  """The game, from which states and observers can be made."""

  # pylint:disable=dangerous-default-value
  def __init__(self, params=_DEFAULT_PARAMS):
    max_game_length = params["max_game_length"]
    super().__init__(
        _GAME_TYPE,
        pyspiel.GameInfo(
            num_distinct_actions= 3,
            max_chance_outcomes=0, # CORRECT ??
            num_players=2,
            min_utility=1,
            max_utility=-1,
            utility_sum=None,
            max_game_length=max_game_length,
        ),
        params,
    )
    # self._termination_probability = params["termination_probability"]

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return VaqueritosState(self)

  # def make_py_observer(self, iig_obs_type=None, params=None):
  #   """Returns an object used for observing game state."""
  #   return VaqueritosObserver( # TODO: understand observer
  #       iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
  #       params)


class VaqueritosState(pyspiel.State):
  """Current state of the game."""

  def __init__(self, game): # self._player_won == 0
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._current_iteration = 1
    # self._termination_probability = termination_probability
    # self._is_chance = False
    self._player_won = None
    # self._rewards = np.zeros(_NUM_PLAYERS)
    # self._returns = np.zeros(_NUM_PLAYERS)
    self._bullets = np.zeros(_NUM_PLAYERS) # both players start with zero bullets

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every simultaneous-move game with chance.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._player_won != None:
      return pyspiel.PlayerId.TERMINAL
    # elif self._is_chance:
    #   return pyspiel.PlayerId.CHANCE
    else:
      return pyspiel.PlayerId.SIMULTANEOUS

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    print(player)
    assert player >= 0 
    # can only shoot if player has bullets loaded loaded
    return [Action.DEFEND, Action.LOAD, Action.SHOOT] if self._bullets[player] > 0 else [Action.DEFEND, Action.LOAD]

  # def chance_outcomes(self):
  #   """Returns the possible chance outcomes and their probabilities."""
  #   assert self._is_chance
  #   return [(Chance.CONTINUE, 1 - self._termination_probability),
  #           (Chance.STOP, self._termination_probability)]

  def _apply_action(self, action): # TODO: check if I can completely delete this as it is a simultaneous game
    """Applies the specified action to the state."""
    # This is not called at simultaneous-move states!!! 
    # assert self._is_chance and not self._game_over
    # self._current_iteration += 1
    # # self._is_chance = False
    # self._game_over = (action == Chance.STOP)
    # if self._current_iteration > self.get_game().max_game_length():
    #   self._game_over = True
    pass

  def _apply_actions(self, actions):
    """Applies the specified actions (per player) to the state."""
    assert not self.is_terminal()

    # check kill conditions
    if actions[0] == 2 and actions[1] == 1: # player 0 won
      self._player_won = 1
      return
    if actions[1] == 2 and actions[0] == 1: # player 1 won
      self._player_won = 0
      return 
    
    action_effects = np.array(actions)
    action_effects[action_effects == Action.SHOOT] = -1
    action_effects[action_effects == Action.LOAD] = 1
    action_effects[action_effects == Action.DEFEND] = 0
    self._bullets += action_effects # bullets are added/subtracted according to the action

  def _action_to_string(self, player, action):
    """Action -> string."""
    return Action(action).name

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._player_won != None

  def returns(self):
    """Total reward for each player. """
    if not self.is_terminal():
      return [0, 0]
    
    return [-1,1] if self._player_won == 0 else [1,-1]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return (f"p0:{self.action_history_string(0)} "
            f"p1:{self.action_history_string(1)}")

  def action_history_string(self, player):
    return "".join(
        self._action_to_string(pa.player, pa.action)[0]
        for pa in self.full_history()
        if pa.player == player)


# class VaqueritosObserver:
#   """Observer, conforming to the PyObserver interface (see observation.py)."""

#   def __init__(self, iig_obs_type, params):
#     """Initializes an empty observation tensor."""
#     assert not bool(params)
#     self.iig_obs_type = iig_obs_type
#     self.tensor = None
#     self.dict = {}

#   def set_from(self, state, player):
#     pass

#   def string_from(self, state, player):
#     """Observation of `state` from the PoV of `player`, as a string."""
#     if self.iig_obs_type.public_info:
#       return (f"us:{state.action_history_string(player)} "
#               f"op:{state.action_history_string(1 - player)}")
#     else:
#       return None


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, VaqueritosGame)

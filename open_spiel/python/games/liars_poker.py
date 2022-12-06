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

# Lint as python3
"""Liar's Poker implemented in Python."""

import enum

import numpy as np

import pyspiel


class Action(enum.IntEnum):
  BID = 0
  CHALLENGE = 1

_NUM_PLAYERS = 2
_HAND_LENGTH = 3
_NUM_DIGITS = 3 # Number of digits to include from the range 1, 2, ..., 9, 0
_FULL_DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
_DECK = [_FULL_DECK[i] for i in range(_NUM_DIGITS)]

_GAME_TYPE = pyspiel.GameType(
    short_name="python_liars_poker",
    long_name="Python Liars Poker",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=False,
    provides_observation_tensor=True,
    parameter_specification={
      "players": _NUM_PLAYERS,
      "hand_length": _HAND_LENGTH,
      "num_digits": _NUM_DIGITS
    })
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(Action),
    max_chance_outcomes=_HAND_LENGTH * _NUM_DIGITS,
    num_players=_NUM_PLAYERS)

class LiarsPoker(pyspiel.Game):
  """A Python version of Liar's poker."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return LiarsPokerState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return LiarsPokerObserver(
      iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
      params)


class LiarsPokerState(pyspiel.State):
  """A python version of the Liars Poker state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.hands = [] # List containing the hands for each player, represented as a list.
    self._current_player = 0
    self._current_bid = -1

  def current_player(self):
    """Returns id of the current player to act.
    
    The id is:
      - TERMINAL if game is over.
      - CHANCE if a player is drawing a number to fill out their hand.
      - a number otherwise.
    """
    if self._is_terminal:
      return pyspiel.PlayerId.TERMINAL
    elif len(self.hands) < _NUM_PLAYERS or len(self.hands[_NUM_PLAYERS - 1]) < _HAND_LENGTH:
      return pyspiel.PlayerId.CHANCE
    else:
      return self._current_player

  def _is_call_possible(self):
    raise NotImplementedError()

  def _is_challenge_possible(self):
    raise NotImplementedError()

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    actions = []
    # Any move higher than the current bid is allowed. (Bids start at 0)
    for b in range(self._current_bid + 1, _HAND_LENGTH * _NUM_PLAYERS):
      actions.append(b)
    
    if self._is_call_possible():
      actions.append(Action.BID)
    # TODO: verify Action.BID is not the same as the nubmer 0.
    if self._is_challenge_possible():
      actions.append(Action.CHALLENGE)
    # TODO: add game logic for when all players challenge - automatically count
    return actions

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    probability = 1.0 / len(_DECK)
    return [(digit, probability) for digit in _DECK]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self.is_chance_node():
      self.cards.append(action)
    else:
      self.bets.append(action)
      if action == Action.BET:
        self.pot[self._next_player] += 1
      self._next_player = 1 - self._next_player
      if ((min(self.pot) == 2) or
          (len(self.bets) == 2 and action == Action.PASS) or
          (len(self.bets) == 3)):
        self._game_over = True

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Deal:{action}"
    elif action == Action.PASS:
      return "Pass"
    else:
      return "Bet"

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    pot = self.pot
    winnings = float(min(pot))
    if not self._game_over:
      return [0., 0.]
    elif pot[0] > pot[1]:
      return [winnings, -winnings]
    elif pot[0] < pot[1]:
      return [-winnings, winnings]
    elif self.cards[0] > self.cards[1]:
      return [winnings, -winnings]
    else:
      return [-winnings, winnings]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return "".join([str(c) for c in self.cards] + ["pb"[b] for b in self.bets])


class LiarsPokerObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""
  raise NotImplementedError()

# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, LiarsPoker)

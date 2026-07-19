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

"""Python implementation of a parameterized N-player social dilemma.

This generalizes the (2-player) Iterated Prisoner's Dilemma
(iterated_prisoners_dilemma.py) to an arbitrary number of players, and adds
support for non-stationary ("dynamic") payoffs and discretized stochastic
reward noise, so it can be used as an N-player benchmark for MARL algorithms
in non-stationary or noisy-reward settings. See
https://github.com/google-deepmind/open_spiel/issues/1431.

Each round, all players simultaneously choose to COOPERATE or DEFECT. A
player's payoff depends on their own action and the number of *other*
players who cooperated that round, generalizing the classic 2x2
Temptation/Reward/Punishment/Sucker payoff matrix linearly in the fraction of
other players who cooperated:

  defect payoff  (k other cooperators) = P + (T - P) * k / (N - 1)
  cooperate payoff (k other cooperators) = S + (R - S) * k / (N - 1)

which reduces exactly to the 2-player payoff matrix when N == 2. This is the
standard linear generalization of the Prisoner's Dilemma to N players; see
e.g. Hauert & Schuster, 1997, "Effects of increasing the number of players
and memory size in the iterated Prisoner's Dilemma: a numerical approach",
Proc. Royal Society B.

Non-stationary payoffs are supported by specifying multiple (T, R, P, S)
"regimes" via the `payoff_regimes` parameter; when `dynamic_payoffs` is set,
the active regime can switch (round-robin) with probability
`payoff_change_prob` after each round. Stochastic rewards are supported via
`reward_noise_std`, which adds a shared, zero-mean, discretized
(binomially-approximated Gaussian) noise term to every player's reward each
round.

As with iterated_prisoners_dilemma.py, the number of rounds is geometrically
distributed via `termination_probability` (bounded by `max_game_length`).
"""

import enum

import numpy as np

import pyspiel

_MIN_PLAYERS = 2
_MAX_PLAYERS = 10
_DEFAULT_PLAYERS = 2

# Default single payoff regime (Temptation, Reward, Punishment, Sucker),
# matching the constants used in iterated_prisoners_dilemma.py.
_DEFAULT_PAYOFF_REGIMES = "10 5 1 0"

_DEFAULT_PARAMS = {
    "players": _DEFAULT_PLAYERS,
    "termination_probability": 0.125,
    "max_game_length": 9999,
    "payoff_regimes": _DEFAULT_PAYOFF_REGIMES,
    "dynamic_payoffs": False,
    "payoff_change_prob": 0.1,
    "reward_noise_std": 0.0,
}

# Discretized zero-mean noise outcomes approximating a Gaussian via a
# binomial(4, 0.5) weighting, scaled by reward_noise_std / 2.
_NOISE_LEVELS = (-2, -1, 0, 1, 2)
_NOISE_WEIGHTS = (1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16)

# Chance-outcome codes. These are globally disjoint (rather than reusing
# small ints per phase) so that action_to_string can decode a raw chance
# action without needing to know which phase produced it.
_REGIME_STAY = len(_NOISE_LEVELS)
_REGIME_SWITCH = _REGIME_STAY + 1
_TERMINATION_CONTINUE = _REGIME_SWITCH + 1
_TERMINATION_STOP = _TERMINATION_CONTINUE + 1
_NUM_CHANCE_OUTCOMES = _TERMINATION_STOP + 1

_GAME_TYPE = pyspiel.GameType(
    short_name="python_param_social_dilemma",
    long_name="Python Parameterized N-Player Social Dilemma",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_MAX_PLAYERS,
    min_num_players=_MIN_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS)


class Action(enum.IntEnum):
  COOPERATE = 0
  DEFECT = 1


class ChancePhase(enum.IntEnum):
  """The sequence of chance events resolved after each round of actions."""
  NOISE = 0
  REGIME = 1
  TERMINATION = 2


def _parse_payoff_regimes(payoff_regimes_str):
  """Parses a whitespace-separated string of (T, R, P, S) groups of 4."""
  flat = np.fromstring(payoff_regimes_str, dtype=np.float64, sep=" ")
  if flat.size == 0 or flat.size % 4 != 0:
    raise ValueError(
        "payoff_regimes must be a whitespace-separated list of "
        "(temptation, reward, punishment, sucker) groups of 4 floats each, "
        f"got: {payoff_regimes_str!r}")
  return flat.reshape(-1, 4)


class ParamSocialDilemmaGame(pyspiel.Game):
  """The game, from which states and observers can be made."""

  # pylint:disable=dangerous-default-value
  def __init__(self, params=_DEFAULT_PARAMS):
    params = {**_DEFAULT_PARAMS, **(params or {})}
    num_players = int(params["players"])
    if not _MIN_PLAYERS <= num_players <= _MAX_PLAYERS:
      raise ValueError(
          f"players must be between {_MIN_PLAYERS} and {_MAX_PLAYERS}, got "
          f"{num_players}")
    max_game_length = int(params["max_game_length"])

    self._payoff_regimes = _parse_payoff_regimes(params["payoff_regimes"])
    self._dynamic_payoffs = bool(params["dynamic_payoffs"])
    if self._dynamic_payoffs and self._payoff_regimes.shape[0] < 2:
      raise ValueError(
          "dynamic_payoffs=True requires at least 2 payoff regimes to be "
          "specified in payoff_regimes")
    self._payoff_change_prob = float(params["payoff_change_prob"])
    self._reward_noise_std = float(params["reward_noise_std"])
    self._termination_probability = float(params["termination_probability"])

    max_temptation = float(np.max(self._payoff_regimes[:, 0]))
    min_sucker = float(np.min(self._payoff_regimes[:, 3]))
    max_noise = (
        max(_NOISE_LEVELS) * self._reward_noise_std / 2.0
        if self._reward_noise_std else 0.0)

    super().__init__(
        _GAME_TYPE,
        pyspiel.GameInfo(
            num_distinct_actions=2,
            max_chance_outcomes=_NUM_CHANCE_OUTCOMES,
            num_players=num_players,
            min_utility=(min_sucker - max_noise) * max_game_length,
            max_utility=(max_temptation + max_noise) * max_game_length,
            utility_sum=None,
            max_game_length=max_game_length,
        ),
        params,
    )

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return ParamSocialDilemmaState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return ParamSocialDilemmaObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)


class ParamSocialDilemmaState(pyspiel.State):
  """Current state of the game."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._num_players = game.num_players()
    # pylint: disable=protected-access
    self._payoff_regimes = game._payoff_regimes
    self._dynamic_payoffs = game._dynamic_payoffs
    self._payoff_change_prob = game._payoff_change_prob
    self._reward_noise_std = game._reward_noise_std
    self._termination_probability = game._termination_probability
    # pylint: enable=protected-access

    # The ordered sequence of chance phases resolved after each round.
    self._active_phases = []
    if self._reward_noise_std > 0:
      self._active_phases.append(ChancePhase.NOISE)
    if self._dynamic_payoffs and self._payoff_regimes.shape[0] > 1:
      self._active_phases.append(ChancePhase.REGIME)
    self._active_phases.append(ChancePhase.TERMINATION)

    self._round = 0
    self._regime_index = 0
    self._phase_idx = None
    self._chance_phase = None  # None <=> a SIMULTANEOUS node.
    self._game_over = False
    self._rewards = np.zeros(self._num_players)
    self._returns = np.zeros(self._num_players)

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every simultaneous-move game with chance.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif self._chance_phase is not None:
      return pyspiel.PlayerId.CHANCE
    else:
      return pyspiel.PlayerId.SIMULTANEOUS

  def regime_index(self):
    """Index into the game's payoff_regimes currently in effect."""
    return self._regime_index

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    return [Action.COOPERATE, Action.DEFECT]

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self._chance_phase is not None
    if self._chance_phase == ChancePhase.NOISE:
      return list(enumerate(_NOISE_WEIGHTS))
    elif self._chance_phase == ChancePhase.REGIME:
      return [(_REGIME_STAY, 1 - self._payoff_change_prob),
              (_REGIME_SWITCH, self._payoff_change_prob)]
    else:
      assert self._chance_phase == ChancePhase.TERMINATION
      return [(_TERMINATION_CONTINUE, 1 - self._termination_probability),
              (_TERMINATION_STOP, self._termination_probability)]

  def _advance_chance_phase(self):
    """Moves to the next phase this round, or back to a SIMULTANEOUS node."""
    self._phase_idx += 1
    if self._phase_idx < len(self._active_phases):
      self._chance_phase = self._active_phases[self._phase_idx]
    else:
      self._chance_phase = None
      # self._rewards now holds the round's finalized reward (base payoff
      # plus any noise); fold it into the running returns exactly once.
      self._returns += self._rewards

  def _apply_action(self, action):
    """Applies the specified chance action to the state."""
    assert self._chance_phase is not None and not self._game_over
    # Note: `self._rewards` is intentionally left untouched by the REGIME and
    # TERMINATION phases below (other than by NOISE, which adds to it). Per
    # OpenSpiel's simultaneous-move + chance-node convention, Rewards() must
    # keep returning the most recently *finalized* round reward across chance
    # transitions, only changing again once the following round's
    # _apply_actions() (or another NOISE resolution) updates it. Zeroing it
    # out mid-sequence would cause it to be under-counted by callers (e.g.
    # basic_tests.cc) that sum Rewards() once per simultaneous-node visit.
    if self._chance_phase == ChancePhase.NOISE:
      noise = _NOISE_LEVELS[action] * self._reward_noise_std / 2.0
      self._rewards = self._rewards + noise
    elif self._chance_phase == ChancePhase.REGIME:
      if action == _REGIME_SWITCH:
        self._regime_index = (
            (self._regime_index + 1) % self._payoff_regimes.shape[0])
    else:
      assert self._chance_phase == ChancePhase.TERMINATION
      self._game_over = (action == _TERMINATION_STOP)
      if self._round >= self.get_game().max_game_length():
        self._game_over = True
    self._advance_chance_phase()

  def _apply_actions(self, actions):
    """Applies the specified actions (per player) to the state."""
    assert self._chance_phase is None and not self._game_over
    temptation, reward, punishment, sucker = self._payoff_regimes[
        self._regime_index]
    n = self._num_players
    num_cooperators = sum(1 for a in actions if a == Action.COOPERATE)

    self._rewards = np.zeros(n)
    for i, a in enumerate(actions):
      other_cooperators = num_cooperators - (1 if a == Action.COOPERATE else 0)
      frac = other_cooperators / (n - 1) if n > 1 else 0.0
      if a == Action.DEFECT:
        self._rewards[i] = punishment + (temptation - punishment) * frac
      else:
        self._rewards[i] = sucker + (reward - sucker) * frac
    # self._returns is updated once self._rewards is finalized; see
    # _advance_chance_phase, which is guaranteed to run at least once
    # (TERMINATION is always in self._active_phases) before control returns
    # to a SIMULTANEOUS node.

    self._round += 1
    self._phase_idx = 0
    self._chance_phase = self._active_phases[0]

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      if action < len(_NOISE_LEVELS):
        return f"Noise({_NOISE_LEVELS[action]:+d})"
      elif action == _REGIME_STAY:
        return "RegimeStay"
      elif action == _REGIME_SWITCH:
        return "RegimeSwitch"
      elif action == _TERMINATION_CONTINUE:
        return "Continue"
      else:
        assert action == _TERMINATION_STOP
        return "Stop"
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
    histories = " ".join(
        f"p{p}:{self.action_history_string(p)}"
        for p in range(self._num_players))
    return f"{histories} regime:{self._regime_index}"

  def action_history_string(self, player):
    return "".join(
        self._action_to_string(pa.player, pa.action)[0]
        for pa in self.full_history()
        if pa.player == player)


class ParamSocialDilemmaObserver:
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
      num_players = state._num_players  # pylint: disable=protected-access
      histories = " ".join(
          f"p{p}:{state.action_history_string(p)}"
          for p in range(num_players))
      return f"{histories} regime:{state.regime_index()}"
    else:
      return None


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, ParamSocialDilemmaGame)

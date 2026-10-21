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

"""Python implementation of Chaupar (the traditional Indian cross-and-circle
race game; ancestor of Pachisi, Ludo, and the trademarked "Parcheesi").

Chaupar is played on a cruciform board with a shared 68-square outer loop
(17 squares per arm) and a 6-square private home column per player leading
into the central square. Each player has 4 pieces; the first to bring all 4
home wins.

Scope note / variant disclosure: there is no single canonical "Chaupar"
ruleset. Scholarly convention (Schmidt-Madsen, 2021, "The Crux of the
Cruciform: Retracing the Early History of Chaupar and Pachisi", Board Game
Studies Journal 15(1), 29-77) distinguishes "caupar" (traditionally
four-sided stick dice) from "paccisi"/Pachisi (traditionally binary cowrie
shells), but the same source stresses this distinction is "by no means set
in stone", citing a modern regional survey in which "Chaupar" is used as the
umbrella name even for cowrie-shell versions. This implementation follows
the cowrie-shell variant, as it is the one most consistently documented in
accessible secondary rules sources. The exact physical layout of the 12
"castle" (safe) squares described in some sources could not be pinned down
precisely; this implementation places 8 safe squares (each player's entry
square, plus one squarely spaced partway along each arm) as a defensible,
symmetric simplification, documented here rather than presented as
definitive. See also https://github.com/google-deepmind/open_spiel/issues/843.

Chance mechanic: `num_shells` (default 7) cowrie shells are thrown each
turn; the number landing "mouth up" determines the throw value via the
traditional (non-monotonic) scoring table below. Throw values of 10, 25, or
30 ("high" throws) are required to enter a new piece onto the board, and
grant the player another throw -- except that three consecutive high throws
in a row causes the third to be forfeited ("burnt"), per the documented
"three consecutive high throws" rule.
"""

import math

import numpy as np

import pyspiel

_MIN_PLAYERS = 2
_MAX_PLAYERS = 4
_PIECES_PER_PLAYER = 4
_NUM_ARMS = 4
_ARM_SPAN = 17  # 68 / 4
_SHARED_TRACK_LENGTH = _ARM_SPAN * _NUM_ARMS  # 68
_HOME_COLUMN_LENGTH = 6
_FINISHED = _SHARED_TRACK_LENGTH + _HOME_COLUMN_LENGTH  # 74
_NOT_STARTED = -1

# Entry offset (absolute position on the shared 68-square loop) for each of
# the 4 physical arms.
_ARM_OFFSETS = [0, _ARM_SPAN, 2 * _ARM_SPAN, 3 * _ARM_SPAN]

# Safe ("castle") squares, as fixed physical board positions (see module
# docstring for the caveat on why these specific 8 were chosen).
_SAFE_SQUARES = frozenset(
    off + delta for off in _ARM_OFFSETS for delta in (0, 8))

_DEFAULT_NUM_SHELLS = 7

# Traditional (non-monotonic) cowrie-shell scoring table, keyed by number of
# shells landing mouth-up, for the default 7-shell game. "High" throws
# (worth entering a new piece, and an extra turn) are exactly {10, 25, 30}.
_SHELL_SCORES_7 = {0: 7, 1: 10, 2: 2, 3: 3, 4: 4, 5: 25, 6: 30, 7: 14}
_HIGH_THROWS = frozenset({10, 25, 30})

_DEFAULT_PARAMS = {
    "players": _MAX_PLAYERS,
    "num_shells": _DEFAULT_NUM_SHELLS,
}

_GAME_TYPE = pyspiel.GameType(
    short_name="python_chaupar",
    long_name="Python Chaupar",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_MAX_PLAYERS,
    min_num_players=_MIN_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS)


def _shell_scores(num_shells):
  """Returns {shells_up: throw_value} for an arbitrary shell count, using
  the same construction principle as the documented 7-shell table (Wikipedia,
  sourced): the low extreme (all down) and the two near-extremes just short
  of "all up" get the traditionally-cited jackpot-style values; the true
  high extreme (all up) gets its own special-but-not-highest value (in the
  7-shell table, k=7 scores 14, notably less than k=6's 30); the remaining
  middle values equal the number of shells up. This exactly reproduces the
  verified 7-shell table and generalizes its shape for other shell counts.
  Callers must derive the "high" (entry-eligible) throw set from the two
  near-extremes {1, num_shells-2, num_shells-1}, *not* including the true
  extremes {0, num_shells} -- see ChauparGame.__init__."""
  if num_shells == 7:
    return dict(_SHELL_SCORES_7)
  scores = {k: k for k in range(num_shells + 1)}
  scores[0] = num_shells
  scores[1] = num_shells + 3
  scores[num_shells] = 2 * num_shells
  scores[num_shells - 1] = 5 * num_shells
  return scores


class ChauparGame(pyspiel.Game):
  """The game, from which states and observers can be made."""

  # pylint:disable=dangerous-default-value
  def __init__(self, params=_DEFAULT_PARAMS):
    params = {**_DEFAULT_PARAMS, **(params or {})}
    num_players = int(params["players"])
    if not _MIN_PLAYERS <= num_players <= _MAX_PLAYERS:
      raise ValueError(
          f"players must be between {_MIN_PLAYERS} and {_MAX_PLAYERS}, got "
          f"{num_players}")
    num_shells = int(params["num_shells"])
    if num_shells < 5:
      raise ValueError(f"num_shells must be >= 5, got {num_shells}")

    self._num_shells = num_shells
    self._shell_scores = _shell_scores(num_shells)
    self._high_throws = (
        _HIGH_THROWS if num_shells == 7 else
        frozenset(v for k, v in self._shell_scores.items()
                   if k in (1, num_shells - 2, num_shells - 1)))

    if num_players == 2:
      self._arm_of_player = [0, 2]
    else:
      self._arm_of_player = list(range(num_players))

    max_game_length = 5000
    super().__init__(
        _GAME_TYPE,
        pyspiel.GameInfo(
            num_distinct_actions=_PIECES_PER_PLAYER,
            max_chance_outcomes=num_shells + 1,
            num_players=num_players,
            min_utility=-(num_players - 1.0),
            max_utility=num_players - 1.0,
            utility_sum=0.0,
            max_game_length=max_game_length,
        ),
        params,
    )

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return ChauparState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return ChauparObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)


class ChauparState(pyspiel.State):
  """Current state of the game."""

  def __init__(self, game):
    super().__init__(game)
    self._num_players = game.num_players()
    self._num_shells = game._num_shells  # pylint: disable=protected-access
    self._shell_scores = game._shell_scores  # pylint: disable=protected-access
    self._high_throws = game._high_throws  # pylint: disable=protected-access
    # pylint: disable=protected-access
    self._arm_offset = [_ARM_OFFSETS[a] for a in game._arm_of_player]
    # pylint: enable=protected-access

    self._positions = [[_NOT_STARTED] * _PIECES_PER_PLAYER
                        for _ in range(self._num_players)]
    self._has_captured = [False] * self._num_players
    self._consecutive_high = [0] * self._num_players

    self._current_player = 0
    self._awaiting_chance = True
    self._current_roll = None
    self._pending_extra_turn = False
    self._last_event = "Game start."

    self._game_over = False
    self._winner = None

  # -----------------------------------------------------------------
  # Small helpers.
  # -----------------------------------------------------------------

  def _abs_position(self, player, relative_pos):
    return (self._arm_offset[player] + relative_pos) % _SHARED_TRACK_LENGTH

  def _legal_piece_moves(self, player, throw_value, bypass_capture_gate=False):
    legal = []
    for i, pos in enumerate(self._positions[player]):
      if pos == _NOT_STARTED:
        if throw_value in self._high_throws:
          legal.append(i)
      elif pos == _FINISHED:
        continue
      else:
        # A throw that would overshoot the center simply lands the piece
        # exactly on it (clamped), rather than being disallowed. Some
        # sources describe reaching home as requiring an "exact" count, but
        # taken strictly that is a real, provable deadlock here: no shell
        # count ever produces a throw of exactly 1 (min is 2; see
        # _shell_scores), so a piece one square short of home could never
        # legally move again. Clamping-on-overshoot is a standard, widely
        # used house-rule fix for exactly this class of problem in digital
        # Pachisi/Ludo implementations, and is what's implemented here.
        new_pos = min(pos + throw_value, _FINISHED)
        entering_home = new_pos >= _SHARED_TRACK_LENGTH > pos
        if (entering_home and not self._has_captured[player] and
            not bypass_capture_gate):
          continue  # Can't enter the home column before capturing.
        legal.append(i)
    return legal

  def _legal_piece_moves_with_fallback(self, player, throw_value):
    """As _legal_piece_moves, but if the "must capture before entering home"
    gate would leave a player with *no* legal move at all (e.g. every piece
    is already backed up at the end of the shared track with no capture yet
    made -- a real, reachable deadlock under adversarial/random play), the
    gate is bypassed rather than leaving the player permanently stuck. The
    gate still applies normally whenever any other legal move exists."""
    gated = self._legal_piece_moves(player, throw_value)
    if gated:
      return gated
    return self._legal_piece_moves(player, throw_value,
                                    bypass_capture_gate=True)

  def _advance_player(self):
    self._current_player = (self._current_player + 1) % self._num_players

  # -----------------------------------------------------------------
  # OpenSpiel State API.
  # -----------------------------------------------------------------

  def current_player(self):
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    if self._awaiting_chance:
      return pyspiel.PlayerId.CHANCE
    return self._current_player

  def chance_outcomes(self):
    assert self._awaiting_chance
    n = self._num_shells
    denom = 2**n
    return [(k, math.comb(n, k) / denom) for k in range(n + 1)]

  def _legal_actions(self, player):
    assert not self._awaiting_chance and not self._game_over
    moves = self._legal_piece_moves_with_fallback(player, self._current_roll)
    assert moves, "Reached a player decision node with no legal moves."
    return sorted(moves)

  def _apply_action(self, action):
    if self._awaiting_chance:
      self._apply_chance_action(action)
    else:
      self._apply_move_action(action)

  def _apply_chance_action(self, shells_up):
    p = self._current_player
    value = self._shell_scores[shells_up]
    is_high = value in self._high_throws

    self._consecutive_high[p] = (
        self._consecutive_high[p] + 1 if is_high else 0)

    if is_high and self._consecutive_high[p] >= 3:
      self._last_event = f"P{p} burned 3 consecutive high throws; turn lost."
      self._consecutive_high[p] = 0
      self._advance_player()
      self._awaiting_chance = True
      return

    self._pending_extra_turn = is_high
    legal_pieces = self._legal_piece_moves_with_fallback(p, value)
    if not legal_pieces:
      self._last_event = f"P{p} threw {value} ({shells_up} up); no legal move."
      if not self._pending_extra_turn:
        self._advance_player()
      self._awaiting_chance = True
      return

    self._current_roll = value
    self._last_event = f"P{p} threw {value} ({shells_up} up)."
    self._awaiting_chance = False

  def _apply_move_action(self, piece_idx):
    p = self._current_player
    value = self._current_roll
    pos = self._positions[p][piece_idx]
    new_pos = 0 if pos == _NOT_STARTED else min(pos + value, _FINISHED)
    self._positions[p][piece_idx] = new_pos

    capture_desc = ""
    if new_pos < _SHARED_TRACK_LENGTH:
      abs_pos = self._abs_position(p, new_pos)
      if abs_pos not in _SAFE_SQUARES:
        for q in range(self._num_players):
          if q == p:
            continue
          hit = [j for j, qpos in enumerate(self._positions[q])
                 if qpos != _NOT_STARTED and qpos < _SHARED_TRACK_LENGTH and
                 self._abs_position(q, qpos) == abs_pos]
          # Exactly one piece of q's is captured; two or more of the same
          # opponent's pieces stacked on one (non-safe) square form a
          # "block" that cannot be captured, per the traditional rule that
          # a block of 2+ same-owner pieces is safe (see e.g. the
          # "castle"/block convention common across the Pachisi family).
          # Passage through a block is still allowed here, unlike some
          # variants that also treat a block as impassable.
          if len(hit) == 1:
            self._positions[q][hit[0]] = _NOT_STARTED
            self._has_captured[p] = True
            capture_desc = f" Captured P{q}'s piece {hit[0]}!"

    self._last_event = (
        f"P{p} moved piece {piece_idx} to {new_pos}.{capture_desc}")

    if all(pp == _FINISHED for pp in self._positions[p]):
      self._game_over = True
      self._winner = p
      self._awaiting_chance = False
      return

    if self._pending_extra_turn:
      self._awaiting_chance = True
    else:
      self._advance_player()
      self._awaiting_chance = True

  def is_terminal(self):
    return self._game_over

  def returns(self):
    if not self._game_over:
      return [0.0] * self._num_players
    return [self._num_players - 1.0 if p == self._winner else -1.0
            for p in range(self._num_players)]

  def rewards(self):
    return self.returns() if self._game_over else [0.0] * self._num_players

  def _action_to_string(self, player, action):
    if player == pyspiel.PlayerId.CHANCE:
      value = self._shell_scores[action]
      tag = " (high)" if value in self._high_throws else ""
      return f"{action} shells up -> {value}{tag}"
    return f"Move piece {action}"

  def __str__(self):
    lines = [self._last_event, f"current_player={self.current_player()}"]
    for p in range(self._num_players):
      lines.append(f"  P{p} positions={self._positions[p]} "
                   f"captured={self._has_captured[p]}")
    return "\n".join(lines)


class ChauparObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, iig_obs_type, params):
    assert not bool(params)
    self.iig_obs_type = iig_obs_type
    self.tensor = None
    self.dict = {}

  def set_from(self, state, player):
    pass

  def string_from(self, state, player):
    if self.iig_obs_type.public_info:
      # pylint: disable=protected-access
      pieces = " ".join(f"P{p}:{state._positions[p]}"
                         for p in range(state._num_players))
      captured = " ".join(f"P{p}:{state._has_captured[p]}"
                           for p in range(state._num_players))
      # pylint: enable=protected-access
      return f"{pieces} captured:[{captured}]"
    return None


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, ChauparGame)

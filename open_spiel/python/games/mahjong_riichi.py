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

"""Python implementation of (one hand of) four-player Riichi Mahjong.

This models a single hand (deal through win/draw), not a full multi-hand
match with dealer rotation and score carryover across hands -- much like
other single-hand card games in OpenSpiel (e.g. leduc_poker is one hand of
poker, not a whole session). `returns()` is the point swing for the hand.

Scope: this is a *sequential* game, not a simultaneous one. Real-life
calling (Pon/Chi/Kan/Ron) is a real-time interrupt, which OpenSpiel has no
native support for (see github.com/google-deepmind/open_spiel/issues/979).
This implements it as sequential, priority-ordered polling instead: after a
discard, every player who *could* legally Ron is asked in turn order first
(multiple simultaneous ron is allowed); only if nobody rons is the single
Pon/Kan-eligible player (if any) asked; only if nobody calls that is the
single Chi-eligible player (the discarder's next player, if eligible)
asked. This reproduces standard priority (Ron > Pon/Kan > Chi) without
requiring true real-time simultaneity.

The wall is fully pre-shuffled via 136 sequential chance draws at the start
of the hand; everything afterwards (dealing, drawing, dora reveals, kan
replacement draws) is deterministic bookkeeping over that fixed sequence.

Hand evaluation (yaku/fu/scoring) lives in mahjong_riichi_utils.py.
"""

import enum

import numpy as np

from open_spiel.python.games import mahjong_riichi_utils as u
import pyspiel

NUM_PLAYERS = 4
NUM_TILES = 136
_HAND_SIZE = 13

# --- Action encoding -------------------------------------------------------
# Discards/kans are parameterized by tile type (0-33); the remaining actions
# are singletons. IDs are globally disjoint across all phases so that
# action_to_string never needs phase context to decode an action.
_DISCARD_BASE = 0
_RIICHI_DISCARD_BASE = _DISCARD_BASE + u.NUM_TILE_TYPES  # 34
_CLOSED_KAN_BASE = _RIICHI_DISCARD_BASE + u.NUM_TILE_TYPES  # 68
_ADDED_KAN_BASE = _CLOSED_KAN_BASE + u.NUM_TILE_TYPES  # 102
_TSUMO_ACTION = _ADDED_KAN_BASE + u.NUM_TILE_TYPES  # 136
_KYUUSHU_ACTION = _TSUMO_ACTION + 1  # 137
_PASS_ACTION = _KYUUSHU_ACTION + 1  # 138
_CHI_LOW_ACTION = _PASS_ACTION + 1  # 139
_CHI_MID_ACTION = _CHI_LOW_ACTION + 1  # 140
_CHI_HIGH_ACTION = _CHI_MID_ACTION + 1  # 141
_PON_ACTION = _CHI_HIGH_ACTION + 1  # 142
_OPEN_KAN_ACTION = _PON_ACTION + 1  # 143
_RON_ACTION = _OPEN_KAN_ACTION + 1  # 144
_NUM_DISTINCT_ACTIONS = _RON_ACTION + 1  # 145


def _discard_action(t):
  return _DISCARD_BASE + t


def _riichi_discard_action(t):
  return _RIICHI_DISCARD_BASE + t


def _closed_kan_action(t):
  return _CLOSED_KAN_BASE + t


def _added_kan_action(t):
  return _ADDED_KAN_BASE + t


_DEFAULT_PARAMS = {
    "dealer": 0,
    "round_wind": int(u.EAST),
    "starting_points": 25000,
    "use_aka_dora": True,
    "honba": 0,
}

_GAME_TYPE = pyspiel.GameType(
    short_name="python_mahjong_riichi",
    long_name="Python Riichi Mahjong (single hand)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=NUM_PLAYERS,
    min_num_players=NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS)


class Phase(enum.Enum):
  DEAL = "deal"
  DRAW = "draw"
  TURN = "turn"
  REACT_RON = "react_ron"
  REACT_CALL = "react_call"  # Pon/open-kan check (single candidate).
  REACT_CHI = "react_chi"  # Chi check (single candidate).
  KAN_DRAW = "kan_draw"


class MahjongRiichiGame(pyspiel.Game):
  """The game, from which states and observers can be made."""

  # pylint:disable=dangerous-default-value
  def __init__(self, params=_DEFAULT_PARAMS):
    params = {**_DEFAULT_PARAMS, **(params or {})}
    self.dealer = int(params["dealer"])
    self.round_wind = int(params["round_wind"])
    self.starting_points = int(params["starting_points"])
    self.use_aka_dora = bool(params["use_aka_dora"])
    self.honba = int(params["honba"])
    max_swing = 48000 + self.honba * 300  # Generous bound (yakuman ron).
    super().__init__(
        _GAME_TYPE,
        pyspiel.GameInfo(
            num_distinct_actions=_NUM_DISTINCT_ACTIONS,
            max_chance_outcomes=NUM_TILES,
            num_players=NUM_PLAYERS,
            min_utility=-float(max_swing),
            max_utility=float(max_swing),
            utility_sum=0.0,
            max_game_length=400,
        ),
        params,
    )

  def new_initial_state(self):
    return MahjongRiichiState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    return MahjongRiichiObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)


def _tile_type(tile_id):
  return tile_id // 4


def _is_red_five(tile_id, use_aka_dora):
  if not use_aka_dora:
    return False
  t = tile_id // 4
  return t in (4, 13, 22) and tile_id % 4 == 3


def _next_player(p):
  return (p + 1) % NUM_PLAYERS


class GameMeld:
  """A formed meld, tracking the physical tile ids involved."""

  def __init__(self, kind, tile, is_concealed, tile_ids, called_from=None):
    self.kind = kind  # "triplet" | "sequence" | "kan"
    self.tile = tile  # type index (lowest type, for a sequence)
    self.is_concealed = is_concealed
    self.tile_ids = list(tile_ids)
    self.called_from = called_from  # player index the call was made on, or None

  def to_utils_meld(self):
    return u.Meld(self.kind, self.tile, self.is_concealed)


class MahjongRiichiState(pyspiel.State):
  """Current state of one hand of Riichi Mahjong."""

  def __init__(self, game):
    super().__init__(game)
    # Note: intentionally not caching `game` itself (e.g. self._game = game).
    # A pybind11 trampoline quirk means a raw Python-object reference to the
    # Game does not survive state (de)serialization -- self.get_game() does,
    # since it goes through the C++ binding rather than a stashed Python
    # attribute. Only plain values (ints/bools) are cached below.
    self.dealer = game.dealer
    self.round_wind = game.round_wind
    self.use_aka_dora = game.use_aka_dora
    self.honba = game.honba

    # -- Wall / dealing --
    self._wall_order = []  # Filled in during the DEAL chance phase.
    self._phase = Phase.DEAL
    self._live_wall_pointer = 0
    self._rinshan_pointer = 0
    self._num_dora_revealed = 1

    # -- Per-player state --
    self._hands = [[] for _ in range(NUM_PLAYERS)]
    self._discards = [[] for _ in range(NUM_PLAYERS)]
    self._melds = [[] for _ in range(NUM_PLAYERS)]
    self._riichi = [False] * NUM_PLAYERS
    self._double_riichi = [False] * NUM_PLAYERS
    self._riichi_discard_index = [None] * NUM_PLAYERS
    self._ippatsu_active = [False] * NUM_PLAYERS
    self._temporary_furiten = [False] * NUM_PLAYERS
    self._scores = [game.starting_points] * NUM_PLAYERS

    self._current_turn_player = (game.dealer - 1) % NUM_PLAYERS
    self._any_call_happened = False
    self._is_rinshan_draw = False
    self._pending_added_kan = None  # For chankan (robbing a kan).
    self._kan_count = 0
    self._must_discard_no_draw = False

    self._last_discard_player = None
    self._last_discard_tile_id = None
    self._last_discard_type = None
    self._is_chankan_reaction = False
    self._ron_candidates = []
    self._ron_winners = []
    self._call_candidate = None
    self._call_options = ()
    self._chi_candidate = None
    self._chi_options = ()

    self._game_over = False
    self._returns = np.zeros(NUM_PLAYERS)
    self._result_summary = ""

  # -----------------------------------------------------------------
  # Small helpers.
  # -----------------------------------------------------------------

  def seat_wind(self, player):
    return u.EAST + ((player - self.dealer) % NUM_PLAYERS)

  def _live_wall(self):
    return self._wall_order[53:122]

  def _dead_wall(self):
    return self._wall_order[122:136]

  def _dora_indicators(self):
    return self._dead_wall()[0:self._num_dora_revealed]

  def _ura_dora_indicators(self):
    return self._dead_wall()[5:5 + self._num_dora_revealed]

  def _live_wall_remaining(self):
    return len(self._live_wall()) - self._live_wall_pointer

  def _hand_counts(self, player):
    return u.counts_from_tiles(_tile_type(i) for i in self._hands[player])

  def _open_melds_utils(self, player):
    return [m.to_utils_meld() for m in self._melds[player]]

  def _is_menzen(self, player):
    return all(m.is_concealed for m in self._melds[player])

  def _remove_tile_of_type(self, player, tile_type):
    """Removes one tile of `tile_type` from player's hand, preferring to
    keep a red five in hand if a non-red copy of the same type is present.
    Returns the removed tile id."""
    hand = self._hands[player]
    candidates = [i for i in hand if _tile_type(i) == tile_type]
    non_red = [i for i in candidates if not _is_red_five(i, self.use_aka_dora)]
    removed = non_red[0] if non_red else candidates[0]
    hand.remove(removed)
    return removed

  def _aka_dora_count(self, player):
    count = sum(1 for i in self._hands[player]
                if _is_red_five(i, self.use_aka_dora))
    for m in self._melds[player]:
      count += sum(1 for i in m.tile_ids if _is_red_five(i, self.use_aka_dora))
    return count

  def _all_tile_ids(self, player, extra=()):
    ids = list(self._hands[player])
    for m in self._melds[player]:
      ids.extend(m.tile_ids)
    ids.extend(extra)
    return ids

  def _waits(self, player):
    return u.get_waits(self._hand_counts(player),
                        self._open_melds_utils(player))

  def _is_furiten(self, player):
    waits = self._waits(player)
    if not waits:
      return False
    own_discards = {_tile_type(i) for i in self._discards[player]}
    if own_discards & waits:
      return True
    return self._temporary_furiten[player]

  def _make_win_context(self, player, winning_tile, is_tsumo, **kwargs):
    # For tsumo, the winning tile is the one just drawn, already reflected
    # in self._hands[player]. For ron, it's an opponent's discard, not yet
    # part of this player's hand, so it must be added in.
    concealed = self._hand_counts(player)
    if not is_tsumo:
      concealed[winning_tile] += 1
    return u.WinContext(
        concealed_counts=concealed,
        open_melds=self._open_melds_utils(player),
        winning_tile=winning_tile,
        is_tsumo=is_tsumo,
        seat_wind=self.seat_wind(player),
        round_wind=self.round_wind,
        is_riichi=self._riichi[player],
        is_double_riichi=self._double_riichi[player],
        is_ippatsu=self._ippatsu_active[player],
        is_haitei=(is_tsumo and self._live_wall_remaining() == 0 and
                   not self._is_rinshan_draw),
        is_houtei=(not is_tsumo and self._live_wall_remaining() == 0),
        is_rinshan=(is_tsumo and self._is_rinshan_draw),
        is_chankan=kwargs.get("is_chankan", False),
        is_tenhou=kwargs.get("is_tenhou", False),
        is_chiihou=kwargs.get("is_chiihou", False),
        dora_indicators=[_tile_type(i) for i in self._dora_indicators()],
        ura_dora_indicators=(
            [_tile_type(i) for i in self._ura_dora_indicators()]
            if self._riichi[player] else []),
        aka_dora_count=self._aka_dora_count(player))

  def _can_ron(self, player, tile_type):
    if self._is_furiten(player):
      return False
    if tile_type not in self._waits(player):
      return False
    context = self._make_win_context(player, tile_type, is_tsumo=False)
    return u.evaluate_hand(context) is not None

  def _chi_options_for(self, player, tile_type):
    """Returns the subset of {"low","mid","high"} chi is legal for."""
    if tile_type >= u.HONOR:
      return set()
    counts = self._hand_counts(player)
    options = set()
    n = tile_type % 9  # 0-8, relative position within the suit.
    base = (tile_type // 9) * 9
    if n <= 6 and counts[tile_type + 1] > 0 and counts[tile_type + 2] > 0:
      options.add("low")
    if 1 <= n <= 7 and counts[tile_type - 1] > 0 and counts[tile_type + 1] > 0:
      options.add("mid")
    if n >= 2 and counts[tile_type - 2] > 0 and counts[tile_type - 1] > 0:
      options.add("high")
    del base
    return options

  # -----------------------------------------------------------------
  # OpenSpiel State API.
  # -----------------------------------------------------------------

  def current_player(self):
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    if self._phase == Phase.DEAL:
      return pyspiel.PlayerId.CHANCE
    if self._phase == Phase.REACT_RON:
      return self._ron_candidates[0]
    if self._phase == Phase.REACT_CALL:
      return self._call_candidate
    if self._phase == Phase.REACT_CHI:
      return self._chi_candidate
    return self._current_turn_player

  def chance_outcomes(self):
    assert self._phase == Phase.DEAL
    dealt = set(self._wall_order)
    remaining = [i for i in range(NUM_TILES) if i not in dealt]
    prob = 1.0 / len(remaining)
    return [(i, prob) for i in remaining]

  def _legal_actions(self, player):
    if self._phase == Phase.REACT_RON:
      return [_PASS_ACTION, _RON_ACTION]
    if self._phase == Phase.REACT_CALL:
      actions = [_PASS_ACTION]
      if "pon" in self._call_options:
        actions.append(_PON_ACTION)
      if "kan" in self._call_options:
        actions.append(_OPEN_KAN_ACTION)
      return sorted(actions)
    if self._phase == Phase.REACT_CHI:
      actions = [_PASS_ACTION]
      option_to_action = {
          "low": _CHI_LOW_ACTION, "mid": _CHI_MID_ACTION,
          "high": _CHI_HIGH_ACTION,
      }
      for opt in self._chi_options:
        actions.append(option_to_action[opt])
      return sorted(actions)
    assert self._phase == Phase.TURN
    return self._turn_legal_actions(player)

  def _turn_legal_actions(self, player):
    actions = []
    counts = self._hand_counts(player)
    just_drawn_type = (_tile_type(self._hands[player][-1])
                        if not self._must_discard_no_draw else None)

    if self._must_discard_no_draw:
      # Just called pon/chi/open-kan: must discard, no draw-only options.
      for t in range(u.NUM_TILE_TYPES):
        if counts[t] > 0:
          actions.append(_discard_action(t))
      return sorted(actions)

    if self._riichi[player]:
      # Locked in: must discard exactly the tile just drawn.
      actions.append(_discard_action(just_drawn_type))
    else:
      for t in range(u.NUM_TILE_TYPES):
        if counts[t] > 0:
          actions.append(_discard_action(t))
      if (self._is_menzen(player) and self._scores[player] >= 1000 and
          self._live_wall_remaining() >= 4):
        for t in range(u.NUM_TILE_TYPES):
          if counts[t] == 0:
            continue
          counts[t] -= 1
          if u.get_waits(counts, self._open_melds_utils(player)):
            actions.append(_riichi_discard_action(t))
          counts[t] += 1

    # Tsumo.
    context = self._make_win_context(
        player, just_drawn_type, is_tsumo=True,
        is_tenhou=self._is_tenhou_eligible(player, dealer=True),
        is_chiihou=self._is_tenhou_eligible(player, dealer=False))
    if u.evaluate_hand(context) is not None:
      actions.append(_TSUMO_ACTION)

    if not self._riichi[player]:
      # Closed kan.
      for t in range(u.NUM_TILE_TYPES):
        if counts[t] == 4 and self._kan_count < 4:
          actions.append(_closed_kan_action(t))
      # Added kan (must use the just-drawn tile).
      if self._kan_count < 4:
        for m in self._melds[player]:
          if (m.kind == "triplet" and m.tile == just_drawn_type and
              counts[just_drawn_type] >= 1):
            actions.append(_added_kan_action(just_drawn_type))

    if self._is_kyuushu_eligible(player):
      actions.append(_KYUUSHU_ACTION)

    return sorted(actions)

  def _is_tenhou_eligible(self, player, dealer):
    if self._any_call_happened or self._discards[player]:
      return False
    offset = (player - self.dealer) % NUM_PLAYERS
    if dealer:
      return offset == 0 and self._live_wall_pointer == 0
    # Non-dealer's first draw is their `offset`-th live-wall draw overall,
    # assuming (as guaranteed by _any_call_happened above) nothing has
    # interrupted the deterministic deal-order sequence so far.
    return offset != 0 and self._live_wall_pointer == offset

  def _is_kyuushu_eligible(self, player):
    if self._any_call_happened or self._discards[player]:
      return False
    if any(self._discards[p] for p in range(NUM_PLAYERS)):
      return False
    counts = self._hand_counts(player)
    distinct_orphans = sum(1 for t in u.ORPHANS if counts[t] > 0)
    return distinct_orphans >= 9

  def _apply_action(self, action):
    if self._phase == Phase.DEAL:
      self._wall_order.append(action)
      if len(self._wall_order) == NUM_TILES:
        self._finish_deal()
      return
    if self._phase == Phase.REACT_RON:
      self._apply_react_ron(action)
      return
    if self._phase == Phase.REACT_CALL:
      self._apply_react_call(action)
      return
    if self._phase == Phase.REACT_CHI:
      self._apply_react_chi(action)
      return
    assert self._phase == Phase.TURN
    self._apply_turn_action(action)

  # -- DEAL --------------------------------------------------------------

  def _finish_deal(self):
    order = [(self.dealer + i) % NUM_PLAYERS for i in range(NUM_PLAYERS)]
    sizes = [14, 13, 13, 13]
    idx = 0
    for p, size in zip(order, sizes):
      self._hands[p] = list(self._wall_order[idx:idx + size])
      idx += size
    assert idx == 53
    self._must_discard_no_draw = False
    self._phase = Phase.TURN
    self._current_turn_player = self.dealer

  # -- TURN ----------------------------------------------------------------

  def _apply_turn_action(self, action):
    player = self._current_turn_player
    if action == _TSUMO_ACTION:
      self._resolve_tsumo(player)
    elif action == _KYUUSHU_ACTION:
      self._resolve_abortive_draw("Kyuushu Kyuuhai")
    elif _CLOSED_KAN_BASE <= action < _ADDED_KAN_BASE:
      self._do_closed_kan(player, action - _CLOSED_KAN_BASE)
    elif _ADDED_KAN_BASE <= action < _TSUMO_ACTION:
      self._do_added_kan(player, action - _ADDED_KAN_BASE)
    elif _RIICHI_DISCARD_BASE <= action < _CLOSED_KAN_BASE:
      self._do_discard(player, action - _RIICHI_DISCARD_BASE, riichi=True)
    else:
      assert _DISCARD_BASE <= action < _RIICHI_DISCARD_BASE
      self._do_discard(player, action - _DISCARD_BASE, riichi=False)

  def _do_discard(self, player, tile_type, riichi):
    tile_id = self._remove_tile_of_type(player, tile_type)
    self._discards[player].append(tile_id)
    if riichi:
      self._riichi[player] = True
      self._scores[player] -= 1000
      if not self._any_call_happened and len(self._discards[player]) == 1:
        self._double_riichi[player] = True
      self._ippatsu_active[player] = True
    else:
      self._ippatsu_active[player] = False
    self._temporary_furiten[player] = False
    self._must_discard_no_draw = False
    self._last_discard_player = player
    self._last_discard_tile_id = tile_id
    self._last_discard_type = tile_type
    self._start_reactions(player, tile_type, is_chankan=False)

  def _do_closed_kan(self, player, tile_type):
    ids = [i for i in self._hands[player] if _tile_type(i) == tile_type]
    for i in ids:
      self._hands[player].remove(i)
    self._melds[player].append(GameMeld("kan", tile_type, True, ids))
    self._kan_count += 1
    for p in range(NUM_PLAYERS):
      self._ippatsu_active[p] = False
    self._any_call_happened = True
    if self._check_four_kan_abort():
      return
    self._reveal_new_dora()
    self._draw_rinshan(player)

  def _do_added_kan(self, player, tile_type):
    meld = next(m for m in self._melds[player]
                if m.kind == "triplet" and m.tile == tile_type)
    tile_id = self._remove_tile_of_type(player, tile_type)
    meld.kind = "kan"
    meld.tile_ids.append(tile_id)
    self._kan_count += 1
    for p in range(NUM_PLAYERS):
      self._ippatsu_active[p] = False
    self._any_call_happened = True
    self._last_discard_player = player
    self._last_discard_tile_id = tile_id
    self._last_discard_type = tile_type
    self._pending_added_kan = (player, tile_type)
    self._start_reactions(player, tile_type, is_chankan=True)

  def _reveal_new_dora(self):
    if self._num_dora_revealed < 5:
      self._num_dora_revealed += 1

  def _draw_rinshan(self, player):
    rinshan_pool = self._dead_wall()[10:14]
    tile_id = rinshan_pool[self._rinshan_pointer]
    self._rinshan_pointer += 1
    self._hands[player].append(tile_id)
    self._is_rinshan_draw = True
    self._must_discard_no_draw = False
    self._phase = Phase.TURN
    self._current_turn_player = player

  # -- Reactions (Ron / Pon / Kan / Chi) ------------------------------------

  def _start_reactions(self, discarder, tile_type, is_chankan):
    self._is_chankan_reaction = is_chankan
    self._ron_candidates = [
        (discarder + offset) % NUM_PLAYERS
        for offset in range(1, NUM_PLAYERS)
        if self._can_ron((discarder + offset) % NUM_PLAYERS, tile_type)
    ]
    self._ron_winners = []
    self._call_candidate = None
    self._call_options = set()
    self._chi_candidate = None
    self._chi_options = set()

    if not is_chankan:
      for offset in range(1, NUM_PLAYERS):
        p = (discarder + offset) % NUM_PLAYERS
        counts = self._hand_counts(p)
        opts = set()
        if counts[tile_type] >= 2:
          opts.add("pon")
        if counts[tile_type] >= 3 and self._kan_count < 4:
          opts.add("kan")
        if opts:
          self._call_candidate = p
          self._call_options = opts
          break
      next_p = _next_player(discarder)
      chi_opts = self._chi_options_for(next_p, tile_type)
      if chi_opts:
        self._chi_candidate = next_p
        self._chi_options = chi_opts

    if self._ron_candidates:
      self._phase = Phase.REACT_RON
    elif self._call_candidate is not None:
      self._phase = Phase.REACT_CALL
    elif self._chi_candidate is not None:
      self._phase = Phase.REACT_CHI
    else:
      self._no_reaction_taken()

  def _advance_reaction_cascade_after_ron(self):
    if self._call_candidate is not None:
      self._phase = Phase.REACT_CALL
    elif self._chi_candidate is not None:
      self._phase = Phase.REACT_CHI
    else:
      self._no_reaction_taken()

  def _advance_reaction_cascade_after_call(self):
    if self._chi_candidate is not None:
      self._phase = Phase.REACT_CHI
    else:
      self._no_reaction_taken()

  def _no_reaction_taken(self):
    if self._pending_added_kan is not None:
      player, _ = self._pending_added_kan
      self._pending_added_kan = None
      if self._check_four_kan_abort():
        return
      self._reveal_new_dora()
      self._draw_rinshan(player)
      return
    if self._check_four_riichi_abort() or self._check_four_wind_discard_abort():
      return
    self._advance_to_next_turn()

  def _check_four_kan_abort(self):
    if self._kan_count != 4:
      return False
    owners = {p for p in range(NUM_PLAYERS)
              for m in self._melds[p] if m.kind == "kan"}
    if len(owners) > 1:
      self._resolve_abortive_draw("Suukaikan (four kans by different players)")
      return True
    return False

  def _check_four_riichi_abort(self):
    if all(self._riichi):
      self._resolve_abortive_draw("Suuchariichi (four riichi)")
      return True
    return False

  def _check_four_wind_discard_abort(self):
    if self._any_call_happened:
      return False
    if not all(len(self._discards[p]) == 1 for p in range(NUM_PLAYERS)):
      return False
    first_types = {_tile_type(self._discards[p][0]) for p in range(NUM_PLAYERS)}
    if len(first_types) == 1 and next(iter(first_types)) in u.WINDS:
      self._resolve_abortive_draw("Suufon Renda (four-wind discard)")
      return True
    return False

  def _apply_react_ron(self, action):
    candidate = self._ron_candidates.pop(0)
    if action == _RON_ACTION:
      self._ron_winners.append(candidate)
    else:
      self._temporary_furiten[candidate] = True
    if self._ron_candidates:
      return
    if self._ron_winners:
      self._resolve_multi_ron()
    else:
      self._advance_reaction_cascade_after_ron()

  def _apply_react_call(self, action):
    candidate = self._call_candidate
    self._call_candidate = None
    if action == _PON_ACTION:
      self._do_pon(candidate)
    elif action == _OPEN_KAN_ACTION:
      self._do_open_kan(candidate)
    else:
      assert action == _PASS_ACTION
      self._advance_reaction_cascade_after_call()

  def _apply_react_chi(self, action):
    candidate = self._chi_candidate
    self._chi_candidate = None
    if action == _PASS_ACTION:
      self._no_reaction_taken()
    else:
      option = {_CHI_LOW_ACTION: "low", _CHI_MID_ACTION: "mid",
                _CHI_HIGH_ACTION: "high"}[action]
      self._do_chi(candidate, option)

  def _do_pon(self, caller):
    tile_type = self._last_discard_type
    ids_in_hand = [i for i in self._hands[caller]
                   if _tile_type(i) == tile_type][:2]
    for i in ids_in_hand:
      self._hands[caller].remove(i)
    meld_ids = ids_in_hand + [self._last_discard_tile_id]
    self._melds[caller].append(
        GameMeld("triplet", tile_type, False, meld_ids,
                 called_from=self._last_discard_player))
    self._finish_call(caller)

  def _do_open_kan(self, caller):
    tile_type = self._last_discard_type
    ids_in_hand = [i for i in self._hands[caller]
                   if _tile_type(i) == tile_type]
    for i in ids_in_hand:
      self._hands[caller].remove(i)
    meld_ids = ids_in_hand + [self._last_discard_tile_id]
    self._melds[caller].append(
        GameMeld("kan", tile_type, False, meld_ids,
                 called_from=self._last_discard_player))
    self._kan_count += 1
    for p in range(NUM_PLAYERS):
      self._ippatsu_active[p] = False
    self._any_call_happened = True
    if self._check_four_kan_abort():
      return
    self._reveal_new_dora()
    self._draw_rinshan(caller)

  def _do_chi(self, caller, option):
    tile_type = self._last_discard_type
    if option == "low":
      needed, seq_start = (tile_type + 1, tile_type + 2), tile_type
    elif option == "mid":
      needed, seq_start = (tile_type - 1, tile_type + 1), tile_type - 1
    else:
      needed, seq_start = (tile_type - 2, tile_type - 1), tile_type - 2
    hand = self._hands[caller]
    ids_in_hand = []
    for t in needed:
      tid = next(i for i in hand if _tile_type(i) == t and i not in ids_in_hand)
      ids_in_hand.append(tid)
    for i in ids_in_hand:
      hand.remove(i)
    meld_ids = ids_in_hand + [self._last_discard_tile_id]
    self._melds[caller].append(
        GameMeld("sequence", seq_start, False, meld_ids,
                 called_from=self._last_discard_player))
    self._finish_call(caller)

  def _finish_call(self, caller):
    for p in range(NUM_PLAYERS):
      self._ippatsu_active[p] = False
    self._any_call_happened = True
    self._must_discard_no_draw = True
    self._phase = Phase.TURN
    self._current_turn_player = caller

  def _advance_to_next_turn(self):
    self._start_turn(_next_player(self._last_discard_player))

  def _start_turn(self, player):
    if self._live_wall_remaining() <= 0:
      self._resolve_exhaustive_draw()
      return
    tile_id = self._live_wall()[self._live_wall_pointer]
    self._live_wall_pointer += 1
    self._hands[player].append(tile_id)
    self._is_rinshan_draw = False
    self._must_discard_no_draw = False
    self._phase = Phase.TURN
    self._current_turn_player = player

  # -- Settlement ------------------------------------------------------------

  def _resolve_tsumo(self, player):
    just_drawn_type = _tile_type(self._hands[player][-1])
    context = self._make_win_context(
        player, just_drawn_type, is_tsumo=True,
        is_tenhou=self._is_tenhou_eligible(player, dealer=True),
        is_chiihou=self._is_tenhou_eligible(player, dealer=False))
    result = u.evaluate_hand(context)
    is_dealer = (player == self.dealer)
    payments = u.compute_payments(result, is_dealer, is_tsumo=True,
                                   honba=self.honba)
    if is_dealer:
      each = payments["each"]
      for p in range(NUM_PLAYERS):
        if p != player:
          self._scores[p] -= each
          self._scores[player] += each
    else:
      for p in range(NUM_PLAYERS):
        if p == player:
          continue
        amt = payments["dealer"] if p == self.dealer else payments["non_dealer"]
        self._scores[p] -= amt
        self._scores[player] += amt
    self._collect_riichi_sticks(player)
    self._result_summary = f"Tsumo by {player}: {result}"
    self._finalize_scores()

  def _resolve_multi_ron(self):
    discarder = self._last_discard_player
    tile_type = self._last_discard_type
    is_chankan = self._is_chankan_reaction
    for winner in self._ron_winners:
      context = self._make_win_context(winner, tile_type, is_tsumo=False,
                                        is_chankan=is_chankan)
      result = u.evaluate_hand(context)
      is_dealer = (winner == self.dealer)
      payments = u.compute_payments(result, is_dealer, is_tsumo=False,
                                     honba=self.honba)
      amount = payments["loser"]
      self._scores[discarder] -= amount
      self._scores[winner] += amount
    self._collect_riichi_sticks(self._ron_winners[0])
    self._result_summary = f"Ron by {self._ron_winners}"
    self._finalize_scores()

  def _collect_riichi_sticks(self, winner):
    riichi_sticks = sum(1 for p in range(NUM_PLAYERS) if self._riichi[p])
    self._scores[winner] += 1000 * riichi_sticks

  def _resolve_abortive_draw(self, reason):
    for p in range(NUM_PLAYERS):
      if self._riichi[p]:
        self._scores[p] += 1000
    self._result_summary = f"Abortive draw: {reason}"
    self._finalize_scores()

  def _resolve_exhaustive_draw(self):
    tenpai = [p for p in range(NUM_PLAYERS) if self._waits(p)]
    noten = [p for p in range(NUM_PLAYERS) if p not in tenpai]
    if 0 < len(tenpai) < NUM_PLAYERS:
      gain = 3000 // len(tenpai)
      loss = 3000 // len(noten)
      for p in tenpai:
        self._scores[p] += gain
      for p in noten:
        self._scores[p] -= loss
    for p in range(NUM_PLAYERS):
      if self._riichi[p]:
        self._scores[p] += 1000
    self._result_summary = "Exhaustive draw"
    self._finalize_scores()

  def _finalize_scores(self):
    self._returns = np.array(
        [self._scores[p] - self.get_game().starting_points
         for p in range(NUM_PLAYERS)], dtype=float)
    self._game_over = True

  # -----------------------------------------------------------------
  # Remaining OpenSpiel State API.
  # -----------------------------------------------------------------

  def is_terminal(self):
    return self._game_over

  def returns(self):
    return self._returns

  def rewards(self):
    return self._returns

  def _action_to_string(self, player, action):
    if player == pyspiel.PlayerId.CHANCE:
      return f"Deal({u.tile_name(_tile_type(action))}#{action % 4})"
    if action == _TSUMO_ACTION:
      return "Tsumo"
    if action == _KYUUSHU_ACTION:
      return "KyuushuKyuuhai"
    if action == _PASS_ACTION:
      return "Pass"
    if action == _CHI_LOW_ACTION:
      return "ChiLow"
    if action == _CHI_MID_ACTION:
      return "ChiMid"
    if action == _CHI_HIGH_ACTION:
      return "ChiHigh"
    if action == _PON_ACTION:
      return "Pon"
    if action == _OPEN_KAN_ACTION:
      return "OpenKan"
    if action == _RON_ACTION:
      return "Ron"
    if _DISCARD_BASE <= action < _RIICHI_DISCARD_BASE:
      return f"Discard({u.tile_name(action - _DISCARD_BASE)})"
    if _RIICHI_DISCARD_BASE <= action < _CLOSED_KAN_BASE:
      return f"RiichiDiscard({u.tile_name(action - _RIICHI_DISCARD_BASE)})"
    if _CLOSED_KAN_BASE <= action < _ADDED_KAN_BASE:
      return f"ClosedKan({u.tile_name(action - _CLOSED_KAN_BASE)})"
    assert _ADDED_KAN_BASE <= action < _TSUMO_ACTION
    return f"AddedKan({u.tile_name(action - _ADDED_KAN_BASE)})"

  def __str__(self):
    hands = [sorted(u.tile_name(_tile_type(i)) for i in h)
             for h in self._hands]
    return (f"phase={self._phase.value} "
            f"turn_player={self._current_turn_player} "
            f"hands={hands} discards={self._discards} scores={self._scores} "
            f"result={self._result_summary}")


class MahjongRiichiObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, iig_obs_type, params):
    assert not bool(params)
    self.iig_obs_type = iig_obs_type
    self.tensor = None
    self.dict = {}

  def set_from(self, state, player):
    pass

  def string_from(self, state, player):
    # pylint: disable=protected-access
    pieces = [
        "hand:" + ",".join(sorted(
            u.tile_name(_tile_type(i)) for i in state._hands[player]))
    ]
    for p in range(NUM_PLAYERS):
      pieces.append(f"discards{p}:" + ",".join(
          u.tile_name(_tile_type(i)) for i in state._discards[p]))
      melds_str = ";".join(
          f"{m.kind}:{u.tile_name(m.tile)}" for m in state._melds[p])
      pieces.append(f"melds{p}:{melds_str}")
    pieces.append(f"riichi:{state._riichi}")
    pieces.append("dora:" + ",".join(
        u.tile_name(_tile_type(i)) for i in state._dora_indicators()))
    # pylint: enable=protected-access
    return " ".join(pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, MahjongRiichiGame)

# Copyright 2026 DeepMind Technologies Limited
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

"""Contract bridge Env over the JAX engine `bridge`.

Auction and card play together.

Four seats (0..3 = North, East, South, West; partnership = seat & 1), 90
actions with pyspiel's ids (0..51 cards, 52 pass, 53 double, 54 redouble,
55..89 the bids 1C..7NT), duplicate-point returns. One hand is one board.

Chance (raw layer): with ``random_dealer_vul`` (the default) one node draws a
position of the 16-board duplicate cycle, which sets the dealer and both
sides' vulnerability, so one network trains on every dealer and
vulnerability; then the 52 deal nodes (outcome = card id; the i-th card dealt
goes to seat i % 4). With ``random_dealer_vul=False`` the dealer and
vulnerability are the options ``dealer``, ``dealer_vul`` and
``non_dealer_vul``, matching pyspiel's dealer and vulnerability game
parameters.

Acting seat: the declarer chooses dummy's cards, so while dummy is due the
actor is the declarer and the legal mask holds dummy's cards. Dummy never
acts.

INFOSTATE. One seat's view, a dict with these keys and dtypes
(``info_template``):

  seat         int8          the observing seat
  hand         bool[52]      the 13 cards the seat was dealt
  dummy        bool[52]      the 13 cards dummy was dealt, once dummy is face
                             up (after the opening lead); all False before
  dummy_shown  bool          dummy is face up
  actions      int16[371]    every call, then every card, in order; -1 pad
  num_actions  int16         the number of entries in ``actions``
  dealer       int8          the dealer's seat
  vul          bool[2]       North-South, East-West vulnerable

Everything else a seat knows follows from these: the contract and declarer,
who played each card (the play order follows from the tricks), the cards
remaining in its hand and in dummy's. The other hands are never in it.
COMPACT HISTORY (``compact_history=True``). The same view with the action
list stored in about a fifth of the bytes; ``actions`` and ``num_actions``
are replaced by:

  calls        uint8[105]    the non-pass calls in order, each one byte:
                             (call index - 1) * 4 + caller seat; 255 pad
  n_calls      int16         the number of calls made, passes included
  play         uint8[52]     the cards in order of play; 255 pad

Passes are implied: callers rotate from the dealer, so the passes before a
non-pass call are the seats between it and the previous one, and the
trailing passes are n_calls less the calls placed. Each of the 35 bids is
made at most once and followed by at most one double and one redouble, so
105 non-pass calls is the longest auction. ``expand_history`` rebuilds the
exact int16[371] list; the encoder reads either form. The raw action list
is 742 bytes, the compact one 159; a learner batch stores one InfoState per
seat per ply, so this is most of a learner batch's memory.

``labels`` carries the full deal (``deal_holder``, the seat each card was
dealt to) for learners. Every action is public (``visible_action`` is the
action). ``world_from_info`` is None: a seat's view does not fix the hidden
hands; ``hide_hidden_cards`` is the privacy hook that re-deals them.

RULES IDENTITY. The env declares its rules_sha (``declared_rules_sha``,
which envs.registry.rules_sha returns) instead of the hash of the engine's
module, so the hash names the rules rather than a source file. The
board-cycle game (``random_dealer_vul``) is pyspiel's bridge without the
double-dummy result (RULES), and RULES_SHA is the sha256 of RULES over
canonical JSON (sorted keys, no spaces). A fixed board is a different rule set:
its sha hashes
RULES with the dealer and vulnerability stated. compact_history is storage,
not rules: it does not change the sha.

Bridge is not served (its GameSpec has no ServingSpec). ``covers`` is the
GameSpec.covers of the game.
"""

import hashlib
import json
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import bridge
from open_spiel.python.scale.envs import jax_game_env

NUM_CARDS = bridge.NUM_CARDS
MAX_LEN = bridge.MAX_GAME_LENGTH  # 371 player decisions in the longest board
#: Non-pass calls in the longest auction: 35 bids, each doubled and redoubled.
MAX_NONPASS_CALLS = 3 * bridge.NUM_BIDS  # 105
PAD_U8 = 255
#: The board-cycle rule set: pyspiel's bridge with the cards played out (no
#: double-dummy result), dealer and vulnerability from the 16-board duplicate
#: cycle, duplicate scoring. The JAX engine is checked move for move against
#: that pyspiel game.
RULES = {
    "schema": "bridge-rules/1",
    "engine": "open_spiel==2.0.1",
    "game": "bridge(use_double_dummy_result=false)",
    "dealer_vul": "board-cycle",
    "scoring": "duplicate (pyspiel bridge_scoring.cc)",
}


def _sha(rules: Mapping[str, Any]) -> str:
  """The digest of a rules mapping.

  Args:
    rules: The rules mapping.

  Returns:
    The hex SHA-256 of its canonical JSON.
  """
  return hashlib.sha256(
      json.dumps(dict(rules), sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()


RULES_SHA = _sha(RULES)


def info_template() -> dict[str, np.ndarray]:
  """One seat's InfoState (unbatched) as zero arrays.

  Of its shapes and dtypes.

  Returns:
    The zero arrays, keyed by field.
  """
  return {
      "seat": np.zeros((), np.int8),
      "hand": np.zeros((NUM_CARDS,), np.bool_),
      "dummy": np.zeros((NUM_CARDS,), np.bool_),
      "dummy_shown": np.zeros((), np.bool_),
      "actions": np.full((MAX_LEN,), -1, np.int16),
      "num_actions": np.zeros((), np.int16),
      "dealer": np.zeros((), np.int8),
      "vul": np.zeros((2,), np.bool_),
  }


def compact_info_template() -> dict[str, np.ndarray]:
  """One seat's compact-history InfoState (unbatched) as zero arrays.

  Returns:
    The zero arrays, keyed by field.
  """
  out = {
      k: v
      for k, v in info_template().items()
      if k not in ("actions", "num_actions")
  }
  out.update(
      calls=np.full((MAX_NONPASS_CALLS,), PAD_U8, np.uint8),
      n_calls=np.zeros((), np.int16),
      play=np.full((NUM_CARDS,), PAD_U8, np.uint8),
  )
  return out


def compact_history(s: bridge.GameState) -> dict[str, jax.Array]:
  """The state's action list as {calls, n_calls, play} (module doc).

  Args:
    s: The engine state.

  Returns:
    {calls, n_calls, play}.
  """
  n_calls = s.n_calls.astype(jnp.int32)
  k = jnp.arange(bridge.MAX_AUCTION_LENGTH, dtype=jnp.int32)
  call = s.calls.astype(jnp.int32)
  nonpass = (k < n_calls) & (call > bridge.CALL_PASS)
  caller = (s.dealer.astype(jnp.int32) + k) % 4
  byte = jnp.where(nonpass, (call - 1) * 4 + caller, PAD_U8)
  # The non-pass calls first, in order (a stable sort on a key unique per k).
  order = jnp.argsort(jnp.where(nonpass, k, bridge.MAX_AUCTION_LENGTH + k))
  calls = byte[order[:MAX_NONPASS_CALLS]]
  j = jnp.arange(NUM_CARDS, dtype=jnp.int32)
  play = jnp.where(
      j < s.n_played.astype(jnp.int32),
      s.trick_cards.astype(jnp.int32).reshape(-1),
      PAD_U8,
  )
  return {
      "calls": calls.astype(jnp.uint8),
      "n_calls": n_calls.astype(jnp.int16),
      "play": play.astype(jnp.uint8),
  }


def expand_history(info) -> tuple[jax.Array, jax.Array]:
  """The action list of one (unbatched) compact InfoState.

  (int16[371] actions, int16 count) of one (unbatched) compact InfoState:
  exactly bridge.action_history of the state it was taken from.

  Args:
    info: One unbatched compact InfoState.

  Returns:
    (int16[371] actions, int16 count): exactly bridge.action_history of the
    state it was taken from.
  """
  calls = jnp.asarray(info["calls"]).astype(jnp.int32)
  n_calls = jnp.asarray(info["n_calls"]).astype(jnp.int32)
  play = jnp.asarray(info["play"]).astype(jnp.int32)
  dealer = jnp.asarray(info["dealer"]).astype(jnp.int32)
  placed = calls != PAD_U8
  caller = calls % 4
  prev = jnp.concatenate([(dealer - 1)[None], caller[:-1]])
  gap = (
      caller - prev - 1
  ) % 4  # passes between the previous non-pass call and this one
  pos = jnp.cumsum(gap + 1) - 1
  k = jnp.arange(MAX_LEN, dtype=jnp.int32)
  out = jnp.where(k < n_calls, bridge.PASS_ACTION, -1)
  out = out.at[jnp.where(placed, pos, MAX_LEN)].set(
      calls // 4 + 1 + bridge.PASS_ACTION, mode="drop"
  )
  n_played = jnp.sum(play != PAD_U8).astype(jnp.int32)
  is_card = (k >= n_calls) & (k < n_calls + n_played)
  out = jnp.where(is_card, play[jnp.clip(k - n_calls, 0, NUM_CARDS - 1)], out)
  return out.astype(jnp.int16), (n_calls + n_played).astype(jnp.int16)


class BridgeEnv(jax_game_env.JaxGameEnv):
  """Contract bridge: 4 seats, 90 actions, duplicate scoring.

  Attributes:
    dealer: The dealer seat.
    dealer_vul: Whether the dealer's side is vulnerable.
    non_dealer_vul: Whether the other side is vulnerable.
    random_dealer_vul: Whether the dealer and vulnerability are dealt at random
      (the 16-board duplicate cycle).
    compact_history: Whether the InfoState stores its action list compactly.
  """

  name = "bridge"
  max_len = MAX_LEN
  zero_sum = True

  def __init__(
      self,
      dealer: int = 0,
      dealer_vul: bool = False,
      non_dealer_vul: bool = False,
      random_dealer_vul: bool = True,
      # The keyword is part of the public signature.
      compact_history: bool = False,  # pylint: disable=redefined-outer-name
  ):
    super().__init__(
        bridge.Bridge(
            use_double_dummy_result=False,
            dealer=dealer,
            dealer_vul=dealer_vul,
            non_dealer_vul=non_dealer_vul,
            random_dealer_vul=random_dealer_vul,
        )
    )
    self.dealer = int(dealer)
    self.dealer_vul = bool(dealer_vul)
    self.non_dealer_vul = bool(non_dealer_vul)
    self.random_dealer_vul = bool(random_dealer_vul)
    self.compact_history = bool(compact_history)
    assert self.num_players == 4 and self.num_actions == bridge.NUM_ACTIONS
    assert self.max_chance_outcomes == NUM_CARDS

  @property
  def declared_rules_sha(self) -> str:
    """The rules identity (module doc, RULES IDENTITY)."""
    if self.random_dealer_vul:
      return RULES_SHA
    return _sha({
        **RULES,
        "dealer_vul": {
            "dealer": self.dealer,
            "dealer_vul": self.dealer_vul,
            "non_dealer_vul": self.non_dealer_vul,
        },
    })

  def info_state(self, s: bridge.GameState, seat) -> dict[str, jax.Array]:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    seat = jnp.asarray(seat, jnp.int32)
    deal = s.deal.astype(jnp.int32)
    shown = bridge.dummy_exposed(s)
    out = {
        "seat": seat.astype(jnp.int8),
        "hand": deal == seat,
        "dummy": (deal == bridge.dummy(s)) & shown,
        "dummy_shown": shown,
        "dealer": s.dealer.astype(jnp.int8),
        "vul": s.vul.astype(jnp.bool_),
    }
    if self.compact_history:
      out.update(compact_history(s))
    else:
      out["actions"], out["num_actions"] = bridge.action_history(s)
    return out

  def labels(self, s: bridge.GameState, seat) -> Mapping[str, jax.Array]:
    """Privileged targets for learners (e.g. the opponent's hand).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      A mapping of label name to array.
    """
    del seat
    return {"deal_holder": s.deal.astype(jnp.int8)}


def hide_hidden_cards(
    env: BridgeEnv, s: bridge.GameState, seat, key
) -> bridge.GameState:
  """Privacy hook: re-deals what `seat` cannot see.

  Privacy hook (check_env and the privacy tests): re-deals what `seat` cannot
  see.

  The unplayed cards of the hands hidden from `seat` (every hand but its own,
  and but dummy's once dummy is face up) are shuffled among those hands, each
  keeping its count. Played cards stay with the seat that played them (that is
  public), and so does everything `seat` may know.

  Args:
    env: The env.
    s: The engine state.
    seat: The seat whose InfoState must not change.
    key: PRNG key.

  Returns:
    The re-dealt state.
  """
  del env
  seat = jnp.asarray(seat, jnp.int32)
  deal = s.deal.astype(jnp.int32)
  shown_dummy = jnp.where(bridge.dummy_exposed(s), bridge.dummy(s), -1)
  hidden = (s.holder >= 0) & (deal != seat) & (deal != shown_dummy)
  idx = jnp.arange(NUM_CARDS, dtype=jnp.int32)
  # src lists the hidden cards in id order, dst in a random order; both then
  # list the visible cards in id order, so those map to themselves.
  src = jnp.argsort(jnp.where(hidden, idx, NUM_CARDS + idx))
  u = jax.random.uniform(key, (NUM_CARDS,), dtype=jnp.float32)
  dst = jnp.argsort(jnp.where(hidden, u, 2.0 + idx.astype(jnp.float32)))
  holder = s.holder.at[dst].set(s.holder[src])
  new_deal = s.deal.at[dst].set(s.deal[src])
  return s._replace(holder=holder, deal=new_deal)


def options_check(kw: Mapping[str, Any]) -> dict[str, Any]:
  """The canonical options of an EnvSpec (envs.registry canon).

  Args:
    kw: The env options.

  Returns:
    The canonical options.

  Raises:
    TypeError: A flag is not a bool.
    ValueError: A dealer outside 0..3, or a fixed board with random_dealer_vul.
  """
  dealer = kw["dealer"]
  if (
      isinstance(dealer, bool)
      or not isinstance(dealer, int)
      or not 0 <= dealer < 4
  ):
    raise ValueError(f"bridge dealer must be an int 0..3, got {dealer!r}")
  out = {"dealer": int(dealer)}
  for k in (
      "dealer_vul",
      "non_dealer_vul",
      "random_dealer_vul",
      "compact_history",
  ):
    if not isinstance(kw[k], bool):
      raise TypeError(f"bridge {k} must be a bool, got {kw[k]!r}")
    out[k] = kw[k]
  if out["random_dealer_vul"] and (
      out["dealer"] or out["dealer_vul"] or out["non_dealer_vul"]
  ):
    raise ValueError(
        "bridge random_dealer_vul draws the dealer and vulnerability; leave "
        "dealer, dealer_vul and non_dealer_vul at their defaults"
    )
  return out


def covers(trained: Mapping[str, Any], served: Mapping[str, Any]) -> bool:
  """Whether a network trained on `trained` plays the served board.

  A network trained on random dealers and vulnerability plays every board;
  one trained on a fixed board plays that board only. compact_history is the
  InfoState's storage form, not the game: the encoder reads both.

  Args:
    trained: The training env's options.
    served: The served env's options.

  Returns:
    True when the network plays every board of `served`.
  """

  def game_opts(kw):
    """The game options other than compact_history.

    Args:
      kw: The options.

    Returns:
      The options.
    """
    return {k: v for k, v in kw.items() if k != "compact_history"}

  return bool(trained["random_dealer_vul"]) or game_opts(trained) == game_opts(
      served
  )


# -- deal identity (sources.deals)
# -------------------------------------------------
def bridge_hand_masks(deal_holder) -> np.ndarray:
  """The hand of each seat 0-3 as a 52-bit card mask.

  uint64[N, 4]: the hand of each seat 0-3 as a 52-bit card mask, from
  deal_holder int8[N, 52] (the seat dealt each card). Raises unless every
  row is a deal: holders in 0-3, 13 cards each.

  Args:
    deal_holder: int8[N, 52] the seat dealt each card.

  Returns:
    uint64[N, 4] hand masks.

  Raises:
    ValueError: A row is not a deal: holders in 0-3, 13 cards each.
  """
  h = np.asarray(deal_holder)
  if h.ndim != 2 or h.shape[1] != 52:
    raise ValueError(f"deal_holder must be [N, 52], got {h.shape}")
  h = h.astype(np.int64)
  if h.size and (h.min() < 0 or h.max() > 3):
    raise ValueError(f"deal_holder holds seats outside 0-3: {np.unique(h)}")
  onehot = h[:, :, None] == np.arange(4)[None, None, :]  # [N, 52, 4]
  counts = onehot.sum(axis=1)
  if (counts != 13).any():
    bad = np.flatnonzero((counts != 13).any(axis=1))
    raise ValueError(
        f"{bad.size} rows of deal_holder are not deals of 13 cards per seat "
        f"(first {bad[:5]}: {counts[bad[:5]].tolist()})"
    )
  bits = np.left_shift(np.uint64(1), np.arange(52, dtype=np.uint64))  # [52]
  return (onehot.astype(np.uint64) * bits[None, :, None]).sum(
      axis=1, dtype=np.uint64
  )


def deal_canon(info: Any, labels: Mapping[str, Any]) -> np.ndarray:
  """uint8[N, 32]: the canonical form of each row's deal (sources.deals).

  The four hands as 52-bit card masks (bit c set when card c is in the
  hand; uint64 little-endian), sorted, from labels["deal_holder"] (int8[52],
  the seat dealt each card). Sorting drops the seat names, so a deal, its
  rotations and every permutation of its seats share one hash. The label
  holds the whole deal on every ply (`every_ply`).

  Args:
    info: Unused InfoState batch.
    labels: The labels, with "deal_holder".

  Returns:
    uint8[N, 32] canonical deals.
  """
  del info
  masks = np.sort(bridge_hand_masks(labels["deal_holder"]), axis=1)
  return (
      np.ascontiguousarray(masks.astype("<u8"))
      .view(np.uint8)
      .reshape(masks.shape[0], 32)
  )


deal_canon.every_ply = True

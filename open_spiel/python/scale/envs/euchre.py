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

"""Euchre Env over the JAX engine `euchre`: two partnerships of two.

Seats 0..3 (North, East, South, West; North-South against East-West), 31
actions with OpenSpiel's ids (0..23 cards, to discard or play; 24 pass;
25..28 name clubs, diamonds, hearts, spades; 29 go alone; 30 play with
partner), zero-sum returns: the makers' score (1, 2 or 4; -2 or -4 when
euchred) for both makers, its negative for the defenders. One hand is one
episode: chance draws the dealer, deals 20 cards and turns the up-card, then
at most 8 bids, the go-alone decisions, the dealer's discard and 20 plays.
The rules are OpenSpiel's corrected euchre (the games package's euchre module
doc); the options are its two game parameters at their defaults (``OPTIONS``).

INFOSTATE. One seat's view, a dict with these keys and dtypes
(``info_template``):

  seat           int8      the observing seat
  dealer         int8      the dealer
  upcard         int8      the up-card (face up for everyone)
  num_passes     int8      the passes made
  trump          int8      the trump suit, -1 until named
  declarer       int8      the seat that named trump, -1 until then
  go_alone       int8      the declarer's decision: 1 alone, 0 with partner,
                           -1 undecided
  lone_defender  int8      the defender playing alone, else -1
  discard        int8      the dealer's discard, in the dealer's view only
                           (-1 for every other seat, and before the discard)
  hand           bool[24]  the cards the seat holds now
  play           int8[20]  the cards in order of play; -1 pad
  play_seat      int8[20]  who played each of them; -1 pad
  n_played       int8      the cards played

That is exactly what OpenSpiel's information-state tensor holds (bids are
made in turn from the dealer's left, so the passes and the declarer fix the
auction), and the encoder ``euchre_info_state`` (encoders.euchre) rebuilds
that tensor from it bit for bit. The other hands are never in it.
``labels`` carries the deal (``deal_holder``, the seat dealt each card).
Every action is public except the dealer's discard (``visible_action`` is -1
for it from every seat but the dealer's). ``world_from_info`` is None;
``hide_hidden_cards`` is the privacy hook.

SIDES. The two partnerships (``Sides.partnerships(4, 2, constant_sum=True)``:
North-South {0, 2}, East-West {1, 3}); a side's return is its seats'.

RULES IDENTITY. Declared (``declared_rules_sha``): the sha256 over canonical
JSON of RULES with the two parameters, so the hash names OpenSpiel's
corrected euchre at those parameters, not a source file. The reference
engine is pinned by the games recorded from it in the JAX game's
``euchre_golden`` module (``REFERENCE``), which the tests replay without
that build; a reference change that changes the rules fails the replay, and
its re-record bumps RULES ``schema`` (so the hash) in the same change.

SERVING. Served (envs.registry: EUCHRE_PYSPIEL_PARAMS) at both values of
the two parameters; a pyspiel history is the raw history. Only a pyspiel
whose euchre is the corrected engine's (a 959-entry tensor) is served; a
released pyspiel's euchre plays other rules and is refused.
"""

import hashlib
import json
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import euchre
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.envs import jax_game_env

NUM_CARDS = euchre.NUM_CARDS
NUM_ACTIONS = euchre.NUM_DISTINCT_ACTIONS
NUM_PLAYERS = euchre.NUM_PLAYERS
NUM_PLAYS = euchre.NUM_HAND_CARDS
#: The most player decisions in a hand at any options (31 with
#: allow_lone_defender; a hand with a discard is shorter).
MAX_LEN = euchre.max_game_length(allow_lone_defender=True)
#: OpenSpiel euchre's game parameters at their defaults (every key is an
#: option).
OPTIONS = {"allow_lone_defender": False, "stick_the_dealer": True}
assert tuple(OPTIONS) == euchre.PARAMS
RULES = {
    "schema": "euchre-rules/1",
    "engine": "open_spiel euchre (euchre.cc), corrected: 959-entry tensor",
    "game": "euchre",
    "scoring": (
        "makers 1 for 3-4 tricks, 2 for 5 (4 alone), -2 euchred (-4 against "
        "a lone defender); go-alone decided before the dealer's pick-up"
    ),
}
#: North-South against East-West, zero-sum.
SIDES = sides_lib.Sides.partnerships(NUM_PLAYERS, 2, constant_sum=True)


def rules_sha(options: Mapping[str, bool]) -> str:
  """The declared rules identity of euchre at `options` (module doc).

  Args:
    options: the two game parameters.

  Returns:
    The sha256 hex digest of RULES with the parameters, as canonical JSON.
  """
  rules = {**RULES, "params": {k: bool(options[k]) for k in OPTIONS}}
  return hashlib.sha256(
      json.dumps(rules, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()


def info_template() -> dict[str, np.ndarray]:
  """One seat's InfoState (unbatched) as zero arrays.

  Of its shapes and dtypes.

  Returns:
    {key: zero array}.
  """
  scalars = (
      "seat",
      "dealer",
      "upcard",
      "num_passes",
      "trump",
      "declarer",
      "go_alone",
      "lone_defender",
      "discard",
  )
  return {
      **{k: np.zeros((), np.int8) for k in scalars},
      "hand": np.zeros((NUM_CARDS,), np.bool_),
      "play": np.full((NUM_PLAYS,), -1, np.int8),
      "play_seat": np.full((NUM_PLAYS,), -1, np.int8),
      "n_played": np.zeros((), np.int8),
  }


class EuchreEnv(jax_game_env.JaxGameEnv):
  """Euchre: 4 seats in 2 partnerships, 31 actions, zero-sum (module doc)."""

  name = "euchre"
  max_len = MAX_LEN
  zero_sum = True
  sides = SIDES

  def __init__(self, **options):
    unknown = sorted(set(options) - set(OPTIONS))
    if unknown:
      raise TypeError(
          f"euchre: unknown options {unknown}; the options are "
          f"{sorted(OPTIONS)}"
      )
    self.options = {**OPTIONS, **options}
    # registry.spec_of reads the options as attributes
    for k, v in self.options.items():
      setattr(self, k, v)
    super().__init__(euchre.Euchre(**self.options))
    assert self.num_players == NUM_PLAYERS and self.num_actions == NUM_ACTIONS
    assert self.max_chance_outcomes == NUM_CARDS

  @property
  def declared_rules_sha(self) -> str:
    """The rules identity: rules_sha of the options (module doc)."""
    return rules_sha(self.options)

  def info_state(self, s: euchre.GameState, seat) -> dict[str, jax.Array]:
    """One seat's InfoState (module doc, INFOSTATE).

    Args:
      s: The state.
      seat: The observing seat.

    Returns:
      {key: array}.
    """
    seat = jnp.asarray(seat, jnp.int32)

    def i8(x):
      """`x` as int8.

      Args:
        x: An array.

      Returns:
        The int8 array.
      """
      return jnp.asarray(x, jnp.int8)

    return {
        "seat": i8(seat),
        "dealer": i8(s.dealer),
        "upcard": i8(s.upcard),
        "num_passes": i8(s.num_passes),
        "trump": i8(s.trump),
        "declarer": i8(s.declarer),
        "go_alone": i8(s.go_alone),
        "lone_defender": i8(s.lone_defender),
        "discard": i8(jnp.where(seat == s.dealer, s.discard, -1)),
        "hand": s.holder == seat,
        "play": i8(s.play),
        "play_seat": i8(s.play_seat),
        "n_played": i8(s.n_played),
    }

  def labels(self, s: euchre.GameState, seat) -> Mapping[str, jax.Array]:
    """The hidden-information labels: the deal.

    Args:
      s: The state.
      seat: Unused (the labels are the same for every seat).

    Returns:
      {"deal_holder": int8[24]}, the seat dealt each card (-1 for the
      up-card and the three cards never dealt).
    """
    del seat
    return {"deal_holder": s.initial_deal.astype(jnp.int8)}

  def visible_action(self, s: euchre.GameState, a, seat) -> jax.Array:
    """The dealer's discard is seen by the dealer only; the rest by all.

    Args:
      s: The state before `a`.
      a: The action.
      seat: The observing seat.

    Returns:
      i32: `a`, or -1 for the discard seen from another seat.
    """
    discarding = s.phase == euchre.PHASE_DISCARD
    seen = ~discarding | (jnp.asarray(seat, jnp.int32) == s.current_player)
    return jnp.where(seen, jnp.asarray(a, jnp.int32), jnp.int32(-1))


def hide_hidden_cards(
    env: EuchreEnv, s: euchre.GameState, seat, key
) -> euchre.GameState:
  """Privacy hook (check_env and the privacy tests).

  Re-deals what `seat` cannot see: the cards the other seats hold (except
  the up-card once the dealer has taken it: everyone saw it), the three
  cards never dealt, and the dealer's discard (unless `seat` is the dealer)
  are shuffled among those places, each place keeping its count. Each card
  carries its own deal record, so the deal moves with it.

  Args:
    env: Unused.
    s: The state.
    seat: The seat whose view is kept.
    key: The PRNG key of the shuffle.

  Returns:
    The state with the hidden cards re-dealt.
  """
  del env
  seat = jnp.asarray(seat, jnp.int32)
  idx = jnp.arange(NUM_CARDS, dtype=jnp.int32)
  t = jnp.arange(NUM_PLAYS)
  played_ids = jnp.where(
      t < s.n_played.astype(jnp.int32), s.play.astype(jnp.int32), NUM_CARDS
  )
  played = (
      jnp.zeros((NUM_CARDS,), jnp.bool_).at[played_ids].set(True, mode="drop")
  )
  holder = s.holder.astype(jnp.int32)
  upcard = s.upcard.astype(jnp.int32)
  discard = s.discard.astype(jnp.int32)
  own_discard = (seat == s.dealer.astype(jnp.int32)) & (idx == discard)
  hidden = (holder != seat) & ~played & (idx != upcard) & ~own_discard
  # An "is the discard" mark travels with the card.
  is_discard = (idx == discard) & (discard >= 0)
  src = jnp.argsort(jnp.where(hidden, idx, NUM_CARDS + idx))
  u = jax.random.uniform(key, (NUM_CARDS,), dtype=jnp.float32)
  dst = jnp.argsort(jnp.where(hidden, u, 2.0 + idx.astype(jnp.float32)))
  new_discard_mark = is_discard.at[dst].set(is_discard[src])
  new_discard = jnp.where(
      discard >= 0, jnp.argmax(new_discard_mark), discard
  ).astype(jnp.int8)
  return s._replace(
      holder=s.holder.at[dst].set(s.holder[src]),
      initial_deal=s.initial_deal.at[dst].set(s.initial_deal[src]),
      discard=new_discard,
  )


def options_check(kw: Mapping[str, Any]) -> dict[str, Any]:
  """The canonical options of an EnvSpec (envs.registry canon).

  Args:
    kw: The options.

  Returns:
    The canonical options.

  Raises:
    TypeError: an option is not a bool.
  """
  out = {}
  for k in OPTIONS:
    if not isinstance(kw[k], bool):
      raise TypeError(f"euchre {k} must be a bool, got {kw[k]!r}")
    out[k] = kw[k]
  return out

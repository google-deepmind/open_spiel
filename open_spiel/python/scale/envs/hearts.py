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

"""Hearts Env over the JAX engine `hearts`: four players, each for themselves.

Seats 0..3 (North, East, South, West), 52 actions with pyspiel's ids (the
card id, to pass or to play), returns 26 - points (general-sum: a moon
changes the total). One hand is one episode: chance draws the pass direction
and deals, then 12 passes (unless the direction is "no pass") and 52 plays,
so ``max_len`` is 64 player decisions. The options are pyspiel hearts' eight
game parameters at pyspiel's defaults (``OPTIONS``).

INFOSTATE. One seat's view, a dict with these keys and dtypes
(``info_template``):

  seat       int8       the observing seat
  pass_dir   int8       0 no pass, 1 left, 2 across, 3 right
  dealt      bool[52]   the 13 cards the seat was dealt
  passed     bool[52]   the cards the seat has passed
  received   bool[52]   the cards passed to it, once every seat has passed
  hand       bool[52]   the cards it holds now
  points     int16[4]   each seat's points so far (public: tricks are face up)
  play       int8[52]   the cards in order of play; -1 pad
  play_seat  int8[52]   who played each of them; -1 pad
  n_played   int8       the cards played

That is exactly what pyspiel's information-state tensor holds, and the
encoder ``hearts_info_state`` (encoders.hearts) rebuilds that tensor from it
bit for bit. The other hands are never in it. ``labels`` carries the deal
(``deal_holder``, the seat dealt each card). Every action is public except a
pass, which only the passer and (once the passes are done) the receiver see;
``visible_action`` hides a pass from everyone but the passer (-1).
``world_from_info`` is None; ``hide_hidden_cards`` is the privacy hook.

SIDES. Every seat is its own side, not constant-sum
(``Sides.individual(4, constant_sum=False)``): a league seats four entrants
(or one entrant and three draws from a field), and a seat's return is its
side's.

RULES IDENTITY. Declared (``declared_rules_sha``): the sha256 over canonical
JSON of RULES with the eight parameters, so the hash names pyspiel's hearts
at those parameters, not a source file.

SERVING. Hearts is served (its GameSpec's ServingSpec): pyspiel's hearts at
any of its eight game parameters plays through serve (PolicyServer, the GUI's
checkpoint seat, the Remote bot) with an artifact trained on the same
parameters. A pyspiel history is the env's raw history, no translation.
"""

import hashlib
import json
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import hearts
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.envs import jax_game_env

NUM_CARDS = hearts.NUM_CARDS
NUM_PLAYERS = hearts.NUM_PLAYERS
MAX_LEN = hearts.MAX_GAME_LENGTH  # 64
#: pyspiel hearts' game parameters at their defaults (every key is an option).
OPTIONS = {
    "pass_cards": True,
    "no_pts_on_first_trick": True,
    "can_lead_any_club": False,
    "jd_bonus": False,
    "avoid_all_tricks_bonus": False,
    "must_break_hearts": True,
    "qs_breaks_hearts": True,
    "can_lead_hearts_instead_of_qs": False,
}
assert tuple(OPTIONS) == hearts.PARAMS
RULES = {
    "schema": "hearts-rules/1",
    "engine": "open_spiel hearts (hearts.cc)",
    "game": "hearts",
    "scoring": "returns 26 - points; moon by pyspiel's per-trick test",
}
#: Free-for-all: one side per seat, general-sum.
SIDES = sides_lib.Sides.individual(NUM_PLAYERS, constant_sum=False)


def rules_sha(options: Mapping[str, bool]) -> str:
  """The declared rules identity of hearts at `options` (module doc).

  Args:
    options: the eight game parameters.

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

  def cards():
    """A zero card set.

    Returns:
      bool[52].
    """
    return np.zeros((NUM_CARDS,), np.bool_)

  return {
      "seat": np.zeros((), np.int8),
      "pass_dir": np.zeros((), np.int8),
      "dealt": cards(),
      "passed": cards(),
      "received": cards(),
      "hand": cards(),
      "points": np.zeros((NUM_PLAYERS,), np.int16),
      "play": np.full((NUM_CARDS,), -1, np.int8),
      "play_seat": np.full((NUM_CARDS,), -1, np.int8),
      "n_played": np.zeros((), np.int8),
  }


class HeartsEnv(jax_game_env.JaxGameEnv):
  """Hearts: 4 seats, 52 actions, returns 26 - points (module doc)."""

  name = "hearts"
  max_len = MAX_LEN
  zero_sum = False
  sides = SIDES

  def __init__(self, **options):
    unknown = sorted(set(options) - set(OPTIONS))
    if unknown:
      raise TypeError(
          f"hearts: unknown options {unknown}; the options are "
          f"{sorted(OPTIONS)}"
      )
    self.options = {**OPTIONS, **options}
    # registry.spec_of reads the options as attributes
    for k, v in self.options.items():
      setattr(self, k, v)
    super().__init__(hearts.Hearts(**self.options))
    assert self.num_players == NUM_PLAYERS and self.num_actions == NUM_CARDS
    assert self.max_chance_outcomes == NUM_CARDS

  @property
  def declared_rules_sha(self) -> str:
    """The rules identity: rules_sha of the options (module doc)."""
    return rules_sha(self.options)

  def info_state(self, s: hearts.GameState, seat) -> dict[str, jax.Array]:
    """One seat's InfoState (module doc, INFOSTATE).

    Args:
      s: The state.
      seat: The observing seat.

    Returns:
      {key: array}.
    """
    seat = jnp.asarray(seat, jnp.int32)
    passed = (
        jnp.zeros((NUM_CARDS,), jnp.bool_)
        .at[
            jnp.where(
                s.passed[seat] >= 0, s.passed[seat].astype(jnp.int32), NUM_CARDS
            )
        ]
        .set(True, mode="drop")
    )
    return {
        "seat": seat.astype(jnp.int8),
        "pass_dir": s.pass_dir.astype(jnp.int8),
        "dealt": s.initial_deal == seat,
        "passed": passed,
        "received": hearts.received_cards(s, seat),
        "hand": s.holder == seat,
        "points": s.points.astype(jnp.int16),
        "play": s.play.astype(jnp.int8),
        "play_seat": s.play_seat.astype(jnp.int8),
        "n_played": s.n_played.astype(jnp.int8),
    }

  def labels(self, s: hearts.GameState, seat) -> Mapping[str, jax.Array]:
    """The hidden-information labels: the deal.

    Args:
      s: The state.
      seat: Unused (the labels are the same for every seat).

    Returns:
      {"deal_holder": int8[52]}, the seat dealt each card.
    """
    del seat
    return {"deal_holder": s.initial_deal.astype(jnp.int8)}

  def visible_action(self, s: hearts.GameState, a, seat) -> jax.Array:
    """A pass is seen by the passer only (-1 for the others); a play by all.

    Args:
      s: The state before `a`.
      a: The action.
      seat: The observing seat.

    Returns:
      i32: `a`, or -1 for a pass seen from another seat.
    """
    passing = s.phase == hearts.PHASE_PASS
    seen = ~passing | (jnp.asarray(seat, jnp.int32) == s.current_player)
    return jnp.where(seen, jnp.asarray(a, jnp.int32), jnp.int32(-1))


def hide_hidden_cards(
    env: HeartsEnv, s: hearts.GameState, seat, key
) -> hearts.GameState:
  """Privacy hook (check_env and the privacy tests).

  Re-deals what `seat` cannot see. The cards held by the other seats that
  `seat` cannot place are shuffled among those seats, each keeping its count:
  every card another seat holds, except a card that was passed (its passer,
  and after the passes its receiver, know where it is) and 2C (its holder
  leads it). Those cards never moved since the deal, so the deal moves with
  them.

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
  flat = s.passed.reshape(-1).astype(jnp.int32)
  was_passed = (
      jnp.zeros((NUM_CARDS,), jnp.bool_)
      .at[jnp.where(flat >= 0, flat, NUM_CARDS)]
      .set(True, mode="drop")
  )
  holder = s.holder.astype(jnp.int32)
  hidden = (
      (holder >= 0)
      & (holder != seat)
      & ~was_passed
      & (idx != hearts.TWO_OF_CLUBS)
  )
  src = jnp.argsort(jnp.where(hidden, idx, NUM_CARDS + idx))
  u = jax.random.uniform(key, (NUM_CARDS,), dtype=jnp.float32)
  dst = jnp.argsort(jnp.where(hidden, u, 2.0 + idx.astype(jnp.float32)))
  return s._replace(
      holder=s.holder.at[dst].set(s.holder[src]),
      initial_deal=s.initial_deal.at[dst].set(s.initial_deal[src]),
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
      raise TypeError(f"hearts {k} must be a bool, got {kw[k]!r}")
    out[k] = kw[k]
  return out

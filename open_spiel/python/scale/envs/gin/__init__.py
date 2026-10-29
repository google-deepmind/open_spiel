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

"""Gin rummy as an Env, over the JAX gin_rummy game engine.

One episode is one hand; returns are the engine's hand scores. Every decision
of both seats is a decision of this Env, the KNOCK / LAYOFF / WALL endgame
included; a league routes those to a scripted endgame policy when its config
asks for one (GameSpec.endgame).

  from open_spiel.python.scale.envs import registry
  env = registry.make("gin_rummy", oklahoma_frac=1.0)
  s = env.init(jax.random.key(0))
  info = env.info_state(s, 0)                   # GinInfo of seat 0

Options (each an EnvSpec option; defaults are OpenSpiel's):
  oklahoma_frac: None deals standard gin; p in [0, 1] makes each deal
    Oklahoma (knock limit = the upcard's value) with probability p.
  allow_upcard_throwback: the upcard taken this turn may be discarded.
  wall_knock_pick: in standard gin, the player who draws the 50th card may
    still pick up and knock at the wall.
  spade_doubles: in Oklahoma, a spade initial upcard doubles the hand score
    (requires oklahoma_frac=1.0).

Chance (raw layer). Every chance event is explicit, in this order:
  1. the variant node, only when 0 < oklahoma_frac < 1: outcome 1 is an
     Oklahoma deal, with probability oklahoma_frac;
  2. the first-player node: outcome 0 or 1, probability 1/2 each;
  3. the 21 deal cards (10 to seat 0, 10 to seat 1, the upcard; each outcome
     is a card id, uniform over the deck);
  4. every stock draw (a card id, uniform over the stock).
The first player is installed when the deal completes, and the engine's meld
cache is computed for that player's hand.

InfoState (`GinInfo`): the seat's own hand, the public piles and bookkeeping
(discard pile, upcard, the cards each seat is publicly known to hold, laid
melds and layoffs, phase, knock card, stock count, scores) and the hand's
event transcript (who passed, took, drew, discarded and knocked; the card the
seat itself drew from the stock). It never holds the opponent's hidden cards,
the stock's contents or order, or engine caches computed from the hand of the
player to move. `labels()` carries the opponent's hand for learners.

Engine. Two engines of the JAX gin_rummy game step the same game:
"reference" (gin_rummy.game) and "dense_cache" (the same step with its meld
caches computed by dense arithmetic instead of a large table gather, which
suits accelerators that gather slowly). The engine is a constructor argument,
not an env option: the EnvSpec, the rules identity and everything a seat or
a learner sees are the same for both. `set_default_engine` picks the engine
of envs built without one.
"""

import functools
from typing import Any, Mapping, NamedTuple, Optional

import jax
from jax import custom_batching
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.jax.games.gin_rummy import deadwood
from open_spiel.python.jax.games.gin_rummy import dense_cache
from open_spiel.python.jax.games.gin_rummy import dense_cache_game
from open_spiel.python.jax.games.gin_rummy import game as gin_game
from open_spiel.python.scale.envs import jax_game_env

NUM_CARDS = cm.NUM_CARDS
NUM_MELDS = cm.NUM_MELDS

#: The rule options of the env and their defaults (OpenSpiel's gin_rummy).
DEFAULT_RULES = {
    "allow_upcard_throwback": True,
    "wall_knock_pick": True,
    "spade_doubles": False,
}

# -- the event transcript ------------------------------------------------------
#: The most events one hand can hold: 1 pass + 2 x (1 take + 29 stock draws +
#: 49 upcard draws) + 1 wall knock.
EVENTS_MAX = (
    1
    + 2
    * (
        1
        + (NUM_CARDS - 2 * cm.HAND_SIZE - 1 - cm.WALL_STOCK_SIZE)
        + cm.MAX_NUM_DRAW_UPCARD_ACTIONS
        - 1
    )
    + 1
)
assert EVENTS_MAX == 160, EVENTS_MAX
(
    EVT_PAD,
    EVT_PASS_UPCARD,
    EVT_DRAW_UPCARD,
    EVT_DRAW_STOCK,
    EVT_DISCARD,
    EVT_KNOCK,
) = range(6)
ERR_OVERFLOW = 1  # an event arrived with the buffer full
ERR_PRIVATE = 2  # a stock card was dealt with no matching stock draw
#: GinInfo.ev_actor codes (relative to the observing seat).
ACTOR_PAD, ACTOR_SELF, ACTOR_OPP = 0, 1, 2

#: Max player decisions in one hand: every event is one decision (160); the
#: other decisions are the wall pass (1), the knocker's knock-phase discard,
#: <= 3 melds and pass (5), and the defender's layoffs, melds and two passes,
#: holding at most 10 cards (<= 10 + 2 + 3 = 15).
MAX_LEN = EVENTS_MAX + 1 + 5 + 15

STAGE_VARIANT, STAGE_FIRST_PLAYER, STAGE_ENGINE = 0, 1, 2

ENGINES = ("reference", "dense_cache")
_DEAL_CARDS = 2 * cm.HAND_SIZE + 1  # 21


class Events(NamedTuple):
  """The hand's event buffer, in absolute seats (GinInfo makes them relative).

  Attributes:
    actor: i8[160] absolute seat (0 on pad rows too: use count).
    type: i8[160] EVT_*.
    public_card: i8[160] card + 1 (0 = none).
    private_card: i8[160] card + 1 of the drawer's stock card.
    count: i32.
    error: i32 ERR_* bits.
  """

  actor: jax.Array  # i8[160] absolute seat (0 on pad rows too: use count)
  type: jax.Array  # i8[160] EVT_*
  public_card: jax.Array  # i8[160] card + 1 (0 = none)
  private_card: jax.Array  # i8[160] card + 1 of the drawer's stock card
  count: jax.Array  # i32
  error: jax.Array  # i32 ERR_* bits


class GinState(NamedTuple):
  """The env state: the engine's state plus the env's own stages and events.

  Attributes:
    game: The JAX gin_rummy game's state dict.
    stage: i32 STAGE_*.
    first_player: i32, valid from STAGE_ENGINE on.
    events: The hand's event buffer.
  """

  game: Mapping[str, jax.Array]  # the JAX gin_rummy game's state dict
  stage: jax.Array  # i32 STAGE_*
  first_player: jax.Array  # i32, valid from STAGE_ENGINE on
  events: Events


class GinInfo(NamedTuple):
  """What `seat` knows. Dtypes are part of the contract (x64-proof).

  Attributes:
    seat: i32.
    hand: i8[52] own hand.
    discard_pile: i8[52] (the upcard is not in it).
    upcard: i32, -1 = none.
    known_cards: i8[2, 52] per seat: cards it took from the pile and holds.
    phase: i32 PHASE_*.
    current_player: i32 engine value (-1 chance, -4 terminal).
    knock_card: i32 the hand's knock limit.
    oklahoma: bool.
    stock_count: i32.
    pass_count: i32 first-upcard passes.
    num_draw_upcard: i32 DRAW-phase upcard draws.
    repeated_move: bool.
    upcard_taken: i32 card taken from the upcard this turn (-1 when throwback is
      allowed).
    allow_upcard_throwback: bool.
    layed_melds: bool[2, 185] per seat.
    layoffs_mask: bool[52].
    knocker: i32, -1 = none.
    finished_layoffs: bool.
    knocker_gin: bool: the knocker laid gin (public: its melds cover 10 cards).
    done: bool.
    scores: i32[2] final scores (0 until the hand ends).
    ev_actor: i8[160] ACTOR_* (observer-relative).
    ev_type: i8[160] EVT_*.
    ev_public: i8[160] card + 1.
    ev_private: i8[160] card + 1 on the observer's own stock draws only.
    ev_count: i32 events so far (<= 160; per-hand index, not a counter).
    ev_error: i32 ERR_* bits of the buffer.
  """

  seat: jax.Array
  hand: jax.Array
  discard_pile: jax.Array
  upcard: jax.Array
  known_cards: jax.Array
  phase: jax.Array
  current_player: jax.Array
  knock_card: jax.Array
  oklahoma: jax.Array
  stock_count: jax.Array
  pass_count: jax.Array
  num_draw_upcard: jax.Array
  repeated_move: jax.Array
  upcard_taken: jax.Array
  allow_upcard_throwback: jax.Array
  layed_melds: jax.Array
  layoffs_mask: jax.Array
  knocker: jax.Array
  finished_layoffs: jax.Array
  knocker_gin: jax.Array
  done: jax.Array
  scores: jax.Array
  ev_actor: jax.Array
  ev_type: jax.Array
  ev_public: jax.Array
  ev_private: jax.Array
  ev_count: jax.Array
  ev_error: jax.Array


def is_endgame_phase(phase) -> jax.Array:
  """Whether the phase is one a scripted endgame plays.

  Args:
    phase: An engine phase (array or int).

  Returns:
    bool: the phase is KNOCK, LAYOFF or WALL.
  """
  phase = jnp.asarray(phase)
  return (
      (phase == cm.PHASE_KNOCK)
      | (phase == cm.PHASE_LAYOFF)
      | (phase == cm.PHASE_WALL)
  )


def _empty_events() -> Events:
  """An empty event log.

  Returns:
    The Events.
  """
  z = jnp.zeros(EVENTS_MAX, jnp.int8)
  return Events(z, z, z, z, jnp.int32(0), jnp.int32(0))


def record_event(buf: Events, g, action, active) -> Events:
  """Records `action` taken at the pre-step engine state `g`.

  Public events: a FIRST_UPCARD pass, an upcard take, a stock draw (its
  public card is the declined upcard), a DISCARD-phase discard, a knock
  (DISCARD or WALL). The chance step that deals a stock card writes it as the
  private card of the newest event, which must be that seat's stock draw.
  One-hot writes only.

  Args:
    buf: The event buffer before the step.
    g: The engine state before the step.
    action: The action taken.
    active: Whether the step is live (a finished row records nothing).

  Returns:
    The event buffer with the step's event, if any, written.
  """
  phase = g["phase"]
  player = g["current_player"]
  upcard = g["upcard"]
  action = jnp.asarray(action, jnp.int32)
  live = jnp.asarray(active, jnp.bool_) & ~g["done"]

  is_player = player >= 0
  first_up = (phase == cm.PHASE_FIRST_UPCARD) & is_player
  draw = (phase == cm.PHASE_DRAW) & is_player
  discard_phase = (phase == cm.PHASE_DISCARD) & is_player
  wall = (phase == cm.PHASE_WALL) & is_player

  pass_up = first_up & (action == cm.ACTION_PASS)
  take_up = (first_up | draw) & (action == cm.ACTION_DRAW_UPCARD)
  draw_stock = (first_up | draw) & (action == cm.ACTION_DRAW_STOCK)
  discard = discard_phase & (action < NUM_CARDS)
  knock = (discard_phase | wall) & (action == cm.ACTION_KNOCK)

  evt_type = jnp.select(
      [pass_up, take_up, draw_stock, discard, knock],
      [
          jnp.int32(EVT_PASS_UPCARD),
          jnp.int32(EVT_DRAW_UPCARD),
          jnp.int32(EVT_DRAW_STOCK),
          jnp.int32(EVT_DISCARD),
          jnp.int32(EVT_KNOCK),
      ],
      default=jnp.int32(EVT_PAD),
  )
  upcard_code = jnp.maximum(upcard.astype(jnp.int32) + 1, 0)
  public_code = jnp.select(
      [pass_up | take_up | draw_stock, discard],
      [upcard_code, action + jnp.int32(1)],
      default=jnp.int32(0),
  )

  count = buf.count
  slots = jnp.arange(EVENTS_MAX, dtype=jnp.int32)
  is_event = live & (evt_type != EVT_PAD)
  overflow = is_event & (count >= EVENTS_MAX)
  write = is_event & ~overflow
  at_slot = write & (slots == count)

  def put(arr, value, where):
    """Writes `value` into `arr` where `where`, in arr's dtype.

    Args:
      arr: The buffer column.
      value: The value to write.
      where: bool mask of the slots to write.

    Returns:
      The updated column.
    """
    return jnp.where(where, jnp.asarray(value).astype(arr.dtype), arr)

  actor = put(buf.actor, player, at_slot)
  evt = put(buf.type, evt_type, at_slot)
  public = put(buf.public_card, public_code, at_slot)
  private = put(buf.private_card, 0, at_slot)

  zero32 = jnp.int32(0)
  drawer = g["prev_player"].astype(jnp.int32)
  stock_deal = live & (phase == cm.PHASE_DEAL) & (drawer >= 0)
  at_last = slots == (count - 1)
  last_type = jnp.sum(
      jnp.where(at_last, buf.type.astype(jnp.int32), zero32), dtype=jnp.int32
  )
  last_actor = jnp.sum(
      jnp.where(at_last, buf.actor.astype(jnp.int32), zero32), dtype=jnp.int32
  )
  last_private = jnp.sum(
      jnp.where(at_last, buf.private_card.astype(jnp.int32), zero32),
      dtype=jnp.int32,
  )
  last_ok = (
      (count > 0)
      & (last_type == EVT_DRAW_STOCK)
      & (last_actor == drawer)
      & (last_private == 0)
  )
  private = put(private, action + 1, stock_deal & last_ok & at_last)

  # Explicit i32 operands, so no operand of `|` is weakly typed.
  zero = jnp.int32(0)
  error = (
      buf.error.astype(jnp.int32)
      | jnp.where(overflow, jnp.int32(ERR_OVERFLOW), zero)
      | jnp.where(stock_deal & ~last_ok, jnp.int32(ERR_PRIVATE), zero)
  )
  return Events(
      actor,
      evt,
      public,
      private,
      (count + write.astype(jnp.int32)).astype(jnp.int32),
      error,
  )


def install_first_player(g, first_player, engine: str = "reference"):
  """The dealt engine state with `first_player` to act.

  The engine's meld cache is recomputed for that player's hand + upcard: the
  engine step's FIRST_UPCARD recompute (the JAX gin_rummy game.py
  `_step_body`) for another player. For first_player 0 it reproduces the
  engine's own deal-completion cache.

  Args:
    g: The dealt engine state.
    first_player: The seat to act first.
    engine: The engine name (ENGINES).

  Returns:
    The engine state with first_player to move and its meld cache.
  """
  p = jnp.asarray(first_player, jnp.int8)
  hand = gin_game.get_hand(g, p)
  upcard = g["upcard"]
  effective = jnp.where(upcard >= 0, hand.at[upcard].set(1), hand)
  knock_limit = g["knock_card"].astype(jnp.int32)
  eng = _engine(engine)
  _, all_dw = eng.all_meld_info(effective)
  legal_knocker = eng.legal_knocker_melds(effective, knock_limit)
  kept = jnp.where(g["allow_upcard_throwback"], jnp.int8(-1), upcard)
  discardable = (effective > 0) & (
      jnp.arange(NUM_CARDS, dtype=jnp.int32) != kept
  )
  legal_discard = discardable & (all_dw <= knock_limit)
  min_dw = jnp.min(jnp.where(discardable, all_dw, jnp.int32(999)))
  out = dict(g)
  out["current_player"] = p
  out["cached_min_deadwood"] = jnp.int32(min_dw)
  out["cached_legal_knocker_melds"] = legal_knocker
  out["cached_legal_discard_mask"] = legal_discard
  out["cached_all_discard_deadwoods"] = all_dw.astype(jnp.int16)
  out["can_knock"] = jnp.any(legal_discard)
  return out


def _swap_hands(g, swap):
  """Swaps the two hands where `swap`.

  Args:
    g: The engine state.
    swap: Whether to swap.

  Returns:
    The engine state.
  """
  out = dict(g)
  out["player0_hand"] = jnp.where(swap, g["player1_hand"], g["player0_hand"])
  out["player1_hand"] = jnp.where(swap, g["player0_hand"], g["player1_hand"])
  return out


def _engine_chance_step(
    g, outcome, completes, first_player, step=gin_game.step
):
  """gin_game.step(g, outcome) on an engine chance node.

  The deal's last card installs `first_player` (=
  `install_first_player(step(g, o), fp)` when `completes`, else `step(g, o)`,
  field for field) at the cost of ONE engine step: the step that completes the
  deal recomputes the meld cache for player 0's hand + upcard, so for
  first_player 1 the two hands are swapped around that step (the cache is then
  player 1's) and swapped back, and the player to move is set. The deal
  touches no other per-seat field (known cards are empty during the deal).

  Args:
    g: The engine state at a chance node.
    outcome: The chance outcome.
    completes: Whether this card completes the deal.
    first_player: The seat to act first.
    step: The engine step.

  Returns:
    The engine state after the chance step.
  """
  swap = completes & (jnp.asarray(first_player, jnp.int32) == 1)
  g2 = _swap_hands(step(_swap_hands(g, swap), outcome), swap)
  g2["current_player"] = jnp.where(
      completes, jnp.asarray(first_player, jnp.int8), g2["current_player"]
  )
  return g2


def _where(pred, a, b):
  """Selects `a` where `pred`, else `b`, leaf by leaf.

  Args:
    pred: The condition.
    a: The tree where it holds.
    b: The tree elsewhere.

  Returns:
    The selected tree.
  """
  return jax.tree_util.tree_map(lambda x, y: jnp.where(pred, x, y), a, b)


def _deal_card_is_fast(g) -> jax.Array:
  """A deal card that does not complete the deal (deal_cards_dealt < 20).

  Args:
    g: The engine state.

  Returns:
    bool[], whether the fast deal path applies.
  """
  return (
      (g["phase"] == cm.PHASE_DEAL)
      & (g["prev_player"] < 0)
      & (g["deal_cards_dealt"] < _DEAL_CARDS - 1)
  )


def _deal_card_fast(g, outcome):
  """gin_game.step(g, outcome) at a _deal_card_is_fast node, field for field.

  The card goes to player 0 (the first 10) or player 1, off the deck, the deal
  count + 1, the chance player still to move. The engine's step changes
  nothing else there (its meld-cache recompute is selected away: the phase
  stays DEAL), and the full step costs 216 meld scenarios plus the
  32K-combination knocker melds.

  Args:
    g: The engine state at a fast deal node.
    outcome: The card dealt.

  Returns:
    The engine state after the deal card.
  """
  c = jnp.clip(jnp.asarray(outcome, jnp.int32), 0, NUM_CARDS - 1)
  n = g["deal_cards_dealt"]
  out = dict(g)
  out["player0_hand"] = jnp.where(
      n < cm.HAND_SIZE, g["player0_hand"].at[c].set(1), g["player0_hand"]
  )
  out["player1_hand"] = jnp.where(
      n >= cm.HAND_SIZE, g["player1_hand"].at[c].set(1), g["player1_hand"]
  )
  out["deck"] = g["deck"].at[c].set(0)
  out["deal_cards_dealt"] = (n + 1).astype(n.dtype)
  out["current_player"] = jnp.asarray(
      cm.CHANCE_PLAYER, g["current_player"].dtype
  )
  return out


def _make_chance_step(step):
  """The engine chance step over the engine `step` (a custom_vmap per engine).

  Args:
    step: The engine step.

  Returns:
    chance_step(g, outcome, completes, first_player).
  """

  def one(g, outcome, completes, first_player):
    """One row's chance step: the fast deal card or the engine step.

    Args:
      g: The engine state.
      outcome: The chance outcome.
      completes: Whether this card completes the deal.
      first_player: The seat to act first.

    Returns:
      The engine state after the chance step.
    """
    return jax.lax.cond(
        _deal_card_is_fast(g),
        lambda: _deal_card_fast(g, outcome),
        lambda: _engine_chance_step(g, outcome, completes, first_player, step),
    )

  @custom_batching.custom_vmap
  def chance_step(g, outcome, completes, first_player):
    """_engine_chance_step, with a fast deal card where it applies.

    The fast deal card is _deal_card_fast. Unbatched: one lax.cond. Vmapped (the
    Arena's deal and chance resolve run every row's chance events in one vmapped
    while loop): the fast card on every row when every row is at a fast node
    (the 20 cards before the deal's last, all rows in step), else the engine
    step on every row with the fast card selected where it applies; a vmapped
    per-row cond would compute the engine step on every row at every deal card.

    Args:
      g: The engine state.
      outcome: The chance outcome.
      completes: Whether this card completes the deal.
      first_player: The seat to act first.

    Returns:
      The engine state after the chance step.
    """
    return one(g, outcome, completes, first_player)

  @chance_step.def_vmap
  def _vmap(axis_size, in_batched, g, outcome, completes, first_player):
    """The batching rule: the fast deal when every row takes it.

    Args:
      axis_size: The batch size.
      in_batched: Which arguments are batched.
      g: The engine states.
      outcome: The chance outcomes.
      completes: Whether the outcome completes the deal.
      first_player: The first player.

    Returns:
      (the batched engine state, its batched flags).
    """

    def bc(x, b):
      """Broadcasts an unbatched argument to the batch.

      Args:
        x: The argument.
        b: Whether it is batched already.

      Returns:
        x with a leading batch axis.
      """
      return x if b else jnp.broadcast_to(x, (axis_size,) + jnp.shape(x))

    g, outcome, completes, first_player = jax.tree_util.tree_map(
        bc, [g, outcome, completes, first_player], list(in_batched)
    )
    fast = jax.vmap(_deal_card_is_fast)(g)

    def only_fast():
      """The fast deal card on every row.

      Returns:
        The batched engine state after the deal card.
      """
      return jax.vmap(_deal_card_fast)(g, outcome)

    def mixed():
      """The engine step on every row, the fast card where it applies.

      Returns:
        The batched engine state after the chance step.
      """
      full = jax.vmap(lambda *a: _engine_chance_step(*a, step))(
          g, outcome, completes, first_player
      )
      return _where_rows(fast, jax.vmap(_deal_card_fast)(g, outcome), full)

    out = jax.lax.cond(jnp.all(fast), only_fast, mixed)
    return out, jax.tree_util.tree_map(lambda _: True, out)

  return chance_step


class _Engine(NamedTuple):
  """One gin engine's functions (module doc, Engine).

  Attributes:
    name: The engine name ("reference" or "dense_cache").
    step: The engine's game step (state, action) -> state.
    all_meld_info: The engine's meld info of a hand (melds, deadwood).
    legal_knocker_melds: The engine's legal knocker melds of a hand.
    chance_step: The one-step chance transition built on `step`.
  """

  name: str
  step: object
  all_meld_info: object
  legal_knocker_melds: object
  chance_step: object


@functools.cache
def _engine(name: str) -> _Engine:
  """The gin engine of the given name (cached).

  Args:
    name: the engine name, one of ENGINES.

  Returns:
    The _Engine.

  Raises:
    ValueError: name is not a known engine.
  """
  if name == "reference":
    return _Engine(
        name,
        gin_game.step,
        deadwood.compute_all_meld_info,
        deadwood.compute_legal_knocker_melds,
        _make_chance_step(gin_game.step),
    )
  if name == "dense_cache":
    return _Engine(
        name,
        dense_cache_game.step,
        dense_cache.compute_all_meld_info_dense_discard_only,
        dense_cache.compute_legal_knocker_melds_selection_matrix,
        _make_chance_step(dense_cache_game.step),
    )
  raise ValueError(f"gin engine {name!r} not in {ENGINES}")


_DEFAULT_ENGINE = {"name": "reference"}


def set_default_engine(name: str) -> None:
  """Sets the engine of GinEnvs built without an `engine` argument.

  Args:
    name: One of ENGINES.

  Raises:
    ValueError: name is not one of ENGINES.
  """
  if name not in ENGINES:
    raise ValueError(f"gin engine {name!r} not in {ENGINES}")
  _DEFAULT_ENGINE["name"] = name


def default_engine() -> str:
  """The engine of GinEnvs built without an `engine` argument.

  Returns:
    The engine name (one of ENGINES).
  """
  return _DEFAULT_ENGINE["name"]


def _where_rows(pred, a, b):
  """Selects rows of `a` where `pred`, else `b`.

  Args:
    pred: bool[B], the rows.
    a: A tree, leaves [B, ...].
    b: Another such tree.

  Returns:
    The selected tree.
  """
  return jax.tree_util.tree_map(
      lambda x, y: jnp.where(
          pred.reshape(pred.shape + (1,) * (x.ndim - 1)), x, y
      ),
      a,
      b,
  )


class GinEnv(jax_game_env.JaxGameEnv):
  """Gin rummy, one hand per episode (module doc).

  Attributes:
    oklahoma_frac: The fraction of Oklahoma deals, in [0, 1], or None (standard
      only).
    allow_upcard_throwback: Whether the upcard taken may be discarded the same
      turn.
    wall_knock_pick: Whether, in standard gin, the player who draws the 50th
      card may still pick up and knock at the wall.
    spade_doubles: Whether, in Oklahoma, a spade initial upcard doubles the
      hand score.
    engine: The engine name (ENGINES).

  Raises:
    ValueError: an option out of range, an unknown engine, or a rule
      combination OpenSpiel does not define (spade_doubles needs every deal
      Oklahoma; wall_knock_pick=False needs every deal standard).
  """

  name = "gin_rummy"
  max_len = MAX_LEN
  zero_sum = True

  def __init__(
      self,
      *,
      oklahoma_frac: Optional[float] = None,
      allow_upcard_throwback: bool = DEFAULT_RULES["allow_upcard_throwback"],
      wall_knock_pick: bool = DEFAULT_RULES["wall_knock_pick"],
      spade_doubles: bool = DEFAULT_RULES["spade_doubles"],
      engine: Optional[str] = None,
  ):
    if oklahoma_frac is not None:
      f = float(oklahoma_frac)
      if not 0.0 <= f <= 1.0:
        raise ValueError(
            f"oklahoma_frac={oklahoma_frac!r} must be None or in [0, 1]"
        )
      oklahoma_frac = f
    self.oklahoma_frac = oklahoma_frac
    self._variant_node = oklahoma_frac is not None and 0.0 < oklahoma_frac < 1.0
    self._fixed_oklahoma = oklahoma_frac is not None and oklahoma_frac == 1.0
    if spade_doubles and not self._fixed_oklahoma:
      raise ValueError(
          "spade_doubles is an Oklahoma rule: it needs oklahoma_frac=1.0"
      )
    if not wall_knock_pick and oklahoma_frac not in (None, 0.0):
      raise ValueError(
          "wall_knock_pick=False is a standard-gin rule: it needs oklahoma_frac"
          " None or 0"
      )
    self.allow_upcard_throwback = bool(allow_upcard_throwback)
    self.wall_knock_pick = bool(wall_knock_pick)
    self.spade_doubles = bool(spade_doubles)
    self.engine = default_engine() if engine is None else engine
    if self.engine not in ENGINES:
      raise ValueError(f"gin engine {self.engine!r} not in {ENGINES}")
    self._eng = _engine(self.engine)
    # self.game is the reference GinRummy for every engine: the rules identity
    # (registry.rules_sha) is the game's, not the engine's.
    super().__init__(
        gin_game.GinRummy(oklahoma=self._fixed_oklahoma, **self.rules)
    )
    assert self.max_chance_outcomes == NUM_CARDS
    assert self.num_actions == cm.NUM_ACTIONS and self.num_players == 2

  @property
  def rules(self) -> dict[str, bool]:
    """The rule options, as the JAX GinRummy game takes them."""
    return {
        "allow_upcard_throwback": self.allow_upcard_throwback,
        "wall_knock_pick": self.wall_knock_pick,
        "spade_doubles": self.spade_doubles,
    }

  @property
  def options(self) -> dict[str, Any]:
    """Every EnvSpec option of this env."""
    return {"oklahoma_frac": self.oklahoma_frac, **self.rules}

  # -- raw layer ---------------------------------------------------------------
  def root(self) -> GinState:
    """The root state: before the variant or first-player chance node.

    Returns:
      The root GinState.
    """
    g = gin_game.init_state(oklahoma=self._fixed_oklahoma, **self.rules)
    stage = STAGE_VARIANT if self._variant_node else STAGE_FIRST_PLAYER
    return GinState(
        game=g,
        stage=jnp.int32(stage),
        first_player=jnp.int32(0),
        events=_empty_events(),
    )

  def apply(self, s: GinState, a) -> GinState:
    """Applies a player action and records its event.

    Args:
      s: The state.
      a: The action.

    Returns:
      The next state.
    """
    a = jnp.asarray(a, jnp.int32)
    events = record_event(s.events, s.game, a, True)
    return s._replace(game=self._eng.step(s.game, a), events=events)

  def apply_chance(self, s: GinState, outcome) -> GinState:
    """Applies a chance outcome (variant, first player or a card).

    Args:
      s: The state at a chance node.
      outcome: The chance outcome.

    Returns:
      The next state.
    """
    o = jnp.asarray(outcome, jnp.int32)
    # Variant node: set the deal's Oklahoma flag before any card is dealt.
    g_var = dict(s.game)
    g_var["oklahoma"] = o == 1
    at_variant = s.stage == STAGE_VARIANT
    s_var = GinState(
        g_var, jnp.int32(STAGE_FIRST_PLAYER), s.first_player, s.events
    )
    # First-player node.
    s_fp = s._replace(stage=jnp.int32(STAGE_ENGINE), first_player=o)
    # Engine chance (deal card or stock draw).
    in_deal = s.game["prev_player"] < 0
    completes = in_deal & (s.game["deal_cards_dealt"] == _DEAL_CARDS - 1)
    g_eng = self._eng.chance_step(s.game, o, completes, s.first_player)
    s_eng = s._replace(
        game=g_eng, events=record_event(s.events, s.game, o, True)
    )
    out = _where(s.stage == STAGE_FIRST_PLAYER, s_fp, s_eng)
    return _where(at_variant, s_var, out)

  def is_chance(self, s: GinState) -> jax.Array:
    """Whether the state is a chance node.

    Args:
      s: The state.

    Returns:
      bool: an env stage before the engine, or the engine's deal.
    """
    eng = (s.game["phase"] == cm.PHASE_DEAL) & ~s.game["done"]
    return (s.stage < STAGE_ENGINE) | eng

  # -- queries -----------------------------------------------------------------
  def chance_probs(self, s: GinState) -> jax.Array:
    """The chance outcome probabilities at the state.

    Args:
      s: The state at a chance node.

    Returns:
      f32[NUM_CARDS] probabilities.
    """
    f = 0.0 if self.oklahoma_frac is None else self.oklahoma_frac
    variant = jnp.zeros(NUM_CARDS, jnp.float32).at[0].set(1.0 - f).at[1].set(f)
    first = jnp.zeros(NUM_CARDS, jnp.float32).at[0].set(0.5).at[1].set(0.5)
    eng = jnp.asarray(gin_game.chance_outcomes(s.game), jnp.float32)
    return jnp.where(
        s.stage == STAGE_VARIANT,
        variant,
        jnp.where(s.stage == STAGE_FIRST_PLAYER, first, eng),
    )

  def terminal(self, s: GinState) -> jax.Array:
    """Whether the hand is over.

    Args:
      s: The state.

    Returns:
      bool.
    """
    return jnp.asarray(s.game["done"], jnp.bool_)

  def actor(self, s: GinState) -> jax.Array:
    """Seat to move; -1 at a terminal state and at chance nodes.

    Args:
      s: The state.

    Returns:
      i32 seat.
    """
    cp = jnp.asarray(s.game["current_player"], jnp.int32)
    return jnp.where(self.terminal(s) | self.is_chance(s), jnp.int32(-1), cp)

  def legal_mask(self, s: GinState) -> jax.Array:
    """The legal actions of the seat to move (none at chance or terminal).

    Args:
      s: The state.

    Returns:
      bool[num_actions].
    """
    m = jnp.asarray(gin_game.legal_actions_mask(s.game), jnp.bool_)
    return m & ~(self.terminal(s) | self.is_chance(s))

  def returns(self, s: GinState) -> jax.Array:
    """The engine's returns per seat.

    Args:
      s: The state.

    Returns:
      f32[2].
    """
    return jnp.asarray(gin_game.get_returns(s.game), jnp.float32)

  def info_state(self, s: GinState, seat) -> GinInfo:
    """What `seat` observes, with the events made seat-relative.

    Args:
      s: The state.
      seat: The observing seat.

    Returns:
      The seat's GinInfo.
    """
    g = s.game
    seat = jnp.asarray(seat, jnp.int32)
    hand = jnp.where(seat == 0, g["player0_hand"], g["player1_hand"]).astype(
        jnp.int8
    )
    ev = s.events
    valid = jnp.arange(EVENTS_MAX, dtype=jnp.int32) < ev.count
    mine = ev.actor.astype(jnp.int32) == seat

    def i8(x):
      """`x` as int8.

      Args:
        x: A value.

      Returns:
        The int8 array.
      """
      return jnp.asarray(x).astype(jnp.int8)

    return GinInfo(
        seat=seat,
        hand=hand,
        discard_pile=i8(g["discard_pile"]),
        upcard=g["upcard"].astype(jnp.int32),
        known_cards=i8(g["known_cards"]),
        phase=g["phase"].astype(jnp.int32),
        current_player=g["current_player"].astype(jnp.int32),
        knock_card=g["knock_card"].astype(jnp.int32),
        oklahoma=g["oklahoma"].astype(jnp.bool_),
        stock_count=jnp.sum(g["deck"].astype(jnp.int32)).astype(jnp.int32),
        pass_count=g["pass_count"].astype(jnp.int32),
        num_draw_upcard=g["num_draw_upcard"].astype(jnp.int32),
        repeated_move=g["repeated_move"].astype(jnp.bool_),
        upcard_taken=g["upcard_taken"].astype(jnp.int32),
        allow_upcard_throwback=g["allow_upcard_throwback"].astype(jnp.bool_),
        layed_melds=g["layed_melds"].astype(jnp.bool_),
        layoffs_mask=g["layoffs_mask"].astype(jnp.bool_),
        knocker=g["knocker"].astype(jnp.int32),
        finished_layoffs=g["finished_layoffs"].astype(jnp.bool_),
        knocker_gin=(g["knocker_deadwood"] == 0),
        done=g["done"].astype(jnp.bool_),
        scores=jnp.stack([g["p0_score"], g["p1_score"]]).astype(jnp.int32),
        ev_actor=jnp.where(
            valid, jnp.where(mine, ACTOR_SELF, ACTOR_OPP), ACTOR_PAD
        ).astype(jnp.int8),
        ev_type=jnp.where(valid, ev.type, 0).astype(jnp.int8),
        ev_public=jnp.where(valid, ev.public_card, 0).astype(jnp.int8),
        ev_private=jnp.where(valid & mine, ev.private_card, 0).astype(jnp.int8),
        ev_count=ev.count.astype(jnp.int32),
        ev_error=ev.error.astype(jnp.int32),
    )

  def labels(self, s: GinState, seat) -> Mapping[str, jax.Array]:
    """Training labels: the opponent's hand.

    Args:
      s: The state.
      seat: The seat whose opponent is labelled.

    Returns:
      {"opp_hand": bool[52]}.
    """
    seat = jnp.asarray(seat, jnp.int32)
    opp = jnp.where(seat == 0, s.game["player1_hand"], s.game["player0_hand"])
    return {"opp_hand": opp > 0}

  # -- deals -------------------------------------------------------------------
  def deal(self, outcomes) -> GinState:
    """root() with the given chance outcomes applied, in order.

    The order is [variant (only when the env has that node)], first player,
    then the 21 deal cards.

    Args:
      outcomes: i32[n] chance outcomes.

    Returns:
      The dealt state.

    Raises:
      ValueError: outcomes does not hold exactly the deal's outcomes.
    """
    outcomes = jnp.asarray(outcomes, jnp.int32)
    n = (1 if self._variant_node else 0) + 1 + _DEAL_CARDS
    if outcomes.shape != (n,):
      raise ValueError(
          f"deal takes {n} chance outcomes, got shape {outcomes.shape}"
      )
    s, _ = jax.lax.scan(
        lambda st, o: (self.apply_chance(st, o), None), self.root(), outcomes
    )
    return s


def hide_hidden_cards(env: GinEnv, s: GinState, seat, key) -> GinState:
  """Privacy hook for check_env: re-deals what `seat` cannot see.

  The opponent's cards that are not publicly known and the stock are
  pooled and re-split at random (same counts); the opponent's private event
  cards, the engine caches when the opponent is to move, the opponent's
  per-player and final deadwood, and a non-gin knocker deadwood of the
  opponent are scrambled. Everything `seat` may know is kept.

  Args:
    env: The env (unused).
    s: The state.
    seat: The seat whose view is kept.
    key: The PRNG key of the re-deal.

  Returns:
    The state with what `seat` cannot see re-dealt.
  """
  del env
  g = dict(s.game)
  seat = jnp.asarray(seat, jnp.int32)
  opp = 1 - seat
  k_perm, k_ev, k_cache, k_dw = jax.random.split(key, 4)
  opp_hand = jnp.where(seat == 0, g["player1_hand"], g["player0_hand"]) > 0
  known = jnp.where(seat == 0, g["known_cards"][1], g["known_cards"][0]) > 0
  deck = g["deck"] > 0
  hidden = (opp_hand & ~known) | deck
  n_opp = jnp.sum(opp_hand & ~known)
  u = jax.random.uniform(k_perm, (NUM_CARDS,), dtype=jnp.float32)
  score = jnp.where(hidden, u, jnp.float32(2.0))
  rank = jnp.argsort(jnp.argsort(score))  # 0 = the smallest score
  to_opp = hidden & (rank < n_opp)
  new_opp = (to_opp | (opp_hand & known)).astype(jnp.int8)
  new_deck = (hidden & ~to_opp).astype(jnp.int8)
  g["player0_hand"] = jnp.where(seat == 1, new_opp, g["player0_hand"])
  g["player1_hand"] = jnp.where(seat == 0, new_opp, g["player1_hand"])
  g["deck"] = jnp.where(s.stage == STAGE_ENGINE, new_deck, g["deck"])
  # Engine caches belong to the player to move: scramble when that is opp.
  opp_moves = g["current_player"].astype(jnp.int32) == opp
  kc = jax.random.split(k_cache, 5)

  def bits(k, x):
    """Random bits in the shape of `x`.

    Args:
      k: The key.
      x: The array whose shape to match.

    Returns:
      bool of x's shape.
    """
    return jax.random.bernoulli(k, 0.5, x.shape)

  g["cached_legal_knocker_melds"] = jnp.where(
      opp_moves,
      bits(kc[0], g["cached_legal_knocker_melds"]),
      g["cached_legal_knocker_melds"],
  )
  g["cached_legal_discard_mask"] = jnp.where(
      opp_moves,
      bits(kc[1], g["cached_legal_discard_mask"]),
      g["cached_legal_discard_mask"],
  )
  g["can_knock"] = jnp.where(
      opp_moves, bits(kc[2], g["can_knock"]), g["can_knock"]
  )
  g["cached_min_deadwood"] = jnp.where(
      opp_moves,
      jax.random.randint(kc[3], (), 0, 99, jnp.int32),
      g["cached_min_deadwood"],
  )
  g["cached_all_discard_deadwoods"] = jnp.where(
      opp_moves,
      jax.random.randint(kc[4], (NUM_CARDS,), 0, 99, jnp.int32).astype(
          jnp.int16
      ),
      g["cached_all_discard_deadwoods"],
  )
  kd = jax.random.split(k_dw, 3)
  g["player_deadwood"] = (
      g["player_deadwood"]
      .at[opp]
      .set(
          jax.random.randint(kd[0], (), 0, 99, jnp.int32).astype(
              g["player_deadwood"].dtype
          )
      )
  )
  g["deadwood"] = (
      g["deadwood"]
      .at[opp]
      .set(
          jax.random.randint(kd[1], (), 0, 99, jnp.int32).astype(
              g["deadwood"].dtype
          )
      )
  )
  opp_knocker_dw = (g["knocker"].astype(jnp.int32) == opp) & (
      g["knocker_deadwood"] > 0
  )
  g["knocker_deadwood"] = jnp.where(
      opp_knocker_dw,
      jax.random.randint(kd[2], (), 1, 99, jnp.int32),
      g["knocker_deadwood"],
  )
  # The opponent's private stock cards in the transcript.
  ev = s.events
  opp_rows = (ev.actor.astype(jnp.int32) == opp) & (ev.private_card > 0)
  scrambled = (
      jax.random.randint(k_ev, (EVENTS_MAX,), 1, NUM_CARDS + 1, jnp.int32)
  ).astype(jnp.int8)
  events = ev._replace(
      private_card=jnp.where(opp_rows, scrambled, ev.private_card)
  )
  return s._replace(game=g, events=events)


def pyspiel_history_to_raw(history, params) -> list[int]:
  """The env's raw history for a pyspiel gin_rummy history.

  The env draws the first player at an explicit chance node before the deal;
  pyspiel names it with the `starting_player` game parameter (default 0), so
  it is prefixed as that node's outcome. A served game deals only Oklahoma or
  only standard hands, so there is no variant node to fill.

  Args:
    history: The pyspiel action history.
    params: The pyspiel game parameters.

  Returns:
    The env's raw action history.

  Raises:
    ValueError: starting_player is not 0 or 1.
  """
  sp = params.get("starting_player", 0)
  if sp not in (0, 1):
    raise ValueError(f"gin_rummy: starting_player must be 0 or 1, got {sp!r}")
  return [int(sp)] + [int(a) for a in history]


def raw_history_to_pyspiel(
    raw: list[int], kwargs: Mapping[str, Any]
) -> Optional[tuple[dict[str, Any], list[int], tuple[int, int]]]:
  """The pyspiel gin_rummy game, history and seats of a raw env history.

  The inverse of `pyspiel_history_to_raw`, for OpenSpiel's gin_rummy, whose
  player 0 always acts first: the seat that the first-player node chose
  plays pyspiel's player 0. The variant node (0 < oklahoma_frac
  < 1) picks the game's `oklahoma` parameter and the first-player node
  picks the seats; neither is a pyspiel chance node. pyspiel deals its
  player 0's ten cards first, so when seat 1 acts first the deal's two
  hands change places; every other entry (the upcard, stock draws, player
  actions) is kept as it is: gin's action ids name cards and melds, not
  seats.

  Args:
    raw: The env's raw history of one game, from the root.
    kwargs: The env's options (EnvSpec kwargs).

  Returns:
    (pyspiel parameters, the raw indices of pyspiel's history in its order,
    the env seat of each pyspiel player), or None when the env's rules have
    no pyspiel game (a rule option off its default: the served pyspiel plays
    only the defaults, registry.GIN_PYSPIEL_PARAMS).

  Raises:
    ValueError: The history is too short for its deal, or a variant or
      first-player outcome is not 0 or 1.
  """
  if any(kwargs.get(k, v) != v for k, v in DEFAULT_RULES.items()):
    return None
  f = kwargs.get("oklahoma_frac")
  variant = f is not None and 0.0 < f < 1.0
  n = (1 if variant else 0) + 1 + _DEAL_CARDS
  if len(raw) < n:
    raise ValueError(
        f"gin_rummy: a raw history of {len(raw)} entries has no complete deal"
        f" ({n} chance entries)"
    )
  if variant:
    oklahoma = raw[0]
    if oklahoma not in (0, 1):
      raise ValueError(f"gin_rummy: variant outcome {oklahoma!r} is not 0 or 1")
  else:
    oklahoma = int(f == 1.0)
  fp = raw[n - 1 - _DEAL_CARDS]
  if fp not in (0, 1):
    raise ValueError(f"gin_rummy: first-player outcome {fp!r} is not 0 or 1")
  deal = n - _DEAL_CARDS  # the first deal card's index
  h = cm.HAND_SIZE
  hands = [list(range(deal, deal + h)), list(range(deal + h, deal + 2 * h))]
  order = hands[fp] + hands[1 - fp] + list(range(deal + 2 * h, len(raw)))
  params = {"oklahoma": True} if oklahoma else {}
  return params, order, (fp, 1 - fp)


def deal_canon(info, labels):
  """uint8[N, 52 + 52 + 3]: the whole initial deal of each row.

  Seat 0's hand, seat 1's hand (the row's own hand from GinInfo.hand, the
  other from labels["opp_hand"]), the upcard + 1, the first mover and the
  Oklahoma flag. Stock cards are drawn per draw, so the initial deal
  identifies every hand that starts from it. Numpy leaves, rows at the first
  ply of a hand.

  Args:
    info: Batched GinInfo.
    labels: Batched labels (GinEnv.labels).

  Returns:
    uint8[N, 52 + 52 + 3].
  """
  own = np.asarray(info.hand) > 0
  opp = np.asarray(labels["opp_hand"]) > 0
  seat = np.asarray(info.seat).astype(np.int64)
  h0 = np.where((seat == 0)[:, None], own, opp)
  h1 = np.where((seat == 0)[:, None], opp, own)
  up = (np.asarray(info.upcard).astype(np.int64) + 1).astype(np.uint8)
  cp = np.asarray(info.current_player).astype(np.uint8)
  ok = np.asarray(info.oklahoma).astype(np.uint8)
  return np.concatenate(
      [
          h0.astype(np.uint8),
          h1.astype(np.uint8),
          up[:, None],
          cp[:, None],
          ok[:, None],
      ],
      axis=1,
  )

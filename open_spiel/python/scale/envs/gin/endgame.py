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

"""A scripted gin rummy endgame: KNOCK, LAYOFF and WALL as a Policy.

The policy plays over GinInfo.

Learning the endgame of gin (which melds to lay, which cards to lay off on
the knocker's melds) adds little to a gin agent's strength and a lot to its
training cost. A gin entrant therefore usually plays the strategic phases
(FIRST_UPCARD, DRAW, DISCARD) with its own policy and routes KNOCK, LAYOFF and
WALL to this policy (GameSpec.endgame, policies.routed.with_scripted_phase).

  pol = policies.registry.build("gin_rummy.scripted_endgame", {})
  entrant = routed.with_scripted_phase(network_policy,
  registry.game("gin_rummy").endgame)

Rules, per phase:

* KNOCK, 11 cards (the knocker's discard): `knock_discard`, by default the
  legal discard leaving the least deadwood (ties: higher rank, then lower
  index). Subclasses override the hook.
* KNOCK, <= 10 cards: the highest-index meld of the hand's optimal
  arrangement, else PASS.
* LAYOFF before the defender finishes, knocker not gin: a layoff planner.
  Every subset of the held cards that can be laid off one card at a time is
  scored by the optimal deadwood of what is left; the first card of a best
  sequence is laid iff it beats laying nothing (ties: lowest reachable
  deadwood, then lowest deadwood right after the card, then higher card
  value, then higher index), else PASS.
* LAYOFF after the defender finishes, or against gin: the highest-index
  optimal meld, else PASS.
* WALL: the rule bot's move (knock iff legal, else pass).

The policy reads only the mover's own information and public facts:
`knock_cache` recomputes the engine's KNOCK-phase discard cache from the
InfoState's hand, and a gin knock is `GinInfo.knocker_gin`.

Cost. The planner runs only when some row plans, and its 1,024-subset form
only when some planning row has more than six cards that can ever be laid
off; the rule bot runs only when some row needs it. These batch-level
`lax.cond`s keep the endgame cheap; each row's action depends on that row
only.
"""

import functools
from typing import Callable

import jax
from jax import custom_batching
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.jax.games.gin_rummy import deadwood
from open_spiel.python.jax.games.gin_rummy import layoffs
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import gin as gin_env
from open_spiel.python.scale.envs.gin import rule_bot as gin_rule_bot
from open_spiel.python.scale.policies import fixed_action
from open_spiel.python.scale.policies import registry

ARCH = "gin_rummy.scripted_endgame"

_BIG = 1 << 20
SMALL_PLAN_SLOTS = 6
FULL_PLAN_SLOTS = 10
_ROW_BITS = (1 << 13) - 1
_EXTENSION_STEPS = 10  # the JAX game's compute_layoff_cards run-extension steps
_CARD_SUIT = np.arange(cm.NUM_CARDS, dtype=np.int32) // 13
_CARD_RANK = np.arange(cm.NUM_CARDS, dtype=np.int32) % 13


def _subset_bits(k):
  """Every subset of k items as bits.

  Args:
    k: The items.

  Returns:
    bool[2**k, k].
  """
  return np.array(
      [[(s >> j) & 1 for j in range(k)] for s in range(1 << k)], dtype=bool
  )


_SUBSET_BITS = {k: _subset_bits(k) for k in (SMALL_PLAN_SLOTS, FULL_PLAN_SLOTS)}


# -- bitboards: [..., 4] int32, one 13-bit rank mask per suit
# -------------------
def _to_bits(mask):
  """[..., 52] bool -> [..., 4] int32 suit bitboards.

  Args:
    mask: bool[..., 52], the cards.

  Returns:
    int32[..., 4], the bitboards.
  """
  m = mask.reshape(mask.shape[:-1] + (4, 13)).astype(jnp.int32)
  return jnp.sum(m << jnp.arange(13, dtype=jnp.int32), axis=-1).astype(
      jnp.int32
  )


def _from_bits(bits):
  """Suit bitboards back to a card mask.

  Args:
    bits: int32[..., 4], the suit bitboards.

  Returns:
    bool[..., 52].
  """
  return ((bits[..., _CARD_SUIT] >> _CARD_RANK) & 1).astype(jnp.bool_)


def _layoff_seeds(knocker_melds):
  """(base, run) bitboards of the knocker's laid melds ([..., 185]).

  Args:
    knocker_melds: bool[..., 185], the knocker's melds.

  Returns:
    (base, run) bitboards.
  """
  table = jnp.asarray(layoffs.get_meld_layoff_cards())
  suit = jnp.asarray(layoffs.get_is_suit_meld())
  melds = knocker_melds.astype(jnp.bool_)
  base = jnp.any(table & melds[..., :, None], axis=-2)
  run = jnp.any(table & (melds & suit)[..., :, None], axis=-2)
  return _to_bits(base), _to_bits(run)


def _layable_bits(base, run, laid):
  """The cards layable on the melds once `laid` cards extend the runs.

  Args:
    base: The meld bitboards.
    run: The run bitboards.
    laid: The laid cards' bitboards.

  Returns:
    int32[..., 4], the layable bitboards.
  """
  reach = run
  for _ in range(_EXTENSION_STEPS):
    x = laid & reach
    reach = reach | (((x << 1) | (x >> 1)) & _ROW_BITS)
  return base | reach


def _closure_bits(base, run, laid, subsets):
  """Cards of each candidate set layable one at a time.

  That is S & layable(laid | S).

  Args:
    base: The layable cards of the base melds, as row bits.
    run: The run melds, as row bits.
    laid: The cards already laid, as row bits.
    subsets: The candidate sets S, as row bits.

  Returns:
    The row bits S & layable(laid | S), per candidate set.
  """
  return subsets & _layable_bits(base, run, laid | subsets)


def _ever_layable_bits(hand, base, run, laid_b):
  """The cards of `hand` layable in some order.

  Args:
    hand: int8[52], the hand.
    base: The meld bitboards.
    run: The run bitboards.
    laid_b: The laid cards' bitboards.

  Returns:
    int32[..., 4], the bitboards.
  """
  return _closure_bits(base, run, laid_b, _to_bits(hand > 0))


def _plan_over_slots(hand, base, run, laid_b, slot_cards, slot_valid, k):
  """The planner's choice, scoring every subset of the k slot cards.

  Args:
    hand: int8[52], the hand.
    base: The meld bitboards.
    run: The run bitboards.
    laid_b: The laid cards' bitboards.
    slot_cards: int32[k], the candidate cards.
    slot_valid: bool[k], which slots hold a card.
    k: The number of slots.

  Returns:
    i32[], the card to lay off, or ACTION_PASS.
  """
  bits = jnp.asarray(_SUBSET_BITS[k])
  slot_b = jnp.where(
      slot_valid[:, None],
      (jnp.arange(4, dtype=jnp.int32)[None, :] == (slot_cards // 13)[:, None])
      * (jnp.int32(1) << (slot_cards % 13))[:, None],
      0,
  ).astype(jnp.int32)
  subset_ok = ~jnp.any(bits & ~slot_valid[None, :], axis=1)
  subsets = jnp.sum(
      jnp.where(bits[:, :, None], slot_b[None, :, :], 0), axis=1
  ).astype(jnp.int32)
  reached = _closure_bits(base, run, laid_b, subsets)
  reach = subset_ok & jnp.all(reached == subsets, axis=1)
  remaining = jnp.where(
      _from_bits(subsets), jnp.int8(0), hand[None, :].astype(jnp.int8)
  )
  dw = jax.vmap(deadwood.calculate_deadwood_lut)(remaining).astype(jnp.int32)
  now = dw[0]
  poss0 = _from_bits(_layable_bits(base, run, laid_b))
  cand = slot_valid & poss0[slot_cards] & (hand[slot_cards] > 0)
  incl = bits & reach[:, None]
  best = jnp.min(jnp.where(incl, dw[:, None], _BIG), axis=0)
  first = dw[jnp.asarray(np.array([1 << j for j in range(k)], dtype=np.int32))]
  values = jnp.asarray(np.asarray(cm.get_card_values()).astype(np.int32))[
      slot_cards
  ]
  m1 = cand & (best < now)
  b = jnp.min(jnp.where(m1, best, _BIG))
  m2 = m1 & (best == b)
  f = jnp.min(jnp.where(m2, first, _BIG))
  m3 = m2 & (first == f)
  v = jnp.max(jnp.where(m3, values, -1))
  m4 = m3 & (values == v)
  card = jnp.max(jnp.where(m4, slot_cards, -1))
  return jnp.where(jnp.any(m1), card, cm.ACTION_PASS).astype(jnp.int32)


def _planned_layoff_full(hand, base, run, laid_b):
  """The planned layoff over the first FULL_PLAN_SLOTS held cards.

  Args:
    hand: int8[52], the hand.
    base: The meld bitboards.
    run: The run bitboards.
    laid_b: The laid cards' bitboards.

  Returns:
    i32[], the card to lay off, or ACTION_PASS.
  """
  held = hand > 0
  order = jnp.argsort(~held, stable=True)
  slots = order[:FULL_PLAN_SLOTS].astype(jnp.int32)
  return _plan_over_slots(
      hand, base, run, laid_b, slots, held[slots], FULL_PLAN_SLOTS
  )


def _planned_layoff_small(hand, base, run, laid_b, e):
  """The planned layoff over the first SMALL_PLAN_SLOTS cards of `e`.

  Args:
    hand: int8[52], the hand.
    base: The meld bitboards.
    run: The run bitboards.
    laid_b: The laid cards' bitboards.
    e: bool[52], the cards to plan over.

  Returns:
    i32[], the card to lay off, or ACTION_PASS.
  """
  order = jnp.argsort(~e, stable=True)
  slots = order[:SMALL_PLAN_SLOTS].astype(jnp.int32)
  return _plan_over_slots(
      hand, base, run, laid_b, slots, e[slots], SMALL_PLAN_SLOTS
  )


# -- single-row rules ---------------------------------------------------------
def own_meld_action(hand):
  """Highest-index meld of the hand's optimal arrangement, else PASS.

  Args:
    hand: The mover's hand.

  Returns:
    i32 action: the meld action, or PASS.
  """
  melds = deadwood.optimal_melds_mask(hand)
  idx = jnp.arange(cm.NUM_MELDS, dtype=jnp.int32)
  best = jnp.max(jnp.where(melds, idx, -1))
  return jnp.where(
      best >= 0, cm.ACTION_MELD_BASE + best, cm.ACTION_PASS
  ).astype(jnp.int32)


def knock_cache(info):
  """The KNOCK-phase discard cache of ONE GinInfo, from the mover's hand.

  (legal discard bool[52], deadwood after each discard i16[52]) of ONE
  GinInfo at a KNOCK node: the engine's KNOCK-phase recompute
  (`cached_legal_discard_mask`, `cached_all_discard_deadwoods`) from the
  mover's own hand.

  Args:
    info: One unbatched GinInfo at a KNOCK node.

  Returns:
    (legal discard bool[52], deadwood after each discard i16[52]).
  """
  hand = info.hand.astype(jnp.int8)
  _, all_dw = deadwood.compute_all_meld_info(hand)
  kept = jnp.where(
      info.allow_upcard_throwback,
      jnp.int32(-1),
      info.upcard_taken.astype(jnp.int32),
  )
  discardable = (hand > 0) & (jnp.arange(cm.NUM_CARDS, dtype=jnp.int32) != kept)
  legal = discardable & (all_dw <= info.knock_card.astype(jnp.int32))
  return legal, all_dw.astype(jnp.int16)


def min_deadwood_discard(legal, dw):
  """The legal knock discard leaving the least deadwood, for ONE row.

  The legal knock discard leaving the least deadwood (ties: higher rank,
  then lower index), for ONE row: legal bool[52], dw i16[52].

  Args:
    legal: bool[52] legal discards.
    dw: i16[52] deadwood after each discard.

  Returns:
    i32 card index of the discard.
  """
  cards = jnp.arange(cm.NUM_CARDS, dtype=jnp.int32)
  score = -dw.astype(jnp.int32) * 10000 + (cards % 13) * 100 + cards
  return jnp.argmax(jnp.where(legal, score, -(1 << 30))).astype(jnp.int32)


def min_deadwood_knock_discard(info, bot_action) -> jax.Array:
  """i32[B]: the default knock_discard hook, the min-deadwood legal discard.

  Args:
    info: A batched GinInfo.
    bot_action: The rule bot's actions (unused by this hook).

  Returns:
    i32[B] discards.
  """
  del bot_action

  def one(i):
    """The min-deadwood discard of one row.

    Args:
      i: One unbatched GinInfo.

    Returns:
      i32 card index.
    """
    legal, dw = knock_cache(i)
    return min_deadwood_discard(legal, dw)

  return jax.vmap(one)(info).astype(jnp.int32)


KnockDiscard = Callable[[object, jax.Array], jax.Array]


def _bot_actions(info, knock_discard: KnockDiscard):
  """The rule bot's move on every row, with a knock-discard override.

  The rule bot's move on every row, with the knocker's 11-card discard
  replaced by `knock_discard(info, bot_action)`. Batched.

  Args:
    info: A batched GinInfo.
    knock_discard: The knocker's discard, given the info and the bot's move.

  Returns:
    i32[B], the action per row.
  """
  bot = gin_rule_bot.actions(info)
  n_held = jnp.sum(info.hand.astype(jnp.int32), axis=-1)
  knock11 = (info.phase == cm.PHASE_KNOCK) & (n_held >= 11)
  return jnp.where(knock11, knock_discard(info, bot), bot).astype(jnp.int32)


# -- batched action -----------------------------------------------------------
def in_domain(info) -> jax.Array:
  """Whether the row's seat is to move at a live KNOCK / LAYOFF / WALL node.

  bool[B]: the row's seat is to move at a live KNOCK / LAYOFF / WALL node.
  ROUTE names this function.

  Args:
    info: A batched GinInfo.

  Returns:
    bool[B], per row.
  """
  return (
      gin_env.is_endgame_phase(info.phase)
      & (info.current_player == info.seat)
      & ~info.done
  )


#: The RoutedPolicy route of the scripted gin endgame.
ROUTE = f"{__name__}:in_domain"


def actions(
    info, knock_discard: KnockDiscard = min_deadwood_knock_discard
) -> jax.Array:
  """i32[B] scripted actions for a batched GinInfo (meaningful in_domain).

  Args:
    info: A batched GinInfo.
    knock_discard: The knocker's 11-card discard hook.

  Returns:
    i32[B] actions.
  """
  phase = info.phase
  b = phase.shape[0]
  hand = info.hand.astype(jnp.int8)
  n_held = jnp.sum((hand > 0).astype(jnp.int32), axis=1)
  need = in_domain(info)
  gin_knock = info.knocker_gin
  finished = info.finished_layoffs
  layoff = phase == cm.PHASE_LAYOFF
  knock = phase == cm.PHASE_KNOCK
  plan_lane = need & layoff & ~gin_knock & ~finished
  meld_lane = need & (
      (knock & (n_held <= 10)) | (layoff & (finished | gin_knock))
  )
  bot_lane = need & ~plan_lane & ~meld_lane
  knocker = jnp.clip(info.knocker, 0, 1).astype(jnp.int32)
  melds = info.layed_melds[jnp.arange(b), knocker]
  laid = info.layoffs_mask
  pass_b = jnp.full((b,), cm.ACTION_PASS, jnp.int32)

  def plan():
    """The layoff planner's actions, small or full form by batch need.

    Returns:
      i32[B] planned layoff actions.
    """
    base, run = _layoff_seeds(melds)
    laid_b = _to_bits(laid)
    e = _from_bits(jax.vmap(_ever_layable_bits)(hand, base, run, laid_b))

    def small():
      """The planner over the small subset form.

      Returns:
        i32[B] planned layoff actions.
      """
      return jax.vmap(_planned_layoff_small)(hand, base, run, laid_b, e)

    def full():
      """The planner over the full 1,024-subset form.

      Returns:
        i32[B] planned layoff actions.
      """
      return jax.vmap(_planned_layoff_full)(hand, base, run, laid_b)

    n_e = jnp.sum(e, axis=1)
    return jax.lax.cond(
        jnp.any(plan_lane & (n_e > SMALL_PLAN_SLOTS)), full, small
    )

  lay = jax.lax.cond(jnp.any(plan_lane), plan, lambda: pass_b)
  own = jax.vmap(own_meld_action)(hand)
  scripted = jnp.where(plan_lane, lay, own)
  bot = jax.lax.cond(
      jnp.any(bot_lane),
      lambda: _bot_actions(info, knock_discard),
      lambda: pass_b,
  )
  return jnp.where(plan_lane | meld_lane, scripted, bot).astype(jnp.int32)


@functools.cache
def rowwise_actions(knock_discard: KnockDiscard = min_deadwood_knock_discard):
  """`actions` with a vmap rule.

  The Arena calls a slot policy one row per call under vmap; the rule runs every
  vmapped row as ONE batched `actions` call, so its batch-level lax.conds stay
  conds. Without it XLA turns them into selects and computes every branch on
  every row.

  Args:
    knock_discard: The knocker's 11-card discard hook.

  Returns:
    A function of a GinInfo to i32 actions.
  """

  @custom_batching.custom_vmap
  def rowwise(info) -> jax.Array:
    """`actions` over `info`.

    Args:
      info: A GinInfo.

    Returns:
      i32 actions.
    """
    return actions(info, knock_discard)

  @rowwise.def_vmap
  def _rowwise_vmap(axis_size, in_batched, info):
    """The batching rule: folds the vmapped axis into the rows.

    Args:
      axis_size: The vmapped size.
      in_batched: Whether `info` is batched.
      info: The InfoStates.

    Returns:
      (actions [axis_size, B], True).
    """
    (batched,) = in_batched

    def flat(x, bt):
      """Folds the vmapped axis into the batch axis.

      Args:
        x: A leaf.
        bt: Whether the leaf is batched over the vmapped axis.

      Returns:
        The leaf with shape [axis_size * B, ...].
      """
      x = x if bt else jnp.broadcast_to(x, (axis_size,) + jnp.shape(x))
      return x.reshape((axis_size * x.shape[1],) + x.shape[2:])

    out = actions(jax.tree_util.tree_map(flat, info, batched), knock_discard)
    return out.reshape((axis_size, -1)), True

  return rowwise


class GinScriptedEndgamePolicy(types.Policy):
  """Stateless; params {}; carry (). NaN logits outside KNOCK / LAYOFF / WALL.

  Subclasses may set their own architecture name and override the
  knock_discard hook (info [B], bot_action i32[B]) -> i32[B].

  Attributes:
    spec: The policy's PolicySpec.
  """

  ARCH = ARCH
  decider = "scripted"
  ROUTE = ROUTE
  knock_discard = staticmethod(min_deadwood_knock_discard)

  def __init__(self):
    self.spec = types.PolicySpec(
        game="gin_rummy",
        obs_format="gin_info",
        num_actions=cm.NUM_ACTIONS,
        num_players=2,
        stateful=False,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name=self.ARCH,
        arch_kwargs={},
    )

  def init_params(self, key=None):
    """The policy's params (it has none).

    Args:
      key: Unused PRNG key.

    Returns:
      An empty dict.
    """
    del key
    return {}

  def encode(self, info):
    """Empty features (the policy reads the InfoState directly).

    Args:
      info: The InfoState.

    Returns:
      The empty features.
    """
    return fixed_action.empty_features(info)

  def init_carry(self, params, batch, seat, key):
    """The policy's carry (it keeps none).

    Args:
      params: Unused.
      batch: Unused.
      seat: Unused.
      key: Unused.

    Returns:
      An empty tuple.
    """
    del params, batch, seat, key
    return ()

  def observe(self, params, carry, step):
    """Passes the carry through unchanged.

    Args:
      params: Unused.
      carry: The carry.
      step: Unused.

    Returns:
      The carry.
    """
    del params, step
    return carry

  def evaluate(self, params, carry, step, key):
    """One-hot scripted actions on the rows in domain.

    Args:
      params: Unused.
      carry: Unused.
      step: The step; its info is a batched GinInfo.
      key: Unused.

    Returns:
      The policy output.
    """
    del params, carry, key
    info = step.info
    act = rowwise_actions(type(self).knock_discard)(info)
    return fixed_action.one_hot_output(act, in_domain(info), cm.NUM_ACTIONS, 2)


registry.register(ARCH, GinScriptedEndgamePolicy)

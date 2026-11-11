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

"""Determinized gin rummy worlds consistent with one seat's InfoState.

Search in gin needs concrete deals: the opponent's hidden hand, the order of
the stock, and the identity of the cards the opponent drew from the stock.
This sampler draws them from the uniform-consistent posterior, the posterior
in which the opponent's actions carry no information about its cards.

  sampler = registry.build("gin", env)
  worlds = sampler.sample_determinized({}, (), info, k, key)
  worlds.states          # GinStates, leaves [B, K, ...]
  worlds.chance_scores   # f32[B, K, 52]: each world's stock order

The model. The deal and every stock draw are uniform, so when the opponent's
actions are treated as uninformative every chance history consistent with
the public record is equally likely. Under that posterior:

  * the opponent's hidden hand H is a uniform h-subset of the unseen cards U
    (U excludes the seat's hand, the opponent's publicly known cards, the
    upcard, the discard pile and laid or laid-off cards; h is the opponent's
    hand size minus its publicly known cards). No card of U was ever public,
    so the cards of U are exchangeable and H is exactly uniform;
  * the stock is U minus H in uniform random order, an exact sample of every
    later stock draw;
  * the opponent's stock-draw cards are a uniform consistent labelling. Draw
    i's card is in H, or is a card whose first public appearance is an
    opponent discard after draw i. Sampled newest draw first, the candidate
    sets are nested, so a uniform choice among the unused candidates at each
    draw is uniform over all consistent labellings.

Cards the opponent took from the discard pile and still holds are always in
its hand. `count_worlds` gives the exact number of consistent worlds of one
InfoState (hands x stock orders x draw labellings).

A world is a GinState plus f32[52] chance scores: the stock order is the stock
cards sorted by score, highest first (`stock_order`). With
`MCTSConfig(chance="determinized")` SearchPolicy resolves every chance node of
a world's tree to the highest-scored outcome with positive probability, so
the tree draws the stock in exactly that order. `sample_worlds` returns the
GinStates alone.

Domain. At the decision nodes search runs on (FIRST_UPCARD, DRAW and DISCARD
of a live hand) the world built from the true hidden cards equals the real
state leaf for leaf. At KNOCK, LAYOFF and WALL nodes and finished hands the
worlds are well-formed but not guaranteed exact, because the knocker's laid
deadwood is not in the InfoState. The sampler supports the rule options
`allow_upcard_throwback=True` and `spade_doubles=False`, and refuses an env
with other values.

Engine fields the InfoState does not carry are rebuilt from it: the first
player (event 0's actor), pass_on_first_upcard (PASS events), prev_upcard
and drawn_from_discard (the transcript's last upcard take or declined
upcard), player_deadwood (each seat's minimum deadwood at its last discard),
prev_player (the last stock drawer), deal_cards_dealt (every card dealt,
stock draws included), and the meld cache of the seat to move.

The sampler reads only the InfoState: every world is built from it and fresh
randomness.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.jax.games.gin_rummy import deadwood
from open_spiel.python.scale.belief import registry
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import gin as gin_env

N_CARD = gin_env.NUM_CARDS
N_EVENT = gin_env.EVENTS_MAX
I8, I16, I32, F32 = jnp.int8, jnp.int16, jnp.int32, jnp.float32
NEVER = N_EVENT  # first_pub of a card that was never public


class Worlds(NamedTuple):
  """Determinized worlds, leaves [B, K, ...].

  Attributes:
    states: The worlds' gin.GinState.
    chance_scores: f32[52]: the stock order (highest first); -inf off the
      stock.
  """

  states: Any
  chance_scores: Any


def _meld_cards(layed_melds) -> jax.Array:
  """bool[52]: cards in any laid meld (layed_melds bool[2, 185]).

  Args:
    layed_melds: bool[2, 185], the laid melds.

  Returns:
    bool[52], the laid cards.
  """
  masks = jnp.asarray(cm.get_meld_masks()) > 0  # [185, 52]
  return jnp.any(masks[None, :, :] & layed_melds[:, :, None], axis=(0, 1))


def _opp(info):
  """The opponent's seat.

  Args:
    info: The GinInfo.

  Returns:
    i32[].
  """
  return jnp.int32(1) - jnp.asarray(info.seat, I32)


def _known_opp(info) -> jax.Array:
  """The opponent cards known to the seat.

  Args:
    info: The GinInfo.

  Returns:
    bool[52].
  """
  kc = info.known_cards > 0
  return jnp.where(jnp.asarray(info.seat, I32) == 0, kc[1], kc[0])


def unseen(info) -> jax.Array:
  """The cards the seat has never seen and does not know the location of.

  bool[52] (one InfoState): the cards the seat has never seen and does not
  know the location of: the opponent's hidden hand plus the stock.

  Args:
    info: One InfoState.

  Returns:
    bool[52]: the opponent's hidden hand plus the stock.
  """
  cards = jnp.arange(N_CARD, dtype=I32)
  up = (cards == jnp.asarray(info.upcard, I32)) & (
      jnp.asarray(info.upcard, I32) >= 0
  )
  seen = (
      (info.hand > 0)
      | jnp.any(info.known_cards > 0, axis=0)
      | up
      | (info.discard_pile > 0)
      | _meld_cards(info.layed_melds)
      | info.layoffs_mask
  )
  return ~seen


def hidden_count(info) -> jax.Array:
  """The number of the opponent's cards the seat does not know.

  i32: h, the number of the opponent's cards the seat does not know
  (|unseen| - stock count; every card is in exactly one place).

  Args:
    info: One InfoState.

  Returns:
    i32 h = |unseen| - stock count.
  """
  return (
      jnp.sum(unseen(info), dtype=I32) - jnp.asarray(info.stock_count, I32)
  ).astype(I32)


class _Transcript(NamedTuple):
  valid: Any  # bool[E]
  actor: Any  # i32[E] absolute seat (0 on pad rows)
  type: Any  # i32[E]
  pub: Any  # i32[E] public card, -1 = none
  first_pub: (
      Any  # i32[52] slot of the card's first public appearance, NEVER if none
  )
  opp_shed: Any  # bool[52] first public appearance is an opponent DISCARD
  opp_draw: Any  # bool[E] the opponent's stock draws


def _transcript(info) -> _Transcript:
  """The event transcript of `info`, with actors as absolute seats.

  Args:
    info: One InfoState.

  Returns:
    The _Transcript.
  """
  seat, opp = jnp.asarray(info.seat, I32), _opp(info)
  slots = jnp.arange(N_EVENT, dtype=I32)
  valid = slots < jnp.asarray(info.ev_count, I32)
  rel = info.ev_actor.astype(I32)
  actor = jnp.where(
      rel == gin_env.ACTOR_SELF,
      seat,
      jnp.where(rel == gin_env.ACTOR_OPP, opp, I32(0)),
  ).astype(I32)
  typ = jnp.where(valid, info.ev_type.astype(I32), I32(0))
  pub = jnp.where(valid, info.ev_public.astype(I32) - 1, I32(-1))
  hot = (pub[:, None] == jnp.arange(N_CARD, dtype=I32)[None, :]) & (
      pub[:, None] >= 0
  )
  first_pub = jnp.min(
      jnp.where(hot, slots[:, None], I32(NEVER)), axis=0
  ).astype(I32)
  opp_disc = valid & (rel == gin_env.ACTOR_OPP) & (typ == gin_env.EVT_DISCARD)
  at_first = hot & (slots[:, None] == first_pub[None, :])
  opp_shed = jnp.any(at_first & opp_disc[:, None], axis=0)
  opp_draw = (
      valid & (rel == gin_env.ACTOR_OPP) & (typ == gin_env.EVT_DRAW_STOCK)
  )
  return _Transcript(valid, actor, typ, pub, first_pub, opp_shed, opp_draw)


def sample_opp_labels(info, opp_hidden, key) -> jax.Array:
  """A uniform consistent labelling of the opponent's stock draws.

  i8[E]: card + 1 on the opponent's stock-draw events (0 elsewhere), a
  uniform consistent labelling given its hidden hand (module doc). One
  InfoState; opp_hidden bool[52].

  Args:
    info: One InfoState.
    opp_hidden: bool[52] the opponent's hidden hand.
    key: PRNG key.

  Returns:
    i8[E]: card + 1 on the opponent's stock-draw events, 0 elsewhere.
  """
  tr = _transcript(info)
  opp_hidden = jnp.asarray(opp_hidden, jnp.bool_)

  def body(used, i):
    """Labels event i, newest first.

    Args:
      used: bool[52] cards already used as labels.
      i: The event index.

    Returns:
      (used, the event's label i8).
    """
    cand = (opp_hidden | (tr.opp_shed & (tr.first_pub > i))) & ~used
    logits = jnp.where(cand, F32(0.0), F32(-jnp.inf))
    pick = jax.random.categorical(x64.fold_in64(key, i), logits).astype(I32)
    take = tr.opp_draw[i] & jnp.any(cand)
    used = used | (take & (jnp.arange(N_CARD, dtype=I32) == pick))
    return used, jnp.where(take, pick + 1, I32(0)).astype(I8)

  rev = jnp.arange(N_EVENT - 1, -1, -1, dtype=I32)
  _, labels = jax.lax.scan(body, jnp.zeros(N_CARD, jnp.bool_), rev)
  return labels[::-1]


def _min_deadwood_at_last_discard(tr, private, hand, p) -> jax.Array:
  """The engine's player_deadwood[p].

  i16: the engine's player_deadwood[p]: cached_min_deadwood at p's last
  DISCARD-phase discard (min over its 11 cards of the deadwood left), 999
  before p's first discard.

  Args:
    tr: The event transcript.
    private: The private payload of each transcript event.
    hand: p's current hand, bool[52].
    p: The seat.

  Returns:
    i16: the deadwood at p's last discard, or 999.
  """
  slots = jnp.arange(N_EVENT, dtype=I32)
  mine = tr.valid & (tr.actor == p)
  disc = mine & (tr.type == gin_env.EVT_DISCARD)
  j = jnp.max(jnp.where(disc, slots, I32(-1)))
  took = mine & (
      (tr.type == gin_env.EVT_DRAW_UPCARD) | (tr.type == gin_env.EVT_DRAW_STOCK)
  )
  d = jnp.max(jnp.where(took & (slots > j), slots, I32(-1)))
  ds = jnp.maximum(d, 0)
  y = jnp.where(
      tr.type[ds] == gin_env.EVT_DRAW_STOCK,
      private[ds].astype(I32) - 1,
      tr.pub[ds],
  )
  cards = jnp.arange(N_CARD, dtype=I32)
  x = tr.pub[jnp.maximum(j, 0)]
  h11 = (hand & ~((cards == y) & (d >= 0))) | (cards == x)
  _, all_dw = deadwood.compute_all_meld_info(h11.astype(I8))
  dw = jnp.min(jnp.where(h11, all_dw.astype(I32), I32(999)))
  return jnp.where(j >= 0, dw, I32(999)).astype(I16)


def assemble(info, opp_hidden, opp_labels, template) -> gin_env.GinState:
  """The GinState whose seat-`info.seat` InfoState is `info`.

  With the opponent's hidden hand `opp_hidden` (bool[52], a subset of
  `unseen(info)` of size hidden_count(info)), the stock = the rest of the unseen
  cards and the opponent's stock-draw cards `opp_labels` (i8[E], card + 1). One
  InfoState; template: GinEnv.root().game (field set and dtypes).

  Args:
    info: One InfoState.
    opp_hidden: bool[52] the opponent's hidden hand.
    opp_labels: i8[E] the opponent's stock-draw cards, card + 1.
    template: GinEnv.root().game (field set and dtypes).

  Returns:
    The assembled GinState.

  Raises:
    AssertionError: The rebuilt game fields differ from the engine's.
  """
  seat = jnp.asarray(info.seat, I32)
  u = unseen(info)
  opp_hidden = jnp.asarray(opp_hidden, jnp.bool_) & u
  hand = info.hand > 0
  opp_hand = _known_opp(info) | opp_hidden
  p0 = jnp.where(seat == 0, hand, opp_hand)
  p1 = jnp.where(seat == 0, opp_hand, hand)
  tr = _transcript(info)
  rel = info.ev_actor.astype(I32)
  private = jnp.where(
      rel == gin_env.ACTOR_SELF,
      info.ev_private.astype(I32),
      jnp.where(tr.opp_draw, jnp.asarray(opp_labels).astype(I32), I32(0)),
  )
  count = jnp.asarray(info.ev_count, I32)
  slots = jnp.arange(N_EVENT, dtype=I32)
  last = jnp.maximum(count - 1, 0)
  phase = jnp.asarray(info.phase, I32)
  drawn_from_discard = jnp.where(
      (phase == cm.PHASE_DISCARD)
      & (count > 0)
      & (tr.type[last] == gin_env.EVT_DRAW_UPCARD),
      tr.pub[last],
      I32(-1),
  )
  sets_prev = tr.valid & (
      (tr.type == gin_env.EVT_DRAW_UPCARD)
      | ((tr.type == gin_env.EVT_DRAW_STOCK) & (tr.pub >= 0))
  )
  jp = jnp.max(jnp.where(sets_prev, slots, I32(-1)))
  prev_upcard = jnp.where(jp >= 0, tr.pub[jnp.maximum(jp, 0)], I32(-1))
  js = jnp.max(
      jnp.where(tr.valid & (tr.type == gin_env.EVT_DRAW_STOCK), slots, I32(-1))
  )
  passed = jnp.stack([
      jnp.any(tr.valid & (tr.type == gin_env.EVT_PASS_UPCARD) & (tr.actor == q))
      for q in range(2)
  ])
  player_dw = jnp.stack([
      _min_deadwood_at_last_discard(
          tr, private, jnp.where(q == seat, hand, opp_hand), q
      )
      for q in range(2)
  ])
  knocker = jnp.asarray(info.knocker, I32)
  scores = jnp.asarray(info.scores, I32)
  done = jnp.asarray(info.done, jnp.bool_)
  winner = jnp.where(
      done, jnp.where(scores[0] > 0, 0, jnp.where(scores[0] < 0, 1, 2)), -1
  )
  g = {
      "player0_hand": p0,
      "player1_hand": p1,
      "known_cards": info.known_cards,
      "deck": u & ~opp_hidden,
      "discard_pile": info.discard_pile,
      "upcard": info.upcard,
      "current_player": info.current_player,
      "phase": phase,
      "oklahoma": info.oklahoma,
      "allow_upcard_throwback": info.allow_upcard_throwback,
      "wall_knock_pick": template["wall_knock_pick"],
      "spade_doubles": template["spade_doubles"],
      "score_multiplier": jnp.int32(1),
      "upcard_taken": info.upcard_taken,
      "knock_card": info.knock_card,
      # The engine counts every chance card it deals, stock draws included.
      "deal_cards_dealt": jnp.int32(N_CARD) - jnp.asarray(
          info.stock_count, I32
      ),
      # The engine keeps the last stock drawer here after the draw resolves.
      "prev_player": jnp.where(js >= 0, tr.actor[jnp.maximum(js, 0)], I32(-1)),
      "pass_count": info.pass_count,
      "pass_on_first_upcard": passed,
      "drawn_from_discard": drawn_from_discard,
      "num_draw_upcard": info.num_draw_upcard,
      "prev_upcard": prev_upcard,
      "repeated_move": info.repeated_move,
      "knocked": jnp.arange(2, dtype=I32) == knocker,
      "knocker": knocker,
      "layed_melds": info.layed_melds,
      "layoffs_mask": info.layoffs_mask,
      "finished_layoffs": info.finished_layoffs,
      "deadwood": jnp.zeros(2, I32),
      "knocker_deadwood": jnp.where(info.knocker_gin, I32(0), I32(-1)),
      "done": done,
      "winner": winner,
      "p0_score": scores[0],
      "p1_score": scores[1],
      "player_deadwood": player_dw,
  }
  # The meld cache: the seat to move's, from its hand (+ upcard), as the
  # engine computes it on entering these phases (filled in below).
  for k in (
      "cached_min_deadwood",
      "cached_legal_knocker_melds",
      "cached_legal_discard_mask",
      "cached_all_discard_deadwoods",
      "can_knock",
  ):
    g[k] = template[k]
  if set(g) != set(template):
    raise AssertionError(
        f"GinBeliefSampler rebuilds {sorted(set(g) ^ set(template))} "
        "differently from the engine's field set"
    )
  g = {k: jnp.asarray(v).astype(template[k].dtype) for k, v in g.items()}
  for k, v in g.items():
    assert v.shape == template[k].shape, (k, v.shape, template[k].shape)
  mover = jnp.clip(jnp.asarray(info.current_player, I32), 0, 1)
  cached = gin_env.install_first_player(g, mover)
  cached["current_player"] = g["current_player"]
  g = {k: jnp.asarray(cached[k]).astype(template[k].dtype) for k in template}
  first = jnp.where(
      count > 0, tr.actor[0], jnp.asarray(info.current_player, I32)
  )
  events = gin_env.Events(
      actor=tr.actor.astype(I8),
      type=info.ev_type.astype(I8),
      public_card=info.ev_public.astype(I8),
      private_card=private.astype(I8),
      count=count,
      error=jnp.asarray(info.ev_error, I32),
  )
  return gin_env.GinState(
      game=g,
      stage=jnp.int32(gin_env.STAGE_ENGINE),
      first_player=first.astype(I32),
      events=events,
  )


def stock_order(state_deck, chance_scores) -> jax.Array:
  """A world's stock in draw order (first drawn first), -1 padded.

  i32[52]: a world's stock in draw order (first drawn first), -1 padded.
  state_deck: the world's game["deck"]; chance_scores f32[52].

  Args:
    state_deck: The world's game["deck"].
    chance_scores: f32[52] chance scores.

  Returns:
    i32[52] stock cards in draw order.
  """
  deck = jnp.asarray(state_deck) > 0
  s = jnp.where(deck, jnp.asarray(chance_scores, F32), F32(-jnp.inf))
  order = jnp.argsort(-s, stable=True).astype(I32)
  n = jnp.sum(deck, dtype=I32)
  return jnp.where(jnp.arange(N_CARD, dtype=I32) < n, order, I32(-1))


#: The rule options this sampler's engine reconstruction supports.
SUPPORTED_RULES = {"allow_upcard_throwback": True, "spade_doubles": False}


class GinBeliefSampler:
  """Uniform-consistent determinized gin worlds (module doc).

  Stateless; params {}.

  Attributes:
    env: The gin env.
    kwargs: The constructor options as a record (none: {}).

  Raises:
    ValueError: at construction, for a non-gin env or rule options other than
      SUPPORTED_RULES.
  """

  capability = types.Capability.NATIVE_JAX
  stateful = False
  is_perfect_information = False
  params_source = "none"

  def __init__(self, env: types.Env):
    if not isinstance(env, gin_env.GinEnv):
      raise ValueError(
          f"GinBeliefSampler needs the gin env, got {type(env).__name__}"
      )
    template = env.root().game
    unsupported = {
        k: v
        for k, v in env.rules.items()
        if k in SUPPORTED_RULES and v != SUPPORTED_RULES[k]
    }
    if unsupported:
      raise ValueError(
          f"GinBeliefSampler rebuilds the engine under {SUPPORTED_RULES}; the "
          f"env sets {unsupported}"
      )
    self.env = env
    self.kwargs: dict[str, Any] = {}
    self._template = template

  def init_params(self, key: Any = None) -> dict[str, Any]:
    """The sampler's params (it has none).

    Args:
      key: Unused PRNG key.

    Returns:
      An empty dict.
    """
    del key
    return {}

  def init_carry(self, params, batch, seat, key):
    """The sampler's carry (it keeps none).

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

  def world(self, info, key):
    """One uniform consistent world of one InfoState (module doc).

    (GinState, chance scores f32[52]) of one InfoState: one uniform
    consistent world (module doc).

    Args:
      info: One InfoState.
      key: PRNG key.

    Returns:
      (GinState, chance scores f32[52]).
    """
    k_hand, k_order, k_labels = jax.random.split(key, 3)
    u = unseen(info)
    h = hidden_count(info)
    perm = jax.random.permutation(k_hand, N_CARD).astype(I32)
    score = jnp.where(u, perm, I32(-1))
    rank = jnp.argsort(jnp.argsort(-score, stable=True), stable=True).astype(
        I32
    )
    opp_hidden = u & (rank < h)
    labels = sample_opp_labels(info, opp_hidden, k_labels)
    state = assemble(info, opp_hidden, labels, self._template)
    order = jax.random.permutation(k_order, N_CARD).astype(F32)
    stock = state.game["deck"] > 0
    return state, jnp.where(stock, order, F32(-jnp.inf))

  def sample_determinized(self, params, carry, info, k: int, key) -> Worlds:
    """Worlds with leaves [B, K, ...]: GinStates and their stock orders.

    Args:
      params: Unused.
      carry: Unused.
      info: A batched InfoState.
      k: Worlds per row.
      key: PRNG key.

    Returns:
      The Worlds.

    Raises:
      ValueError: k < 1.
    """
    del params, carry
    if k < 1:
      raise ValueError(f"k must be >= 1, got {k}")
    b = jnp.shape(info.seat)[0]
    keys = jax.random.split(key, b)

    def row(i, kk):
      """K worlds of one row.

      Args:
        i: One InfoState.
        kk: PRNG key.

      Returns:
        GinStates and chance scores with leaves [K, ...].
      """
      return jax.vmap(lambda wk: self.world(i, wk))(jax.random.split(kk, k))

    states, scores = jax.vmap(row)(info, keys)
    return Worlds(states=states, chance_scores=scores)

  def sample_worlds(self, params, carry, info, k: int, key):
    """The states of sample_determinized(key), with zero log-weights.

    (GinStates with leaves [B, K, ...], log-weights f32[B, K] of 0): the
    states of sample_determinized(key).

    Args:
      params: Unused.
      carry: Unused.
      info: A batched InfoState.
      k: Worlds per row.
      key: PRNG key.

    Returns:
      (GinStates with leaves [B, K, ...], log-weights f32[B, K] of 0).
    """
    states = self.sample_determinized(params, carry, info, k, key).states
    return states, jnp.zeros((jnp.shape(info.seat)[0], k), F32)


# -- exact counts (host)
# ------------------------------------------------------------


def count_worlds(info) -> dict[str, int]:
  """Exact number of consistent worlds of ONE (unbatched) InfoState.

  As Python ints: hands = C(|U|, h), stock_orders = (|U| - h)!, opp_draw_labels
  = prod over the opponent's stock draws (newest first) of (h + #cards it shed
  after the draw whose first public appearance is that discard - #its draws
  after it), total = their product.

  Args:
    info: One unbatched InfoState.

  Returns:
    {"hands", "stock_orders", "opp_draw_labels", "total"} as Python ints.
  """
  u = int(np.asarray(unseen(info)).sum())
  h = int(np.asarray(hidden_count(info)))
  assert 0 <= h <= u, (h, u)
  tr = jax.tree_util.tree_map(np.asarray, _transcript(info))
  labels = 1
  later = 0
  for i in range(N_EVENT - 1, -1, -1):
    if not tr.opp_draw[i]:
      continue
    n_cand = h + int((tr.opp_shed & (tr.first_pub > i)).sum())
    choices = n_cand - later
    assert choices >= 1, (i, n_cand, later)
    labels *= choices
    later += 1
  hands, orders = math.comb(u, h), math.factorial(u - h)
  return {
      "hands": hands,
      "stock_orders": orders,
      "opp_draw_labels": labels,
      "total": hands * orders * labels,
  }


registry.register("gin", GinBeliefSampler)

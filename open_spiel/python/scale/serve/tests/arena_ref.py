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

"""The in-graph reference for serve == eval.

The in-graph reference for serve == eval: run the Arena, then rebuild each
decision's raw history (chance outcomes included) so HostAgent can be asked
about the same state.

The Arena does not log chance outcomes, so they are re-derived on the host
with the Arena's own key derivation (`Arena._hand_chance_key`, chance event n
of a hand keyed fold_in64(hand chance key, n)) and the one chance sampler
(`core.chance.sample_outcome`). The rebuilt history is checked against the
Arena's PlyLog at every ply (actor, terminal plies, rewards), so a wrong
rebuild fails here rather than passing as a HostAgent bug.
"""

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import chance as chance_lib
from open_spiel.python.scale.core import host_arena
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64

tree_map = jax.tree_util.tree_map


class Case(NamedTuple):
  """One decision the Arena made, with what HostAgent needs to replay it.

  Attributes:
    hands: tuple of tuples of raw actions: the match's hands up to this ply.
    seat: the seat that decided.
    seat_key: the Arena's policy_key[row, seat].
    first_ply: the Arena's row_ply at the first ply of the match.
    logits: f32[A] the Arena's behaviour logits.
    legal: bool[A].
    action: the Arena's action.
    steps: the Arena's SeatSteps of `seat` from the match start, [T].
  """

  hands: tuple[tuple[int, ...], ...]
  seat: int
  seat_key: Any
  first_ply: int
  logits: np.ndarray
  legal: np.ndarray
  action: int
  steps: types.SeatStep


def stack(params, k=1):
  """Params stacked k times on a leading member axis.

  Args:
    params: the params.
    k: the member count.

  Returns:
    The stacked params.
  """
  return tree_map(lambda x: jnp.stack([x] * k), params)


class _EnvFns:
  """The env functions the reference replays with, jitted once."""

  def __init__(self, env):
    self.env = env
    self.root = jax.jit(env.root)
    self.apply = jax.jit(env.apply)
    self.apply_chance = jax.jit(env.apply_chance)
    self.is_chance = jax.jit(env.is_chance)
    self.terminal = jax.jit(env.terminal)
    self.actor = jax.jit(env.actor)
    self.returns = jax.jit(env.returns)
    self.sample = jax.jit(lambda s, k: chance_lib.sample_outcome(env, s, k))
    self.fold = jax.jit(x64.fold_in64)


def _resolve(fns, s, ck, n, raw):
  """Resolves chance nodes from `s`, recording the outcomes in `raw`.

  Args:
    fns: The jitted env helpers.
    s: The state.
    ck: The chance key.
    n: The chance counter.
    raw: The list the outcomes are appended to.

  Returns:
    (state, counter).
  """
  while bool(fns.is_chance(s)):
    o = int(fns.sample(s, fns.fold(ck, jnp.int64(n))))
    raw.append(o)
    s = fns.apply_chance(s, o)
    n += 1
  return s, n


def run(
    env,
    policy,
    params,
    rule: types.DecisionRule,
    *,
    batch: int,
    num_steps: int,
    hands_per_match: int = 1,
    seed: int = 0,
    host: bool = False,
    max_cases: Optional[int] = None
) -> list[Case]:
  """Self-play `policy` (one member, both seats) in the Arena.

  Every decision of every row is a Case (up to max_cases).

  Args:
    env: the env.
    policy: the policy.
    params: its params.
    rule: the DecisionRule.
    batch: Arena rows.
    num_steps: plies.
    hands_per_match: hands per match.
    seed: the key seed.
    host: rebuild on the host path.
    max_cases: the most cases to return.

  Returns:
    The Cases.
  """
  cls = host_arena.HostArena if host else arena_lib.Arena
  ar = cls(
      env, [arena_lib.Slot(policy, 1, rule)], hands_per_match=hands_per_match
  )
  p = env.num_players
  table = arena_lib.self_play_table(batch, p, all_seats=True)
  dk, pk = arena_lib.independent_keys(jax.random.key(seed), batch, p)
  stacked = [[params]] if host else [stack(params)]
  _, tb, log = ar.collect(stacked, table, dk, pk, num_steps)
  fns = _EnvFns(env)
  actor_log = np.asarray(log.actor)
  action_log = np.asarray(log.action)
  term_log = np.asarray(log.terminal_ply)
  done_log = np.asarray(log.done)
  rew_log = np.asarray(log.rewards)
  logits_all = np.asarray(tb.behavior_logits)
  legal_all = np.asarray(tb.steps.legal_mask)
  new_match = np.asarray(tb.steps.new_match)
  cases = []
  for b in range(batch):
    hand_index = 0
    # Reaches private _hand_chance_key of the code under test.
    ck = ar._hand_chance_key(  # pylint: disable=protected-access
        dk[b][None], jnp.asarray([hand_index], jnp.int64)
    )[0]
    raw: list[int] = []
    s, n = _resolve(fns, fns.root(), ck, 0, raw)
    match_hands: list[tuple[int, ...]] = []
    match_start = 0
    hand_actions = 0
    for t in range(num_steps):
      row0 = b * p  # perspective row of seat 0 (flags are seat independent)
      if new_match[row0, t]:
        assert hand_actions == 0, "a match starts at a hand's first ply"
        match_hands, match_start = [], t
      if term_log[b, t]:
        assert bool(fns.terminal(s)), (b, t)
        match_hands.append(tuple(raw))
        hand_index += 1
        # Reaches private _hand_chance_key of the code under test.
        ck = ar._hand_chance_key(  # pylint: disable=protected-access
            dk[b][None], jnp.asarray([hand_index], jnp.int64)
        )[0]
        raw = []
        hand_actions = 0
        s, n = _resolve(fns, fns.root(), ck, 0, raw)
        continue
      actor = int(actor_log[b, t])
      assert int(fns.actor(s)) == actor, (b, t, int(fns.actor(s)), actor)
      a = int(action_log[b, t])
      if max_cases is None or len(cases) < max_cases:
        row = b * p + actor
        cases.append(
            Case(
                hands=tuple(match_hands) + (tuple(raw),),
                seat=actor,
                seat_key=pk[b, actor],
                first_ply=match_start,
                logits=logits_all[row, t],
                legal=legal_all[row, t],
                action=a,
                steps=tree_map(
                    lambda x, r=row, m=match_start, e=t + 1: np.asarray(
                        x[r, m:e]
                    ),
                    tb.steps,
                ),
            )
        )
      raw.append(a)
      hand_actions += 1
      s = fns.apply(s, a)
      s, n = _resolve(fns, s, ck, n, raw)
      if done_log[b, t]:
        assert bool(fns.terminal(s)), (b, t)
        np.testing.assert_array_equal(np.asarray(fns.returns(s)), rew_log[b, t])
  return cases

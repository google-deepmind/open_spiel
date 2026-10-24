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

"""Arena and decide edge cases.

duplicate deals share every chance outcome of a hand, a slot evaluates only rows
whose actor it owns, the policy checker catches a carry not reset at a new hand,
decide stays finite on overflowing logits, and a TrajectoryBatch marks hand
ends.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games import game as game_base
from open_spiel.python.jax.games import leduc_poker
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import jax_game_env
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import check_policy
from open_spiel.python.scale.testing import fixtures
from open_spiel.python.scale.testing import steps as steps_lib
from open_spiel.python.scale.testing.errors import ContractError

tree_map = jax.tree_util.tree_map


def _stack(params, k=1):
  """Stacks `params` k times on a new leading axis.

  Args:
    params: The params.
    k: The copies.

  Returns:
    The stacked params.
  """
  return tree_map(lambda x: jnp.stack([x] * k), params)


class _LeducWithCurrentPlayer(leduc_poker.LeducPoker):
  """LeducPoker exposing its current_player state field."""

  def current_player(self, state):
    """The current_player field of `state`.

    Args:
      state: the state.

    Returns:
      state.current_player.
    """
    return state.current_player


def test_the_engine_adapter_wraps_leduc():
  """JaxGameEnv wraps a game that defines no current_player method."""

  class Plain(jax_game_env.JaxGameEnv):
    """Leduc through the engine adapter, no overrides."""

    name, max_len = "leduc_poker", 8

  Plain(leduc_poker.LeducPoker())


class _LeducEnv(jax_game_env.JaxGameEnv):
  """Minimal test env: info = own card + public card; labels expose the deal."""

  name = "leduc_poker"
  max_len = 8
  zero_sum = True

  def __init__(self):
    super().__init__(_LeducWithCurrentPlayer())

  def info_state(self, s, seat):
    """The seat's own card and the public card.

    Args:
      s: the state.
      seat: the seat.

    Returns:
      {"card", "public"} as int32.
    """
    seat = jnp.asarray(seat, jnp.int32)
    return {
        "card": s.private_cards[seat].astype(jnp.int32),
        "public": jnp.asarray(s.public_card, jnp.int32),
    }

  def labels(self, s, seat):
    """Both private cards and the public card.

    Args:
      s: the state.
      seat: the seat.

    Returns:
      {"private", "public"} as int32.
    """
    del seat
    return {
        "private": s.private_cards.astype(jnp.int32),
        "public": jnp.asarray(s.public_card, jnp.int32),
    }


def _per_hand(nh, x):
  """Splits row x [T, ...] into hands at new_hand plies.

  Args:
    nh: bool[T], the row's new_hand flags.
    x: The row's values, [T, ...].

  Returns:
    The hands' slices of x.
  """
  starts = list(np.flatnonzero(nh)) + [len(nh)]
  return [x[a:b] for a, b in zip(starts[:-1], starts[1:])]


def test_duplicate_rows_share_the_public_card():
  """Seat-swapped duplicate rows see the same public card.

  They do even when their betting lines differ in length: mid-hand chance is
  keyed by the hand's chance-event count, not by the ply.
  """
  env = _LeducEnv()
  pol = uniform.UniformPolicy("leduc_poker", env.num_actions, env.num_players)
  ar = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(pol, 1, types.SAMPLE),
          arena_lib.Slot(pol, 1, types.SAMPLE),
      ],
  )
  n = 64
  table = arena_lib.rotated_table(n, [0, 1], [0, 0])
  dk, pk = arena_lib.duplicate_keys(jax.random.key(0), n, 2)
  _, batch, _ = ar.collect([_stack({}), _stack({})], table, dk, pk, 64)
  nh = np.asarray(batch.steps.new_hand)
  priv = np.asarray(batch.labels["private"])
  pub = np.asarray(batch.labels["public"])
  compared, mismatched = 0, 0
  for i in range(n):
    ra, rb = 2 * i, 2 * i + 1
    ha, hb = _per_hand(nh[ra], np.arange(nh.shape[1])), _per_hand(
        nh[rb], np.arange(nh.shape[1])
    )
    for pa, pb in zip(ha, hb):
      np.testing.assert_array_equal(priv[ra, pa[0]], priv[rb, pb[0]])
      boards_a = pub[ra, pa][pub[ra, pa] != game_base.INVALID_CARD]
      boards_b = pub[rb, pb][pub[rb, pb] != game_base.INVALID_CARD]
      if boards_a.size and boards_b.size:
        compared += 1
        mismatched += int(boards_a[0] != boards_b[0])
  assert compared > 50, compared
  assert (
      mismatched == 0
  ), f"{mismatched}/{compared} duplicate hands saw different boards"


def test_evaluate_runs_only_on_rows_whose_actor_is_in_the_slot():
  """A slot's evaluate never sees a seat it did not observe.

  every carry it evaluates has count >= 1.
  """
  env = kuhn.KuhnEnv()
  seen = []

  class Recording(fixtures.HistoryPolicy):
    """A HistoryPolicy that records the carries it evaluates."""

    def evaluate(self, params, carry, step, key):
      """Records the counts, then evaluates as HistoryPolicy.

      Args:
        params: the policy params.
        carry: the carry.
        step: the SeatStep.
        key: the evaluate key.

      Returns:
        HistoryPolicy's PolicyOutput.
      """
      jax.debug.callback(
          lambda c: seen.append(np.asarray(c).ravel().copy()), carry["count"]
      )
      return super().evaluate(params, carry, step, key)

  hist = Recording("kuhn_poker", env.num_actions, env.num_players)
  ar = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(hist, 1, types.SAMPLE),
          arena_lib.Slot(
              uniform.UniformPolicy("kuhn_poker", 2, 2), 1, types.SAMPLE
          ),
      ],
  )
  table = arena_lib.rotated_table(4, [0, 1], [0, 0])
  dk, pk = arena_lib.duplicate_keys(jax.random.key(1), 4, 2)
  out = ar.collect([_stack(hist.init_params()), _stack({})], table, dk, pk, 8)
  jax.block_until_ready(out)
  counts = np.concatenate(seen)
  assert counts.size > 0
  assert (counts >= 1).all(), (
      f"{int((counts < 1).sum())}/{counts.size} slot-0 evaluate calls were on "
      "a seat "
      "slot 0 never observed (a row whose actor sits in slot 1)"
  )


def test_check_policy_catches_carry_not_reset_at_new_hand():
  """check_policy fails a policy that does not reset its carry at new_hand.

  The policy resets on new_match only.
  """
  env = kuhn.KuhnEnv()

  class NoHandReset(fixtures.HistoryPolicy):
    """A HistoryPolicy that resets at new_match only."""

    def observe(self, params, carry, step):
      """Observes with new_hand replaced by new_match.

      Args:
        params: the policy params.
        carry: the carry.
        step: the SeatStep.

      Returns:
        the carry.
      """
      return super().observe(
          params, carry, step._replace(new_hand=step.new_match)
      )

  pol = NoHandReset("kuhn_poker", env.num_actions, env.num_players)
  s1, s2 = steps_lib.two_ply_steps(env, 8, 0)
  with pytest.raises(ContractError):
    check_policy.check_policy(
        pol, pol.init_params(), pol.make_params(b=jnp.array([1.0, 0.0])), s1, s2
    )


@pytest.mark.parametrize(
    "logits,rule",
    [
        ([[jnp.inf, 0.0]], types.ARGMAX),  # +inf legal logit
        ([[1.0, 1.0]], types.DecisionRule("sample", 1e-39)),  # tiny T overflows
    ],
)
def test_decide_probs_finite_on_overflow(logits, rule):
  """A legal +inf logit, or logits / T that overflow, give finite probs."""
  a, p = decide_lib.decide(
      jnp.array(logits, jnp.float32),
      jnp.ones((1, 2), bool),
      rule,
      jax.random.key(0),
  )
  p = np.asarray(p)
  assert np.isfinite(p).all() and abs(p.sum() - 1.0) < 1e-6, p
  assert int(a[0]) in (0, 1)


def test_trajectory_batch_exposes_hand_end():
  """TrajectoryBatch.done marks the plies whose action ended a hand.

  The last ply of a segment included; it equals PlyLog.done.
  """
  env = kuhn.KuhnEnv()
  pol = uniform.UniformPolicy("kuhn_poker", 2, 2)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  dk, pk = arena_lib.independent_keys(jax.random.key(2), 8, 2)
  _, batch, log = ar.collect(
      [_stack({})], arena_lib.self_play_table(8, 2), dk, pk, 5
  )
  assert "done" in types.TrajectoryBatch._fields, types.TrajectoryBatch._fields
  np.testing.assert_array_equal(np.asarray(batch.done), np.asarray(log.done))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

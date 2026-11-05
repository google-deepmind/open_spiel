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

"""Side credit (PGConfig.credit="side", core.arena Arena credit="side").

* team_pennies_2v2 (partners score alike): a side-credit batch is the
  seat-credit batch bit for bit, but for the SIDE_CREDIT label, so side
  credit cannot change what a learner learns there; kuhn (two players): the
  same.
* team_dilemma_2v2 (partners score differently): every emitted reward is
  the mean of its side's seat rewards, which differs from the seat's own.
* PGLearner refuses a batch whose credit label disagrees with its config.
* The league's PG kind builds its Arena with the learner's credit; a
  HostArena refuses side credit.
* Learning on team_dilemma_2v2 by PPO self-play on every seat: side credit
  learns the cooperative team equilibrium (every seat cooperates), seat
  credit the seat game's all-defect equilibrium, from the same init and
  keys.
"""

import sys
import types as pytypes

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import game_tree
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.envs import nplayer_matrix as nm
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league.kinds import pg as pg_kind
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.policies import registry as pol_registry

tree_map = jax.tree_util.tree_map


def _policy(env):
  """A small network policy of `env` (nplayer_matrix or kuhn).

  Args:
    env: The env.

  Returns:
    The policy.
  """
  if env.name == "nplayer_matrix":
    family = "nplayer_matrix_seat"
    enc = enc_registry.get(
        enc_registry.family_name(
            family, env_registry.spec("nplayer_matrix", variant=env.variant)
        )
    )
  else:
    enc = enc_registry.get("kuhn_poker_observation_tensor")
  kw = common.network_policy_kwargs(
      env.name,
      enc.name,
      list(enc.shape),
      env.num_actions,
      env.num_players,
      catalog.mlp([16], "relu", "separate"),
  )
  return pol_registry.build("network", kw)


def _batch(env, credit, b=8, t=10, seed=0):
  """One self-play batch of `env` under `credit`, every seat a perspective.

  Args:
    env: The env.
    credit: "seat" or "side".
    b: Arena rows.
    t: Steps.
    seed: The seed.

  Returns:
    (trajectory batch, policy, params).
  """
  pol = _policy(env)
  params = pol.init_params(jax.random.key(seed))
  arena = arena_lib.Arena(
      env, [arena_lib.Slot(pol, 1, types.SAMPLE)], credit=credit
  )
  table = arena_lib.self_play_table(b, env.num_players, all_seats=True)
  src = arena_lib.ArenaSource(
      arena, [tree_map(lambda x: x[None], params)], table, t
  )
  return src.next(jax.random.key(seed + 1)), pol, params


def _equal_but_label(seat, side):
  """Asserts two batches are equal but for the SIDE_CREDIT label.

  Args:
    seat: The seat-credit batch.
    side: The side-credit batch.
  """
  assert types.SIDE_CREDIT not in seat.labels
  assert bool(jnp.all(side.labels[types.SIDE_CREDIT]))
  side = side._replace(
      labels={k: v for k, v in side.labels.items() if k != types.SIDE_CREDIT}
  )
  la, lb = jax.tree_util.tree_leaves(seat), jax.tree_util.tree_leaves(side)
  assert jax.tree_util.tree_structure(seat) == jax.tree_util.tree_structure(
      side
  )
  for x, y in zip(la, lb):
    assert x.dtype == y.dtype and x.shape == y.shape
    assert bool(jnp.array_equal(x, y))


@pytest.mark.parametrize("game", ["team_pennies_2v2", "kuhn"])
def test_side_credit_is_seat_credit_where_partners_score_alike(game):
  """team_pennies_2v2 and two-player kuhn: bit for bit, but for the label."""
  # Built here, not in the parametrize list: no env (and no array) at import.
  env = kuhn.KuhnEnv() if game == "kuhn" else nm.NPlayerMatrixEnv(game)
  seat, _, _ = _batch(env, "seat")
  side, _, _ = _batch(env, "side")
  assert float(jnp.abs(seat.reward).sum()) > 0.0
  _equal_but_label(seat, side)


def test_side_credit_emits_the_side_mean_where_partners_differ():
  """team_dilemma_2v2: each emitted reward is its side's mean seat reward."""
  env = nm.NPlayerMatrixEnv("team_dilemma_2v2")
  sides = sides_lib.sides_of(env)
  seat, _, _ = _batch(env, "seat", b=16)
  side, _, _ = _batch(env, "side", b=16)
  p = env.num_players
  # all_seats rows: b * P + m is seat m of Arena row b; the same game.
  r_seat = np.asarray(seat.reward).reshape(-1, p, seat.reward.shape[1])
  r_side = np.asarray(side.reward).reshape(-1, p, side.reward.shape[1])
  want = np.moveaxis(sides.side_returns(np.moveaxis(r_seat, 1, -1)), -1, 1)[
      :, list(sides.side_of_seat)
  ]
  np.testing.assert_array_equal(r_side, want.astype(np.float32))
  assert np.any(r_side != r_seat)
  # Everything but the reward (and the label) is the seat batch's.
  _equal_but_label(seat._replace(reward=side.reward), side)


def _cfg(credit):
  """A PPO config with `credit`.

  Args:
    credit: "seat" or "side".

  Returns:
    The PGConfig.
  """
  return pg.PGConfig(loss="ppo", anneal_lr=False, credit=credit)


def test_the_learner_refuses_a_batch_of_the_other_credit():
  """PGLearner.update checks the SIDE_CREDIT label against its credit."""
  env = nm.NPlayerMatrixEnv("team_dilemma_2v2")
  seat, pol, _ = _batch(env, "seat")
  side, _, _ = _batch(env, "side")
  for credit, batch in (("side", seat), ("seat", side)):
    with pytest.raises(ValueError, match="credit"):
      pg.PGLearner(pol, _cfg(credit)).check_batch(batch)
  pg.PGLearner(pol, _cfg("side")).check_batch(side)
  pg.PGLearner(pol, _cfg("seat")).check_batch(seat)
  with pytest.raises(ValueError, match="credit"):
    arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)], credit="team")


@pytest.mark.parametrize("credit", ["seat", "side"])
def test_the_league_pg_kind_builds_its_arena_with_the_learners_credit(credit):
  """PGDriver.attach passes PGConfig.credit to its Arena."""
  env = nm.NPlayerMatrixEnv("team_dilemma_2v2")
  pol = _policy(env)
  driver = pg_kind.PGDriver(pol, pg.PGLearner(pol, _cfg(credit)))
  league = pytypes.SimpleNamespace(
      env=env,
      config=pytypes.SimpleNamespace(num_steps=10, learner_rule=types.SAMPLE),
      opponent_slots=lambda: [],
  )
  driver.attach(league, None)
  assert driver.arena.credit == credit
  slot = arena_lib.Slot(pol, 1, types.SAMPLE)
  with pytest.raises(NotImplementedError, match="HostArena"):
    driver._arena(  # pylint: disable=protected-access
        league, None, slot, host=True, credit="side"
    )


def _train(credit, updates=60, b=32, t=10, seed=3):
  """Trains team_dilemma_2v2 self-play under `credit`.

  Args:
    credit: "seat" or "side".
    updates: The updates.
    b: Arena rows.
    t: Steps.
    seed: The seed.

  Returns:
    The final cooperation probabilities.
  """
  env = nm.NPlayerMatrixEnv("team_dilemma_2v2")
  pol = _policy(env)
  cfg = pg.PGConfig(
      loss="ppo",
      lr=1e-2,
      anneal_lr=False,
      num_updates=updates,
      gamma=1.0,
      gae_lambda=1.0,
      update_epochs=2,
      num_minibatches=2,
      credit=credit,
  )
  learner = pg.PGLearner(pol, cfg, env_max_len=env.max_len)
  state = learner.init(jax.random.key(seed), env)
  arena = arena_lib.Arena(
      env, [arena_lib.Slot(pol, 1, types.SAMPLE)], credit=credit
  )
  table = arena_lib.self_play_table(b, env.num_players, all_seats=True)

  def stack(s):
    """The learner's params as the one-member pool stack.

    Args:
      s: The learner state.

    Returns:
      [params with a leading member axis].
    """
    return [tree_map(lambda x: x[None], learner.params(s))]

  src = arena_lib.ArenaSource(arena, stack(state), table, t)
  key = jax.random.key(seed + 100)
  for _ in range(updates):
    key, k_roll, k_upd = jax.random.split(key, 3)
    src.set_params(stack(state))
    state, _ = learner.update(state, src.next(k_roll), k_upd)
  tree = game_tree.GameTree(env)
  probs = tree.policy_table(pol, learner.params(state), types.SAMPLE)
  coop = np.zeros(env.num_players)
  for row in range(tree.num_infosets):
    coop[int(tree.player[row])] = probs[row, 0]  # action 0 cooperates
  return coop


def test_side_credit_learns_the_cooperative_equilibrium():
  """team_dilemma_2v2: side credit cooperates, seat credit defects.

  Same init, same keys; only the credit differs. The cooperative profile is
  the team game's equilibrium (each side's return rises 0.5 per seat that
  cooperates), the all-defect profile the seat game's (a seat's own return
  rises 1 when it defects).
  """
  side = _train("side")
  seat = _train("seat")
  print(f"P(cooperate) per seat: side credit {side}, seat credit {seat}")
  assert np.all(side > 0.9), side
  assert np.all(seat < 0.1), seat


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

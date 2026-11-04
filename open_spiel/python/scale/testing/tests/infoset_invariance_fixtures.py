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

"""The infoset-invariance fixtures of the shipped search operators and emitters.

testing.infoset_invariance.walk requires one fixture per architecture or
runtime emitter that decides by search or emits training targets. Importing
this module registers:

  search             SearchPolicy PIMC on leduc (LeducBeliefSampler over the
                     chance tree) and on gin (determinized PIMC over
                     GinBeliefSampler worlds with the routed scripted
                     endgame); partner histories from each env's privacy hook
                     (leduc.hide_opponent_card; gin.hide_hidden_cards, which
                     re-deals the opponent's hand, the stock order and the
                     opponent's private memory of stock draws)
  testing.host_echo  the HOST_DRIVEN test operator
                     (testing.host_operator_fixtures), with the interleaved
                     and fresh-instance checks
  DaggerPolicy       a DAgger mixture whose teacher is a leduc network
  BehaviorMixture    the behaviour/label mixture over the leduc SearchPolicy

Default sizes are a few dozen positions per history length; with
OPEN_SPIEL_SCALE_INFOSET_INVARIANCE_FULL=1 the gin fixture plays 1,100 rows per
history length.

  for case in infoset_invariance.cases("search"):
    infoset_invariance.check_infoset_invariance(case)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import plies as plies_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.encoders import gin as gin_encoder
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.envs.gin import search as gin_search
from open_spiel.python.scale.envs.leduc import belief as leduc_belief
from open_spiel.python.scale.networks import policy as network_policy
from open_spiel.python.scale.policies import operators
from open_spiel.python.scale.policies import search
from open_spiel.python.scale.sources import dagger
from open_spiel.python.scale.testing import check_operator
from open_spiel.python.scale.testing import host_operator_fixtures
from open_spiel.python.scale.testing import infoset_invariance


def leduc_net() -> network_policy.NetworkPolicy:
  """Returns a small leduc MLP network over leduc_poker_info_state.

  Returns:
    The NetworkPolicy.
  """
  return network_policy.NetworkPolicy(
      game="leduc_poker",
      obs_format="leduc_poker_info_state",
      num_actions=3,
      num_players=2,
      encoder="leduc_poker_info_state",
      obs_shape=[30],
      network={
          "compute_dtype": "float32",
          "torso": {"kind": "mlp", "widths": [16], "layout": "shared"},
          "policy_head": {"kind": "dense"},
          "value_head": {"kind": "scalar", "tanh": True},
      },
      value_layout="zero_sum",
  )


def leduc_search(k: int = 2, sims: int = 4):
  """Returns PIMC over k leduc belief worlds.

  Args:
    k: The number of belief worlds.
    sims: The simulations per search.

  Returns:
    (env, base network, SearchPolicy).
  """
  env = leduc.LeducEnv()
  base = leduc_net()
  sp = search.SearchPolicy(
      base,
      env,
      leduc_belief.LeducBeliefSampler(env, base),
      search.MCTSConfig(num_simulations=sims, num_worlds=k),
  )
  return env, base, sp


def gin_net(env) -> network_policy.NetworkPolicy:
  """Returns a small gin MLP network over the OpenSpiel gin observation.

  Args:
    env: The gin env.

  Returns:
    The NetworkPolicy.
  """
  cfg = {
      "compute_dtype": "float32",
      "torso": {"kind": "mlp", "widths": [32], "layout": "shared"},
      "policy_head": {"kind": "dense"},
      "value_head": {"kind": "scalar"},
  }
  return network_policy.NetworkPolicy(
      env.name,
      gin_encoder.NAME,
      env.num_actions,
      2,
      gin_encoder.NAME,
      [gin_encoder.DIM],
      cfg,
  )


def _leduc_case() -> infoset_invariance.InfosetCase:
  """The Leduc search infoset case.

  Returns:
    The InfosetCase.
  """
  env, _, sp = leduc_search()
  return infoset_invariance.InfosetCase(
      "search/leduc_pimc",
      env,
      sp,
      sp.init_params(jax.random.key(0)),
      leduc.hide_opponent_card,
      plies=(1, 2, 3, 4),
      rows=24,
      seed=3,
  )


def gin_case(rows: int = 0) -> infoset_invariance.InfosetCase:
  """Returns the gin PIMC case.

  Args:
    rows: The rows; 0 picks the default or the FULL size.

  Returns:
    The InfosetCase.
  """
  env = gin.GinEnv()
  net = gin_net(env)
  pol = gin_search.make(
      net,
      env,
      gin_search.mcts_config(num_simulations=4, num_worlds=2, value_scale=50.0),
  )
  params = gin_search.make_params(pol, net.init_params(jax.random.key(1)))
  n = rows or (1100 if infoset_invariance.full() else 12)
  return infoset_invariance.InfosetCase(
      "search/gin_pimc",
      env,
      pol,
      params,
      gin.hide_hidden_cards,
      plies=(1, 2, 3, 6, 9),
      rows=n,
      seed=5,
  )


def _search_cases():
  """The search infoset cases.

  Returns:
    The cases.
  """
  return [_leduc_case(), gin_case()]


def host_echo_case() -> infoset_invariance.InfosetCase:
  """Returns the host-driven echo operator case on leduc.

  It has the interleave and fresh checks.

  Returns:
    The InfosetCase.
  """
  env = leduc.LeducEnv()
  base = leduc_net()
  bs = base.spec
  base_arch = dict(arch_name=bs.arch_name, arch_kwargs=dict(bs.arch_kwargs))
  op = host_operator_fixtures.HostEchoOperator(base_arch)
  return infoset_invariance.InfosetCase(
      "host_echo/leduc",
      env,
      op,
      base.init_params(jax.random.key(2)),
      leduc.hide_opponent_card,
      plies=(1, 2, 3),
      rows=24,
      seed=7,
      fresh=lambda: host_operator_fixtures.HostEchoOperator(base_arch),
      interleave=True,
  )


def dagger_case() -> infoset_invariance.InfosetCase:
  """Returns a 50/50 DAgger mixture of two leduc networks.

  Returns:
    The InfosetCase.
  """
  env = leduc.LeducEnv()
  student, teacher = leduc_net(), leduc_net()
  mix = dagger.DaggerPolicy(student, teacher)
  params = mix.make_params(
      student.init_params(jax.random.key(3)),
      teacher.init_params(jax.random.key(4)),
      0.5,
  )
  return infoset_invariance.InfosetCase(
      "dagger/leduc",
      env,
      mix,
      params,
      leduc.hide_opponent_card,
      plies=(1, 2, 3),
      rows=24,
      seed=9,
  )


def _label_mixture_case() -> infoset_invariance.InfosetCase:
  """The Leduc behaviour-mixture infoset case.

  Returns:
    The InfosetCase.
  """
  env, base, sp = leduc_search()
  mixture = operators.BehaviorMixture(base, sp, "mixture(0.5)")
  bp = base.init_params(jax.random.key(5))
  params = mixture.make_params(bp, {"base": bp, "sampler": {}})
  return infoset_invariance.InfosetCase(
      "label_mixture/leduc",
      env,
      mixture,
      params,
      leduc.hide_opponent_card,
      plies=(1, 2, 3),
      rows=24,
      seed=11,
  )


FIXTURE_CASES = {
    "search": _search_cases,
    host_operator_fixtures.HostEchoOperator.ARCH: lambda: [host_echo_case()],
    dagger.DaggerPolicy.__name__: lambda: [dagger_case()],
    operators.BehaviorMixture.__name__: lambda: [_label_mixture_case()],
}
for _key, _fn in FIXTURE_CASES.items():
  if _fn not in infoset_invariance.FIXTURES.get(_key, []):
    infoset_invariance.register_fixture(_key, _fn)


# -- the test's shards ----------------------------------------------------

#: infoset_invariance_test runs as NUM_SHARDS files (shard 0 is that file,
#: shard k is infoset_invariance_shard<k>_test.py), so that each stays within
#: a test runner's per-file time budget. A fixture case is checked one history
#: length at a time (check_infoset_invariance's plies), and the shards take
#: the (case, history length) items round-robin.
NUM_SHARDS = 6

#: The operator family whose check_operator test runs in a shard file.
SHARDED_OPERATOR_FAMILY = ("gin_pimc", 3)


def shard_items(
    shard: int,
) -> list[tuple[str, infoset_invariance.InfosetCase, tuple[int, ...]]]:
  """The fixture items one shard checks.

  Args:
    shard: the shard, 0 <= shard < NUM_SHARDS.

  Returns:
    [(key, case, plies), ...]: every NUM_SHARDS-th (case, history length)
    item, starting at `shard`.

  Raises:
    ValueError: shard is out of range.
  """
  if not 0 <= shard < NUM_SHARDS:
    raise ValueError(f"shard {shard}: want 0 <= shard < {NUM_SHARDS}")
  items = []
  for key in sorted(infoset_invariance.FIXTURES):
    for case in infoset_invariance.cases(key):
      items.extend((key, case, (t,)) for t in sorted(set(case.plies)))
  return items[shard::NUM_SHARDS]


def parametrize_shard(metafunc: Any, shard: int) -> None:
  """Parametrizes a shard file's test with the shard's items, at collection.

  A shard file's `pytest_generate_tests` calls it, so the cases (which hold
  arrays) are built when pytest collects the test, never when the file is
  imported. A test without the (key, case, plies_) arguments is left alone.

  Args:
    metafunc: pytest's Metafunc of the test being collected.
    shard: the file's shard, 0 <= shard < NUM_SHARDS.
  """
  if {"key", "case", "plies_"} <= set(metafunc.fixturenames):
    metafunc.parametrize("key,case,plies_", shard_items(shard), ids=item_id)


def item_id(x) -> str:
  """A parametrize id for a shard_items value.

  Args:
    x: a key, an InfosetCase or a plies tuple.

  Returns:
    The case's name, "ply<t>", or str(x).
  """
  if isinstance(x, infoset_invariance.InfosetCase):
    name = x.name
  elif isinstance(x, tuple):
    name = "ply" + "-".join(map(str, x))
  else:
    name = str(x)
  return name


def check_invariant(case: infoset_invariance.InfosetCase, plies) -> None:
  """The infoset-invariance test of one shard item.

  Args:
    case: the item's case.
    plies: the item's history lengths.
  """
  rep = infoset_invariance.check_infoset_invariance(case, plies)
  assert rep["positions"] > 0 and rep["rows_hidden_differ"] > 0


def operator_steps(env, n=8, seed=0, plies_=3):
  """A SeatStep batch of n playouts at history length plies_.

  Args:
    env: the Env.
    n: playouts.
    seed: the playout seed.
    plies_: the history length.

  Returns:
    The SeatStep of each playout's actor at plies_.
  """
  states, actor, _ = infoset_invariance.playouts(env, n, plies_, seed)
  t = plies_
  s_t = jax.tree_util.tree_map(lambda x: x[:, t], states)
  a = jnp.maximum(actor[:, t], 0)
  return plies_lib.seat_step(
      env, s_t, a, False, False, types.NO_ACTION, types.NO_ACTOR
  )


def check_operator_family(family: str) -> dict[str, Any]:
  """check_operator over one operator family.

  Args:
    family: leduc_pimc, gin_pimc or host_echo.

  Returns:
    check_operator's report.

  Raises:
    ValueError: an unknown family.
  """
  if family == "leduc_pimc":
    env, _, op = leduc_search()
    params = op.init_params(jax.random.key(0))
  elif family == "gin_pimc":
    case = gin_case()
    env, op, params = case.env, case.policy, case.params
  elif family == "host_echo":
    env = leduc.LeducEnv()
    base = leduc_net()
    bs = base.spec
    op = host_operator_fixtures.HostEchoOperator(
        dict(arch_name=bs.arch_name, arch_kwargs=dict(bs.arch_kwargs))
    )
    params = base.init_params(jax.random.key(1))
  else:
    raise ValueError(f"operator family {family!r}")
  return check_operator.check_operator(op, params, operator_steps(env))

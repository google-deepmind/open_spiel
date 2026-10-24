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

"""Known-answer digests of Arena rollouts on CPU: the rollout regression guard.

Each case builds an Arena, rolls it out, continues the rollout from the final
state, and digests every leaf of (final ArenaState, TrajectoryBatch, PlyLog).
arena_golden_test checks the current Arena reproduces every recorded digest
bit for bit, so a change to the driver (chunking, skipped work, sharding)
cannot change what it emits.

Compare, printing each case:

  python -m open_spiel.python.scale.core.tests.arena_golden

Record (every case, or only some with --only=a,b):

  python -m open_spiel.python.scale.core.tests.arena_golden --write

Record only from a tree whose rollouts are known good.

Cases (each crosses hand boundaries so the deal path runs):
  kuhn_history_self_play   stateful HistoryPolicy, every seat a perspective
  kuhn_callback            NATIVE_JAX stateful slot + CALLBACK_BATCHED K=2 slot
  leduc_population         duplicate deals, K=3 population slot, 2-hand matches
  leduc_network_self_play  the MMD leduc network (stateless), all seats
  leduc_network_vs_uniform the network vs the uniform slot (two slots)
  gin_network_vs_uniform   the PPO gin network (stateless) vs uniform, 3 hands+
  gin_network_vs_rule_bot  the PPO gin network vs the scripted rule bot slot
  kuhn_rnn_network_self_play  a recurrent (GRU) network, all seats
  c4_search_self_play      AZ SearchPolicy (target_policy aux), all seats
"""

import dataclasses
import hashlib
import json
import os
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.configs import az_connect_four
from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import frameworks
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import refs
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.league import sampler as league_sampler
from open_spiel.python.scale.learners import az
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.parallel import data_parallel as dpl
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import uniform

tree_map = jax.tree_util.tree_map
#: This module's import name, under any import root (`python -m` runs it as
#: __main__, its spec keeps the name).
_MODULE = __spec__.name if __spec__ is not None else __name__
GOLDEN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "arena_golden.json"
)
#: The gin cases deal with every hand's knock limit at the deadwood maximum.
GIN_OKLAHOMA_FRAC = 1.0


def _host(x):
  """`x` as numpy, typed keys as their key data.

  Args:
    x: An array or key.

  Returns:
    The numpy array.
  """
  x = x if isinstance(x, jax.Array) else jnp.asarray(x)
  if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
    return np.asarray(jax.random.key_data(x))
  return np.asarray(x)


def digest(tree) -> dict[str, Any]:
  """{leaf path: "dtype shape sha256"} over every leaf.

  Args:
    tree: a pytree of arrays.

  Returns:
    the digest dict.
  """
  out = {}
  for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
    x = np.ascontiguousarray(_host(leaf))
    out[jax.tree_util.keystr(path)] = (
        f"{x.dtype} {list(x.shape)} "
        f"{hashlib.sha256(x.tobytes()).hexdigest()[:24]}"
    )
  return out


def _stack(params, k=1):
  """Stacks `params` k times on a new leading axis.

  Args:
    params: The params.
    k: The copies.

  Returns:
    The stacked params.
  """
  return tree_map(lambda x: jnp.stack([jnp.asarray(x)] * k), params)


def _arena(env, policy, opponent, b, rule_bot=False):
  """(Arena, SlotTable, slot_params) of a network case.

  slot 0 the network at sample T=1; self_play: every seat a perspective;
  uniform: slot 1 the uniform seat (or, rule_bot, the scripted gin rule bot at
  argmax) with a one-member entries table. A fixture of the Arena's inputs
  only: runs build their Arenas through the League (league.runner).

  Args:
    env: the Env.
    policy: the network policy.
    opponent: "self_play" or "uniform".
    b: the row count.
    rule_bot: with "uniform", seat the gin rule bot instead.

  Returns:
    (Arena, SlotTable, a function from the network's params to slot_params).
  """
  p = env.num_players
  slots = [arena_lib.Slot(policy, 1, types.DecisionRule("sample", 1.0))]

  def stack(x):
    """Adds a leading member axis of 1.

    Args:
      x: A pytree.

    Returns:
      The stacked tree.
    """
    return tree_map(lambda y: jnp.asarray(y)[None], x)

  if opponent == "self_play":
    assert not rule_bot
    table = arena_lib.self_play_table(b, p, all_seats=True)

    def slot_params(prm):
      """The slot params of self-play: the policy's only.

      Args:
        prm: The policy's params.

      Returns:
        [stacked params].
      """
      return [stack(prm)]

  else:
    assert opponent == "uniform", opponent
    if rule_bot:
      # Lazy: optional dependency (the gin env needs the games package).
      from open_spiel.python.scale.envs.gin import rule_bot as gin_rule_bot  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

      slots.append(
          arena_lib.Slot(gin_rule_bot.GinRuleBotPolicy(), 1, types.ARGMAX)
      )
    else:
      slots.append(
          arena_lib.Slot(
              uniform.UniformPolicy(env.name, env.num_actions, p),
              1,
              types.SAMPLE,
          )
      )
    table = league_sampler.entries_table(b, p, [(1, 0, 1.0)], jax.random.key(0))

    def slot_params(prm):
      """The slot params vs uniform: the policy's, then none.

      Args:
        prm: The policy's params.

      Returns:
        [stacked params, {}].
      """
      return [stack(prm), {}]

  return arena_lib.Arena(env, slots, mesh=dpl.batch_mesh(1)), table, slot_params


def _network_parts(cfg):
  """The env and network policy of a network config.

  Args:
    cfg: The network run config.

  Returns:
    (env, policy).
  """
  assert cfg.policy_arch == "network", cfg.policy_arch
  return config_common.build_env(cfg), registry.build(
      "network", dict(cfg.policy_kwargs)
  )


def _network_case(cfg_name, opponent, b, t, seed):
  """A network golden case: the arena and its inputs.

  Args:
    cfg_name: The config.
    opponent: self_play or uniform.
    b: The rows.
    t: The plies.
    seed: The seed.

  Returns:
    (arena, params, table, deal key, policy key, plies).
  """
  cfg = config_common.with_overrides(
      config_common.load_config(cfg_name), num_envs=b, num_steps=t
  )
  env, policy = _network_parts(cfg)
  arena, table, slot_params = _arena(env, policy, opponent, b)
  params = slot_params(policy.init_params(jax.random.key(seed)))
  dk, pk = arena_lib.independent_keys(
      jax.random.key(seed + 1), b, env.num_players
  )
  return arena, params, table, dk, pk, t


def _ppo_gin_config(rows: int, num_steps: int, opponent: str):
  """A PPO gin league.

  an MLP 256x2 over the 644-wide gin observation vs the rule bot (at argmax)
  or a uniform seat, no endgame routing.

  Args:
    rows: the row count.
    num_steps: plies per batch.
    opponent: "rule_bot" or "uniform".

  Returns:
    the LeagueRunConfig.
  """
  # Lazy: optional dependency (the gin env needs the games package).
  from open_spiel.python.scale.envs.gin import rule_bot as gin_rule_bot  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  net = catalog.mlp([256, 256], "relu", "shared")
  kwargs = config_common.network_policy_kwargs(
      "gin_rummy", "gin_rummy_observation_tensor", [644], 241, 2, net
  )
  ppo = recipes.ppo_vs_uniform(
      "bench_ppo_gin",
      "connect_four",
      "connect_four_flat_planes",
      [126],
      7,
      num_updates=1000,
      width=256,
  )
  if opponent == "rule_bot":
    opp = refs.FixedOpponent(
        "rule_bot", refs.PolicyRef(arch_name=gin_rule_bot.ARCH), types.ARGMAX
    )
  else:
    opp = recipes.uniform_opponent("gin_rummy", 241)
  return runner.one_slot(
      "bench_ppo_gin",
      "gin_rummy",
      ppo.learner,
      num_envs=rows,
      num_steps=num_steps,
      num_updates=1000,
      policy_kwargs=kwargs,
      self_play=False,
      opponents=(opp,),
      env_kwargs={"oklahoma_frac": GIN_OKLAHOMA_FRAC},
  )


def card_at_deal(info):
  """The card-at-deal features, f32[B, 6].

  one_hot(seat, 2) ++ one_hot(card, 3) * deal ++ [deal], where deal = (num_bets
  == 0), the first ply of the hand (the recurrent case's encoder: only a carry
  can remember the card).

  Args:
    info: a batched kuhn InfoState.

  Returns:
    f32[B, 6].
  """
  deal = (jnp.asarray(info.num_bets) == 0).astype(jnp.float32)
  card = jax.nn.one_hot(jnp.asarray(info.card), 3, dtype=jnp.float32)
  seat = jax.nn.one_hot(jnp.asarray(info.seat), 2, dtype=jnp.float32)
  return jnp.concatenate([seat, card * deal[:, None], deal[:, None]], axis=-1)


def _kuhn_rnn_config(num_envs: int, num_steps: int):
  """Kuhn MMD self-play with a recurrent (GRU) card-at-deal network.

  Args:
    num_envs: The rows.
    num_steps: The plies.

  Returns:
    The config.
  """
  learner = pg.PGConfig(
      loss="mmd",
      lr=3e-3,
      lr_end=3e-3,
      anneal_lr=False,
      num_updates=10,
      max_grad_norm=0.5,
      adam_eps=1e-5,
      use_adamw=False,
      update_epochs=4,
      num_minibatches=4,
      shuffle="env_axis",
      gamma=0.99,
      gae_lambda=0.95,
      pg=losses.PGLossConfig(
          clip_eps=0.1, vf_coef=0.5, clip_value=False, kl_init_to_uniform=False
      ),
      ent_coef=schedules.Schedule("constant", 0.05),
      kl_coef=schedules.Schedule("constant", 0.2),
      reference_update_every=1,
      value_ema=True,
      value_ema_beta=0.925,
  )
  network = {
      "compute_dtype": "float32",
      "torso": {
          "kind": "recurrent",
          "hidden_size": 32,
          "use_last_action": True,
          "inner": {
              "kind": "mlp",
              "widths": [32],
              "layout": "shared",
              "activation": "relu",
          },
      },
      "policy_head": {"kind": "dense"},
      "value_head": {"kind": "scalar"},
  }
  policy_kwargs = dict(
      game="kuhn_poker",
      obs_format="kuhn_card_at_deal",
      num_actions=2,
      num_players=2,
      encoder=f"{_MODULE}:card_at_deal",
      obs_shape=[6],
      network=network,
      value_layout="seat",
  )
  return runner.one_slot(
      "control_mmd_kuhn_poker_rnn_card_at_deal",
      "kuhn_poker",
      kinds.PG(learner),
      num_envs=num_envs,
      num_steps=num_steps,
      num_updates=10,
      policy_kwargs=policy_kwargs,
      self_play=True,
      seed=0,
      keep_every=3,
  )


def _gin_case(opponent="uniform", seed=5):
  """A PPO gin network (MLP 256x2) vs uniform or the rule bot.

  Args:
    opponent: uniform or rule_bot.
    seed: The seed.

  Returns:
    (arena, params, table, deal key, policy key, plies).
  """
  b, t = 8, 224
  cfg = _ppo_gin_config(b, t, opponent)
  env, policy = _network_parts(cfg)
  arena, table, slot_params = _arena(
      env, policy, "uniform", b, rule_bot=(opponent == "rule_bot")
  )
  params = slot_params(policy.init_params(jax.random.key(seed)))
  dk, pk = arena_lib.independent_keys(
      jax.random.key(seed + 1), b, env.num_players
  )
  return arena, params, table, dk, pk, t


def _kuhn_rnn_case():
  """A recurrent card-at-deal network (a GRU carry that observe updates).

  Self-play, every seat a perspective.

  Returns:
    (arena, params, table, deal keys, policy keys, plies).
  """
  b, t = 16, 24
  cfg = _kuhn_rnn_config(b, t)
  env, policy = _network_parts(cfg)
  arena, table, slot_params = _arena(env, policy, "self_play", b)
  params = slot_params(policy.init_params(jax.random.key(9)))
  dk, pk = arena_lib.independent_keys(jax.random.key(10), b, env.num_players)
  return arena, params, table, dk, pk, t


def _c4_search_case():
  """A Connect Four search golden case.

  Returns:
    (arena, params, table, deal key, policy key, plies).
  """
  cfg = dataclasses.replace(
      az_connect_four.az_config(), batch_size=4, num_simulations=8
  )
  env, _, sp, arena, table, _, _ = az.build(cfg, mesh=dpl.batch_mesh(1))
  params = [_stack(sp.init_params(jax.random.key(7)))]
  dk, pk = arena_lib.independent_keys(jax.random.key(8), 4, env.num_players)
  return arena, params, table, dk, pk, 50


def _probe_case(build):
  """A golden case from a probe builder.

  Args:
    build: A probe case builder.

  Returns:
    (arena, params, table, deal key, policy key, plies).
  """
  env, slots, params, table, dk, pk, t, hpm = build()
  arena = arena_lib.Arena(
      env, slots, hands_per_match=hpm, mesh=dpl.batch_mesh(1)
  )
  return arena, params, table, dk, pk, t


def cases():
  """The golden cases by name.

  Returns:
    name -> a builder of (arena, params, table, deal keys, policy keys, T).
  """
  # Lazy: optional dependency (probe_arena's leduc env needs the games package).
  from open_spiel.python.scale.parallel.tests import probe_arena  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  return {
      "kuhn_history_self_play": lambda: _probe_case(probe_arena.kuhn_self_play),
      "kuhn_callback": lambda: _probe_case(probe_arena.kuhn_callback),
      "leduc_population": lambda: _probe_case(probe_arena.leduc_population),
      "leduc_network_self_play": lambda: _network_case(
          "mmd_leduc_poker", "self_play", 16, 40, 1
      ),
      "leduc_network_vs_uniform": lambda: _network_case(
          "mmd_leduc_poker", "uniform", 16, 40, 3
      ),
      "gin_network_vs_uniform": _gin_case,
      "gin_network_vs_rule_bot": lambda: _gin_case("rule_bot", 11),
      "kuhn_rnn_network_self_play": _kuhn_rnn_case,
      "c4_search_self_play": _c4_search_case,
  }


def run_case(name) -> dict[str, Any]:
  """Rolls out case `name`, then continues it from the final state.

  Args:
    name: a key of cases().

  Returns:
    the digest of both rollouts.
  """
  # The digests are x64 bits: the test's conftest and the script's main turn
  # x64 on (never this module's import).
  x64.assert_x64(f"{_MODULE}.run_case")
  arena, params, table, dk, pk, t = cases()[name]()
  state, batch, log = arena.collect(params, table, dk, pk, t)
  # Two rollouts from the same state: the second continues hands mid-way.
  state2, batch2, log2 = arena.rollout(params, table, state, t // 2)
  assert (
      int(jnp.sum(log.terminal_ply)) + int(jnp.sum(log2.terminal_ply)) >= 1
  ), f"{name}: no hand ended; the deal path is not covered"
  return digest(
      dict(state=state2, batch=batch, log=log, batch2=batch2, log2=log2)
  )


def main(argv):
  """With --write [--only=a,b], (re)digests cases into GOLDEN.

  Args:
    argv: the command line.
  """
  frameworks.enable_x64()  # a script: x64 on before any array is made
  if "--write" in argv:  # [--only=a,b]: (re)digest those cases, keep the others
    only = [
        a.split("=", 1)[1].split(",") for a in argv if a.startswith("--only=")
    ]
    out = json.load(open(GOLDEN, encoding="utf-8")) if only else {}
    for name in only[0] if only else cases():
      out[name] = run_case(name)
    out["_meta"] = {
        "jax": jax.__version__,
        "platform": jax.devices()[0].platform,
    }
    with open(GOLDEN, "w", encoding="utf-8") as f:
      json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {GOLDEN}")
    return
  want = json.load(open(GOLDEN, encoding="utf-8"))
  for name in cases():
    got = run_case(name)
    bad = sorted(k for k in want[name] if got.get(k) != want[name][k])
    print(
        name,
        "OK" if not bad and set(got) == set(want[name]) else f"DIFF {bad[:8]}",
    )


if __name__ == "__main__":
  main(sys.argv)

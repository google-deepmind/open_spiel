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

r"""Two-player behaviour goldens.

Reference digests for two-player seating, evaluation, and league loops.

two_player_golden_test checks that every two-player seating, Evaluator, and
league-loop digest is reproduced bit for bit alongside the general N-player
and team code paths.

  # compare, one line per case
  python -m open_spiel.python.scale.testing.goldens.two_player
  # record
  python -m open_spiel.python.scale.testing.goldens.two_player --write \
      [--only=a,b]

Record only on a tree whose two-player behaviour is known good, on CPU (the
league and evaluator digests are XLA:CPU bits). XLA:CPU bits also depend on
the CPU model (testing.cpu_class), so --write stamps the file with the host's
CPU class under META ("cpu_class"), and the test skips the XLA:CPU cases,
naming both classes, on a host of another class. A partial --write (--only)
on a host of another class is refused: it would mix two classes' bits in one
file. A lane that must compare the XLA:CPU cases sets REQUIRE_CLASS_ENV
(OPEN_SPIEL_SCALE_REQUIRE_GOLDEN_CPU_CLASS=1): on a host of another class the
test then fails, and this tool refuses to compare, instead of skipping.

Cases. Each returns {key: value}: arrays as "dtype shape sha256[:24]", floats
as float.hex (exact), small ints and strings as they are.

  sampler/*        league.sampler.table_from_mix over a grid of mixes, seeds,
                   batch sizes, P = 2 and P = 4, stratified and iid;
                   entries_table; core.arena.self_play_table. The P = 4 cases
                   pin today's P = 4 table (one entrant fills every
                   non-learner seat, a partner seat included), which N-player
                   seating never builds at P > 2 (unreachable there): they say
                   only that the function is unchanged.
  keys/*           core.arena_engine.deal_keys, core.arena.duplicate_keys and
                   independent_keys.
  evaluator/*      match.Evaluator head_to_head (SAMPLE and ARGMAX),
                   payoff_matrix and mixture_head_to_head on kuhn and leduc,
                   network entrants at fixed init params and uniform.
  league/*         short kuhn Leagues: PSRO on evaluator payoffs and on exact
                   payoffs (league.mixture), ScheduledPromotion with its
                   promotion gate (loop.League.gate_cell), and self-play with
                   self_play_seats="all"; every drawn table, payoff matrix,
                   meta-strategy, gate cell and the final learner params.
"""

import functools
import hashlib
import json
import os
import sys
import tempfile
from typing import Any, Callable, Optional

import jax
import numpy as np

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import arena_engine
from open_spiel.python.scale.core import frameworks
from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.league import artifact_store
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import controller as controller_lib
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import sampler
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import cpu_class

GOLDEN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "two_player.json"
)
#: The file's one non-case entry: {"cpu_class": the recording host's class}.
META = "_meta"
#: Cases whose digests are XLA:CPU bits (CPU only, and only on the recorded
#: CPU class); the sampler and key cases are numpy and threefry.
XLA_CPU_PREFIXES = ("evaluator/", "league/")
#: Set to "1" in a lane that must compare the XLA:CPU cases: a host of another
#: CPU class then fails instead of skipping them (cpu_class.REQUIRE_CLASS_ENV).
REQUIRE_CLASS_ENV = cpu_class.REQUIRE_CLASS_ENV
class_required = cpu_class.class_required


def class_mismatch(golden: dict[str, Any]) -> Optional[str]:
  """Why the XLA:CPU cases cannot be compared on this host, or None.

  Args:
    golden: The loaded golden file.

  Returns:
    A message naming the recorded and the host CPU class when they differ,
    None when they are the same.

  Raises:
    ValueError: a golden file with no recorded CPU class.
  """
  return cpu_class.mismatch(golden.get(META, {}).get("cpu_class"), GOLDEN)


# -- digests -------------------------------------------------------------------
def _host(x):
  """`x` as a host array; a typed PRNG key as its key data.

  Args:
    x: An array or key.

  Returns:
    The numpy array.
  """
  if isinstance(x, jax.Array) and jax.dtypes.issubdtype(
      x.dtype, jax.dtypes.prng_key
  ):
    return np.asarray(jax.random.key_data(x))
  return np.asarray(x)


def arr(x) -> str:
  """The "dtype shape sha256[:24]" of an array.

  Args:
    x: the array.

  Returns:
    The digest string.
  """
  a = np.ascontiguousarray(_host(x))
  digest = hashlib.sha256(a.tobytes()).hexdigest()[:24]
  return f"{a.dtype} {list(a.shape)} {digest}"


def fl(x) -> str:
  """A float, exactly.

  Args:
    x: the float.

  Returns:
    float.hex.
  """
  return float(x).hex()


def tree(t) -> str:
  """One digest of every leaf of a pytree (paths, dtypes, shapes, bytes).

  Args:
    t: the pytree.

  Returns:
    The digest (24 hex digits).
  """
  h = hashlib.sha256()
  for path, leaf in jax.tree_util.tree_flatten_with_path(t)[0]:
    a = np.ascontiguousarray(_host(leaf))
    h.update(f"{jax.tree_util.keystr(path)} {a.dtype} {a.shape}|".encode())
    h.update(a.tobytes())
  return h.hexdigest()[:24]


def _table(prefix: str, t, out: dict[str, Any]) -> None:
  out[f"{prefix}.seat_slot"] = arr(t.seat_slot)
  out[f"{prefix}.seat_member"] = arr(t.seat_member)
  out[f"{prefix}.perspective_seat"] = arr(t.perspective_seat)


def _estimate(prefix: str, e, out: dict[str, Any]) -> None:
  out[prefix] = [fl(e.mean), fl(e.se), fl(e.lo), fl(e.hi), int(e.n)]


def _match(prefix: str, r, out: dict[str, Any]) -> None:
  """Records a MatchResult's fields under `prefix`.

  Args:
    prefix: The key prefix.
    r: The match.MatchResult.
    out: The digests, updated.
  """
  out[f"{prefix}.names"] = [
      r.game,
      r.config,
      r.a,
      r.b,
      str(r.a_rule),
      str(r.b_rule),
  ]
  out[f"{prefix}.shape"] = [
      int(r.n_deals),
      int(r.hands_per_match),
      bool(r.duplicate),
  ]
  out[f"{prefix}.a_payoff"] = arr(r.a_payoff)
  out[f"{prefix}.b_payoff"] = arr(r.b_payoff)
  _estimate(f"{prefix}.mean", r.mean, out)
  _estimate(f"{prefix}.seat0", r.seat[0], out)
  _estimate(f"{prefix}.seat1", r.seat[1], out)
  _estimate(f"{prefix}.diff", r.diff, out)
  out[f"{prefix}.se_unpaired"] = fl(r.se_unpaired)
  out[f"{prefix}.variance_reduction"] = fl(r.variance_reduction)


def _matrix(prefix: str, pm, out: dict[str, Any]) -> None:
  out[f"{prefix}.names"] = list(pm.names)
  for f in ("mean", "se", "lo", "hi"):
    out[f"{prefix}.{f}"] = arr(getattr(pm, f))
  out[f"{prefix}.n_deals"] = int(pm.n_deals)


# -- sampler -------------------------------------------------------------------
_WHERE = {"a@1": (1, 0), "b@2": (1, 1), "c@3": (2, 0), "d@4": (1, 2)}
_MIXES = {
    "one": sampler.Mix(0.0, {"a@1": 1.0}),
    "self_two": sampler.Mix(0.25, {"a@1": 0.5, "b@2": 0.25}),
    "self": sampler.Mix(1.0, {}),
    "thirds": sampler.Mix(0.1, {"a@1": 0.3, "b@2": 0.3, "c@3": 0.3}),
    "uneven": sampler.Mix(
        0.05, {"a@1": 0.61, "b@2": 0.07, "c@3": 0.19, "d@4": 0.08}
    ),
}


def _sampler_cases() -> dict[str, Any]:
  """The sampler/* cases (module doc).

  Returns:
    {case name: fn() -> {key: value}}.
  """
  out = {}
  for p in (2, 4):
    for mix_name in _MIXES:
      for batch in (37, 64):
        for strat in (True, False):
          for seed in (0, 1, 2):
            name = (
                f"sampler/table_from_mix/p{p}/{mix_name}/b{batch}/"
                f"strat{int(strat)}/s{seed}"
            )
            if p > 2:
              name = name.replace("table_from_mix", "table_from_mix_legacy_p4")
            out[name] = functools.partial(
                _table_from_mix, p, mix_name, batch, strat, seed
            )
    out[f"sampler/entries_table/p{p}"] = lambda p=p: _entries(p)
    for all_seats in (False, True):
      out[f"sampler/self_play_table/p{p}/all{int(all_seats)}"] = (
          lambda p=p, a=all_seats: _self_play(p, a)
      )
  out["sampler/table_from_mix/p2/learner_slot/s5"] = lambda: _table_from_mix(
      2, "self_two", 40, True, 5, learner=(0, 3)
  )
  return out


def _table_from_mix(
    p, mix_name, batch, strat, seed, learner=(0, 0)
) -> dict[str, Any]:
  """The digests of one sampler.table_from_mix draw.

  Args:
    p: The players.
    mix_name: A key of _MIXES.
    batch: The rows.
    strat: Whether the draw is stratified.
    seed: The seed.
    learner: The learner's (slot, member).

  Returns:
    {key: value}.
  """
  t, counts = sampler.table_from_mix(
      _MIXES[mix_name], _WHERE, batch, p, rng.key(seed), strat, learner
  )
  out = {}
  _table("table", t, out)
  out["counts"] = {str(k): int(v) for k, v in sorted(counts.items())}
  return out


def _entries(p) -> dict[str, Any]:
  """The digests of sampler.entries_table at two seeds.

  Args:
    p: The players.

  Returns:
    {key: value}.
  """
  out = {}
  for seed in (0, 7):
    t = sampler.entries_table(
        50, p, [(1, 0, 0.5), (1, 1, 0.3), (2, 0, 0.2)], rng.key(seed)
    )
    _table(f"s{seed}", t, out)
  return out


def _self_play(p, all_seats) -> dict[str, Any]:
  """The digests of core.arena.self_play_table.

  Args:
    p: The players.
    all_seats: Whether every seat is the learner.

  Returns:
    {key: value}.
  """
  out = {}
  _table("table", arena_lib.self_play_table(12, p, all_seats=all_seats), out)
  return out


# -- keys ----------------------------------------------------------------------
def _key_cases() -> dict[str, Any]:
  """The keys/* cases (module doc).

  Returns:
    {case name: fn() -> {key: value}}.
  """

  def deal(dup, seed):
    """The deal and policy keys of arena_engine.deal_keys.

    Args:
      dup: duplicate deals.
      seed: the seed.

    Returns:
      {"deal", "policy"} digests.
    """
    d, pk = arena_engine.deal_keys(jax.random.key(seed), 9, dup)
    return {"deal": arr(d), "policy": arr(pk)}

  def dup_keys(p):
    """The keys of core.arena.duplicate_keys.

    Args:
      p: the players.

    Returns:
      {"deal", "policy"} digests.
    """
    d, pk = arena_lib.duplicate_keys(jax.random.key(3), 5, p)
    return {"deal": arr(d), "policy": arr(pk)}

  def ind_keys(p):
    """The keys of core.arena.independent_keys.

    Args:
      p: the players.

    Returns:
      {"deal", "policy"} digests.
    """
    d, pk = arena_lib.independent_keys(jax.random.key(4), 6, p)
    return {"deal": arr(d), "policy": arr(pk)}

  out = {}
  for dup in (True, False):
    for seed in (0, 11):
      out[f"keys/deal_keys/dup{int(dup)}/s{seed}"] = lambda d=dup, s=seed: deal(
          d, s
      )
  for p in (2, 4):
    out[f"keys/duplicate_keys/p{p}"] = lambda p=p: dup_keys(p)
    out[f"keys/independent_keys/p{p}"] = lambda p=p: ind_keys(p)
  return out


# -- evaluator -----------------------------------------------------------------
def _kuhn_policy(width: int = 32):
  """A kuhn MLP network policy.

  Args:
    width: The hidden width.

  Returns:
    The policy.
  """
  kw = common.network_policy_kwargs(
      "kuhn_poker",
      "kuhn_poker_info_index",
      [12],
      2,
      2,
      catalog.mlp([width], "relu", "separate"),
  )
  return registry.build("network", kw)


def _kuhn_env():
  """The kuhn_poker env.

  Returns:
    The env.
  """
  return env_registry.make("kuhn_poker")


def _game(game: str):
  """(env, network policy) of a game.

  Args:
    game: "kuhn" or "leduc".

  Returns:
    (env, policy).
  """
  if game == "kuhn":
    return _kuhn_env(), _kuhn_policy()
  cfg = common.load_config("mmd_leduc_poker")
  return common.build_env(cfg), registry.build(
      cfg.policy_arch, cfg.policy_kwargs
  )


def _entrants(game: str, rule=types.SAMPLE):
  """Three network entrants at fixed init params and uniform.

  Args:
    game: "kuhn" or "leduc".
    rule: The networks' decision rule.

  Returns:
    (env, [three network entrants], the uniform entrant).
  """
  env, pol = _game(game)
  nets = [
      match.Entrant(f"net{i}", pol, pol.init_params(jax.random.key(i)), rule)
      for i in range(3)
  ]
  uni = match.Entrant(
      "uniform",
      uniform.UniformPolicy(env.name, env.num_actions, env.num_players),
      {},
      types.SAMPLE,
  )
  return env, nets, uni


def _evaluator(env, **kw):
  """An Evaluator of `env` at 64 rows.

  Args:
    env: The env.
    **kw: More EvaluatorConfig fields.

  Returns:
    The match.Evaluator.
  """
  return match.Evaluator(env, match.EvaluatorConfig(max_rows=64, **kw))


def _h2h(game, rule_name) -> dict[str, Any]:
  """The evaluator/<game>/head_to_head/<rule> case.

  Args:
    game: "kuhn" or "leduc".
    rule_name: "sample" or "argmax".

  Returns:
    {key: value}.
  """
  rule = {"sample": None, "argmax": types.ARGMAX}[rule_name]
  env, nets, uni = _entrants(game)
  ev = _evaluator(env, rule=rule)
  out = {}
  _match(
      "net0_vs_uniform",
      ev.head_to_head(nets[0], uni, 48, jax.random.key(1)),
      out,
  )
  _match(
      "net1_vs_net2",
      ev.head_to_head(nets[1], nets[2], 48, jax.random.key(2)),
      out,
  )
  _match(
      "net0_vs_net0",
      ev.head_to_head(nets[0], nets[0], 16, jax.random.key(3)),
      out,
  )
  return out


def _h2h_independent(game) -> dict[str, Any]:
  """The evaluator/<game>/head_to_head/independent case.

  Args:
    game: "kuhn" or "leduc".

  Returns:
    {key: value}.
  """
  env, nets, uni = _entrants(game)
  out = {}
  _match(
      "net0_vs_uniform",
      _evaluator(env, duplicate=False).head_to_head(
          nets[0], uni, 32, jax.random.key(4)
      ),
      out,
  )
  return out


def _payoff_matrix(game) -> dict[str, Any]:
  """The evaluator/<game>/payoff_matrix case.

  Args:
    game: "kuhn" or "leduc".

  Returns:
    {key: value}.
  """
  env, nets, uni = _entrants(game)
  pm = _evaluator(env).payoff_matrix(nets + [uni], 24, jax.random.key(5))
  out = {}
  _matrix("pm", pm, out)
  for (i, j), r in sorted(pm.results.items()):
    _match(f"cell{i}{j}", r, out)
  return out


def _mixture(game) -> dict[str, Any]:
  """The evaluator/<game>/mixture_head_to_head case.

  Args:
    game: "kuhn" or "leduc".

  Returns:
    {key: value}.
  """
  env, nets, uni = _entrants(game)
  mm = _evaluator(env).mixture_head_to_head(
      nets[:2],
      [0.25, 0.75],
      [nets[2], uni],
      [0.6, 0.4],
      40,
      jax.random.key(6),
      "A",
      "B",
  )
  out = {}
  _match("result", mm.result, out)
  out["a_members"], out["b_members"] = arr(mm.a_members), arr(mm.b_members)
  out["a_weights"], out["b_weights"] = arr(mm.a_weights), arr(mm.b_weights)
  return out


def _evaluator_cases() -> dict[str, Any]:
  """The evaluator/* cases (module doc).

  Returns:
    {case name: fn() -> {key: value}}.
  """
  out = {}
  for game in ("kuhn", "leduc"):
    for rule in ("sample", "argmax"):
      out[f"evaluator/{game}/head_to_head/{rule}"] = (
          lambda g=game, r=rule: _h2h(g, r)
      )
    out[f"evaluator/{game}/head_to_head/independent"] = (
        lambda g=game: _h2h_independent(g)
    )
    out[f"evaluator/{game}/payoff_matrix"] = lambda g=game: _payoff_matrix(g)
    out[f"evaluator/{game}/mixture_head_to_head"] = lambda g=game: _mixture(g)
  return out


# -- league --------------------------------------------------------------------
def _br_learner(policy, lr: float = 1e-2):
  """A small PPO learner of `policy`.

  Args:
    policy: The network policy.
    lr: The learning rate.

  Returns:
    The PGLearner.
  """
  cfg = pg.PGConfig(
      loss="ppo",
      lr=lr,
      lr_end=lr,
      anneal_lr=False,
      num_updates=1,
      max_grad_norm=1.0,
      update_epochs=2,
      num_minibatches=2,
      gamma=1.0,
      gae_lambda=1.0,
      pg=losses.PGLossConfig(clip_eps=0.2, vf_coef=0.5, clip_value=False),
      ent_coef=schedules.Schedule("constant", 0.01),
      kl_coef=schedules.Schedule("constant", 0.0),
  )
  return pg.PGLearner(policy, cfg)


def _run_league(
    controller,
    cfg,
    iterations: int,
    record_view: Callable[[Any, dict[str, Any], int], None],
) -> dict[str, Any]:
  """Runs a short kuhn League and digests what it drew and learned.

  Args:
    controller: The league controller.
    cfg: The LeagueConfig.
    iterations: The iterations to run.
    record_view: fn(league, out, iteration) recording per-iteration
      digests.

  Returns:
    {key: value}.
  """
  out: dict[str, Any] = {}
  pol = _kuhn_policy(16)
  with tempfile.TemporaryDirectory() as tmp:
    store = artifact_store.ArtifactStore(os.path.join(tmp, "store"))
    lg = loop.League(
        _kuhn_env(),
        pol,
        _br_learner(pol),
        controller,
        store,
        os.path.join(tmp, "run"),
        cfg,
        initial_members={"init@0": pol.init_params(jax.random.key(0))},
    )
    draws = []

    def before(league, ls):
      del league
      table, counts = ls.draw
      d = {}
      _table("t", table, d)
      draws.append([
          ls.name,
          int(ls.updates),
          d["t.seat_slot"],
          d["t.seat_member"],
          d["t.perspective_seat"],
          {str(k): int(v) for k, v in sorted(counts.items())},
      ])

    lg.before_update = before

    def on_iteration(league, it):
      if league.payoff is not None:
        _matrix(f"it{it}.payoff", league.payoff, out)
      record_view(league, out, it)

    lg.run(iterations, on_iteration=on_iteration)
    out["draws"] = draws
    out["stopped"] = [bool(lg.stopped), str(lg.stop_reason or "")]
    out["members"] = [
        [n, m.role, m.lifecycle, m.origin or "", int(m.admitted_at)]
        for n, m in lg.members.items()
    ]
    for n in sorted(lg.learners):
      out[f"learner.{n}.params"] = tree(lg.learner_params(n))
      out[f"learner.{n}.updates"] = int(lg.learners[n].updates)
    for n in sorted(lg.members):
      if lg.members[n].lifecycle != "evicted":
        out[f"member.{n}.params"] = tree(lg.member_params(n))
  return out


def _psro(payoff: str) -> dict[str, Any]:
  """The league/kuhn/psro_<payoff> case.

  Args:
    payoff: "evaluator" or "exact".

  Returns:
    {key: value}.
  """
  ctl = controller_lib.PSRO(meta_solver="nash", iterations=2, br_updates=2)
  cfg = loop.LeagueConfig(
      num_envs=32,
      num_steps=8,
      updates_per_iteration=1,
      pool_capacity=4,
      eval_deals=16,
      eval_max_rows=64,
      payoff=payoff,
      pool_rule=types.ARGMAX,
      seed=0,
  )

  def record(league, out, it):
    del league
    if ctl.last_meta is not None:
      out[f"it{it}.meta"] = [
          list(ctl.last_meta[0]),
          [fl(x) for x in ctl.last_meta[1]],
      ]

  return _run_league(ctl, cfg, 8, record)


def _scheduled_promotion() -> dict[str, Any]:
  """The league/kuhn/scheduled_promotion_gate case.

  Returns:
    {key: value}.
  """
  ctl = controller_lib.ScheduledPromotion(
      snapshot_every=1,
      harvest_every=1,
      max_pool=2,
      self_weight=0.5,
      pool_slots=1,
      seed=3,
      promotion="mean_above_zero",
      gate_deals=16,
  )
  cfg = loop.LeagueConfig(
      num_envs=32,
      num_steps=8,
      updates_per_iteration=1,
      pool_capacity=6,
      eval_max_rows=64,
      payoff="none",
      redraw="on_change",
      seed=1,
  )

  def record(league, out, it):
    del league
    out[f"it{it}.info"] = {
        k: fl(v) if isinstance(v, float) else int(v)
        for k, v in sorted(ctl.last_info.items())
    }

  out = _run_league(ctl, cfg, 5, record)
  for g, a in sorted(ctl.gate_attempts.items()):
    out[f"gate.{g}"] = {
        k: fl(v) if isinstance(v, float) else v for k, v in sorted(a.items())
    }
  assert ctl.gate_attempts, "the gate case played no gate cell"
  return out


def _self_play_all() -> dict[str, Any]:
  """The league/kuhn/self_play_all_seats case.

  Returns:
    {key: value}.
  """
  ctl = controller_lib.FixedMix("main", sampler.Mix(1.0, {}))
  cfg = loop.LeagueConfig(
      num_envs=16,
      num_steps=8,
      updates_per_iteration=2,
      pool_capacity=2,
      payoff="none",
      self_play_seats="all",
      seed=2,
  )
  return _run_league(ctl, cfg, 2, lambda league, out, it: None)


def _league_cases() -> dict[str, Any]:
  """The league/* cases (module doc).

  Returns:
    {case name: fn() -> {key: value}}.
  """
  return {
      "league/kuhn/psro_evaluator": lambda: _psro("evaluator"),
      "league/kuhn/psro_exact": lambda: _psro("exact"),
      "league/kuhn/scheduled_promotion_gate": _scheduled_promotion,
      "league/kuhn/self_play_all_seats": _self_play_all,
  }


# -- registry ------------------------------------------------------------------
def cases() -> dict[str, Any]:
  """{case name: fn() -> {key: value}}.

  Returns:
    The cases.
  """
  return {
      **_sampler_cases(),
      **_key_cases(),
      **_evaluator_cases(),
      **_league_cases(),
  }


def run_case(name: str) -> dict[str, Any]:
  """One case's digests, in their recorded (JSON) form.

  Args:
    name: the case name.

  Returns:
    {key: value}.
  """
  # A JSON round trip, so a computed case compares equal to its recorded form.
  return json.loads(json.dumps(cases()[name]()))


def main(argv) -> int:
  """Compares every case with the recorded goldens, or records them.

  Args:
    argv: the command line (--write, --only=a,b).

  Returns:
    The exit status.

  Raises:
    ValueError: recorded cases that are no longer defined.
  """
  write = "--write" in argv
  only = None
  for a in argv:
    if a.startswith("--only="):
      only = set(a.split("=", 1)[1].split(","))
  names = sorted(cases()) if only is None else sorted(only)
  recorded = {}
  if os.path.exists(GOLDEN):
    with open(GOLDEN, encoding="utf-8") as f:
      recorded = json.load(f)
  here = cpu_class.host()
  # A full --write re-records every case and the class: the old file's class
  # (or its lack of one) does not matter. A partial one and a compare do.
  full_write = write and only is None
  mismatch = class_mismatch(recorded) if recorded and not full_write else None
  if write and only is not None and mismatch:
    raise ValueError(f"refusing a partial --write: {mismatch}")
  if not write and mismatch and class_required():
    raise ValueError(f"{REQUIRE_CLASS_ENV}=1 and {mismatch}")
  print(f"host CPU class [{here}]")
  bad = 0
  for n in names:
    if not write and mismatch and n.startswith(XLA_CPU_PREFIXES):
      print(f"SKIP {n} (another CPU class)")
      continue
    got = run_case(n)
    if write:
      recorded[n] = got
      print(f"recorded {n} ({len(got)} keys)")
      continue
    want = recorded.get(n)
    ok = want == got
    bad += not ok
    print(f"{'ok  ' if ok else 'DIFF'} {n}")
  if mismatch and not write:
    print(f"SKIPPED the XLA:CPU cases: {mismatch}")
  if write:
    recorded[META] = {"cpu_class": here}
    unknown = set(recorded) - set(cases()) - {META}
    if unknown:
      raise ValueError(f"recorded cases no longer defined: {sorted(unknown)}")
    with open(GOLDEN, "w", encoding="utf-8") as f:
      json.dump(recorded, f, indent=1, sort_keys=True)
      f.write("\n")
    print(f"wrote {len(recorded)} cases -> {GOLDEN}")
  return 1 if bad else 0


if __name__ == "__main__":
  # A script: x64 on before any array is made, as the package's entry points
  # and conftest.py set it (never at import).
  frameworks.enable_x64()
  sys.exit(main(sys.argv[1:]))

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

"""The league at any player count.

Side seating, payoff views, gate, evals, end to end.

Three-player Kuhn poker (envs.kuhn_nplayer) is the free-for-all case (three
sides of one seat); two-player Kuhn pins that the two-player league is
unchanged (tables, config hash). Known answers: sampled field cells agree
with the exact field payoffs of the GameTree within 5 SE; the substitution
gate of a learner against its own snapshot is exactly 0 deal by deal (ARGMAX,
shared keys); one drawn opponent per side, so the two opponent seats of a
three-player table hold different members on some rows.
"""

# pylint: disable=protected-access

import dataclasses
import hashlib
import json
import sys
import types as pytypes

import jax
import numpy as np
import pytest

from open_spiel.python.scale import train
from open_spiel.python.scale.configs import kuhn3_league_smoke
from open_spiel.python.scale.core import hash_compat
from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.eval import exploitability as expl
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.league import artifact_store
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import controller as controller_lib
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import mixture
from open_spiel.python.scale.league import payoff_views
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.league import sampler
from open_spiel.python.scale.league.tests import league_fixtures
from open_spiel.python.scale.policies import registry

KUHN3 = "kuhn_nplayer"


def _policy3(width: int = 16):
  """The kuhn3 smoke network policy.

  Args:
    width: The hidden width.

  Returns:
    The policy.
  """
  return registry.build("network", kuhn3_league_smoke.policy_kwargs((width,)))


def _cfg(**kw):
  """A small LeagueConfig.

  Args:
    **kw: Fields over the defaults.

  Returns:
    The LeagueConfig.
  """
  base = dict(
      num_envs=48,
      num_steps=8,
      updates_per_iteration=1,
      pool_capacity=4,
      eval_deals=64,
      payoff="evaluator",
      seed=1,
  )
  base.update(kw)
  return loop.LeagueConfig(**base)


def _league3(tmp_path, controller, cfg=None, members=2, env=None, initial=None):
  """A three-player kuhn League.

  Args:
    tmp_path: The test's temporary directory.
    controller: The controller.
    cfg: The LeagueConfig (default _cfg()).
    members: The initial members when `initial` is None.
    env: The env (default kuhn_nplayer, 3 players).
    initial: {name: params} of the initial members.

  Returns:
    The League.
  """
  pol = _policy3()
  learner = league_fixtures.br_learner(pol)
  store = artifact_store.ArtifactStore(str(tmp_path / "store"))
  if initial is None:
    initial = {
        f"init{i}@0": league_fixtures.init_params(pol, i)
        for i in range(members)
    }
  env = env or common.build_env_by_name(KUHN3)
  return loop.League(
      env,
      pol,
      learner,
      controller,
      store,
      str(tmp_path / "run"),
      cfg or _cfg(),
      initial_members=initial,
  )


def _pool_mix(lg):
  """The uniform mix over the League's placed members.

  Args:
    lg: The League.

  Returns:
    The sampler.Mix.
  """
  names = [n for n in lg.placed() if n]
  return sampler.Mix(0.0, {n: 1.0 / len(names) for n in names})


# -- seating -------------------------------------------------------------------
def test_kuhn3_tables_draw_one_member_per_side(tmp_path):
  """Kuhn3 tables draw one member per side."""
  lg = _league3(
      tmp_path,
      controller_lib.FixedMix(
          "main", sampler.Mix(0.0, {"init0@0": 0.5, "init1@0": 0.5})
      ),
      _cfg(payoff="none"),
  )
  lg.tick(0)
  ls = lg.learners["main"]
  table, _, counts, _ = lg._draw(ls, np.int64(0))
  b = lg.config.num_envs
  persp = np.asarray(table.perspective_seat)
  assert persp.shape == (b,)  # one learner seat per row (M = 1)
  np.testing.assert_array_equal(persp, np.arange(b) % 3)
  slot, member = np.asarray(table.seat_slot), np.asarray(table.seat_member)
  rows = np.arange(b)
  assert np.all(slot[rows, persp] == 0)
  others = [
      [(slot[r, s], member[r, s]) for s in range(3) if s != persp[r]]
      for r in rows
  ]
  # never the learner's slot
  assert all(o[0][0] >= 1 and o[1][0] >= 1 for o in others)
  assert any(o[0] != o[1] for o in others)  # drawn per side, not one per table
  assert sum(counts.values()) == 2 * b  # two pool sides per row
  assert set(counts) == {"init0@0", "init1@0"}


def test_kuhn3_lineup_seats_self_and_pool_sides(tmp_path):
  """Kuhn3 lineup seats self and pool sides."""
  lu = sides_lib.lineup(("learner", "self", "pool"))
  lg = _league3(
      tmp_path,
      controller_lib.FixedMix("main", sampler.Mix(0.0, {"init0@0": 1.0})),
      _cfg(payoff="none", lineup=lu),
  )
  lg.tick(0)
  table, _, counts, _ = lg._draw(lg.learners["main"], np.int64(0))
  b = lg.config.num_envs
  slot = np.asarray(table.seat_slot)
  persp = np.asarray(table.perspective_seat)
  for r in range(b):
    s_self = (persp[r] + 1) % 3
    s_pool = (persp[r] + 2) % 3
    assert (
        slot[r, persp[r]] == 0 and slot[r, s_self] == 0 and slot[r, s_pool] >= 1
    )
  assert counts == {"init0@0": b, sampler.SELF: b}


def test_lineup_that_does_not_fit_the_sides_is_refused(tmp_path):
  """Lineup that does not fit the sides is refused."""
  with pytest.raises(ValueError, match="fills"):
    _league3(
        tmp_path,
        controller_lib.FixedMix("main", sampler.Mix(1.0, {})),
        _cfg(payoff="none", lineup=sides_lib.lineup(("learner", "pool"))),
    )


def test_two_player_draw_is_table_from_mix(tmp_path):
  """Two player draw is table from mix."""
  pol = league_fixtures.kuhn_policy()
  store = artifact_store.ArtifactStore(str(tmp_path / "store"))
  mix = sampler.Mix(0.25, {"a@0": 0.5, "b@0": 0.25})
  lg = loop.League(
      league_fixtures.env(),
      pol,
      league_fixtures.br_learner(pol),
      controller_lib.FixedMix("main", mix),
      store,
      str(tmp_path / "run"),
      _cfg(payoff="none", num_envs=37),
      initial_members={
          "a@0": league_fixtures.init_params(pol, 0),
          "b@0": league_fixtures.init_params(pol, 1),
      },
  )
  lg.tick(0)
  ls = lg.learners["main"]
  for g in range(4):
    table, _, counts, _ = lg._draw(ls, np.int64(g))
    old = sampler.sample(
        sampler.LeagueSlots(
            num_players=2, batch=37, pools=lg.all_pools(), mix=mix
        ),
        rng.fold_in64(lg._k_draw, np.int64(g)),
    )
    for x, y in zip(table, old.table):
      np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert counts == old.counts


# -- config hash ---------------------------------------------------------------
def test_league_config_hash_leaves_out_added_fields_at_default(tmp_path):
  """League config hash leaves out added fields at default."""
  cfg = _cfg(payoff="none")
  added = set(hash_compat.added_fields(loop.LeagueConfig))
  assert added == {"lineup", "payoff_view", "eval_design", "fixed_layout"}
  before = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in added}
  assert loop.hashed_config(cfg) == before
  pol = league_fixtures.kuhn_policy()
  lg = loop.League(
      league_fixtures.env(),
      pol,
      league_fixtures.br_learner(pol),
      controller_lib.FixedMix("main", sampler.Mix(1.0, {})),
      artifact_store.ArtifactStore(str(tmp_path / "store")),
      str(tmp_path / "run"),
      cfg,
  )
  spec = lg.slot_policy.spec
  want = hashlib.sha256(
      json.dumps(
          dict(
              config=before,
              controller="FixedMix",
              arch=spec.arch_name,
              arch_kwargs=dict(spec.arch_kwargs),
              kind=lg.driver.kind,
          ),
          sort_keys=True,
          default=str,
      ).encode()
  ).hexdigest()
  assert lg._config_hash == want


# -- payoff views --------------------------------------------------------------
def test_kuhn3_field_cells_match_exact_field_payoffs(tmp_path):
  """Kuhn3 field cells match exact field payoffs."""
  ctl = controller_lib.FixedMix("main", sampler.Mix(1.0, {}))
  sampled = _league3(tmp_path / "s", ctl, _cfg(eval_deals=1024, num_envs=8))
  exact = _league3(
      tmp_path / "e",
      controller_lib.FixedMix("main", sampler.Mix(1.0, {})),
      _cfg(payoff="exact", num_envs=8),
  )
  assert sampled.view_kind == exact.view_kind == "field"
  ps, pe = sampled.evaluate(), exact.evaluate()
  assert ps.names == pe.names == ("init0@0", "init1@0")
  assert np.all(ps.se > 0) and np.all(pe.se == 0)
  z = np.abs(ps.mean - pe.mean) / ps.se
  assert np.all(z < 5.0), (ps.mean, pe.mean, ps.se)
  # the exact self-play diagonal of a zero-sum game, rotation-balanced: 0
  np.testing.assert_allclose(np.diag(pe.mean), 0.0, atol=1e-12)


def test_exact_field_payoff_is_symmetric_payoff_at_two_players():
  """Exact field payoff is symmetric payoff at two players."""
  pol = league_fixtures.kuhn_policy()
  tree = expl.GameTree(league_fixtures.env())
  ta = tree.policy_table(pol, league_fixtures.init_params(pol, 0), types.SAMPLE)
  tb = tree.policy_table(pol, league_fixtures.init_params(pol, 1), types.SAMPLE)
  va, vb = mixture.field_payoff(
      tree, sides_lib.Sides.individual(2, True), ta, tb
  )
  assert va == pytest.approx(mixture.symmetric_payoff(tree, ta, tb), abs=1e-12)
  assert vb == pytest.approx(-va, abs=1e-12)


def test_field_scores_and_league_table(tmp_path):
  """Field scores and league table."""
  lg = _league3(
      tmp_path,
      controller_lib.FixedMix("main", sampler.Mix(1.0, {})),
      _cfg(eval_deals=32, num_envs=8),
  )
  lg.tick(0)
  pm = lg.payoff
  assert pm.names == ("init0@0", "init1@0", "main")
  scores = lg.field_scores(n_deals=32)
  assert [f.name for f in scores] == list(pm.names)
  for f in scores:
    assert np.isfinite(f.score.mean) and f.score.n == 32
    assert f.placement.shape == (3,) and f.placement.sum() == pytest.approx(1.0)
    assert set(f.winrates) <= set(pm.names)
  out = payoff_views.league_table(pm, lg.view_kind, scores)
  assert "field of j" in out and "main" in out


# -- capability checks ---------------------------------------------------------
@pytest.mark.parametrize(
    "make,refusal",
    [
        (
            lambda: controller_lib.PSRO(meta_solver="nash"),
            "linear program.*3 sides",
        ),
        (
            lambda: controller_lib.PSRO(meta_solver="regret_matching"),
            "two-player zero-sum matrix solver",
        ),
        (lambda: controller_lib.GatedPool("main"), "two-player game"),
        (lambda: controller_lib.ContinualExploiter("main"), "two-player game"),
        (lambda: controller_lib.BROracle("main"), "two-player game"),
        (lambda: controller_lib.PBT(["a", "b"]), "two-player game"),
        (
            lambda: controller_lib.Composite([
                controller_lib.FixedMix("x", sampler.Mix(1.0, {})),
                controller_lib.PSRO(meta_solver="prd"),
            ]),
            "two-player zero-sum matrix solver",
        ),
    ],
)
def test_pairwise_controllers_are_refused_at_more_than_two_sides(
    tmp_path, make, refusal
):
  """Pairwise controllers are refused at more than two sides."""
  with pytest.raises(ValueError, match=refusal):
    _league3(tmp_path, make())


def test_uniform_psro_and_declared_support_are_allowed(tmp_path):
  """Uniform PSRO and declared support are allowed."""
  _league3(tmp_path / "a", controller_lib.PSRO(meta_solver="uniform"))
  # the symmetric meta-game
  for name in (
      "replicator",
      "fictitious_play",
      "alpha_rank",
  ):
    _league3(tmp_path / name, controller_lib.PSRO(meta_solver=name))

  class Declares(controller_lib.GatedPool):
    """A pairwise controller that declares support for three sides."""

    def supports_sides(self, sides):
      """Supports exactly three sides.

      Args:
        sides: the league's sides.

      Returns:
        Whether there are three sides.
      """
      return sides.num_sides == 3

  _league3(tmp_path / "b", Declares("main"))


def test_eval_design_is_checked_against_the_sides(tmp_path):
  """Eval design is checked against the sides."""
  _league3(
      tmp_path / "a",
      controller_lib.FixedMix("main", sampler.Mix(1.0, {})),
      _cfg(eval_design="cyclic+reversed"),
  )
  pol = league_fixtures.kuhn_policy()
  with pytest.raises(ValueError, match="swap2"):
    loop.League(
        league_fixtures.env(),
        pol,
        league_fixtures.br_learner(pol),
        controller_lib.FixedMix("main", sampler.Mix(1.0, {})),
        artifact_store.ArtifactStore(str(tmp_path / "s")),
        str(tmp_path / "r"),
        _cfg(eval_design="cyclic"),
    )


# -- gate ----------------------------------------------------------------------
def test_kuhn3_gate_is_the_substitution_test(tmp_path):
  """Kuhn3 gate is the substitution test."""
  p0 = league_fixtures.init_params(_policy3(), 0)
  # every logit negated: other actions
  neg = jax.tree_util.tree_map(lambda x: -x, p0)
  lg = _league3(
      tmp_path,
      controller_lib.FixedMix("main", sampler.Mix(1.0, {})),
      _cfg(payoff="none", num_envs=8),
      initial={"init0@0": p0, "neg@0": neg},
  )
  lg.tick(0)
  lg._apply([controller_lib.SpawnLearner("b", "main", init_from="init0@0")], 0)
  same = lg.gate_cell(
      controller_lib.GateRequest(
          subject="b", opponent="init0@0", generation=1, seed=5, deals=64
      )
  )
  assert same.mean == 0.0 and same.se == 0.0 and same.n_deals == 64
  other = lg.gate_cell(
      controller_lib.GateRequest(
          subject="b", opponent="neg@0", generation=2, seed=5, deals=64
      )
  )
  assert np.isfinite(other.mean) and other.se > 0
  # A field (any sides): the controller's GateRequest carries it; the league
  # plays it.
  req = controller_lib.GateRequest(
      subject="b",
      opponent="init0@0",
      generation=3,
      seed=5,
      deals=32,
      field=sampler.Mix(0.0, {"init0@0": 0.5, "neg@0": 0.5}),
  )
  f = lg.gate_cell(req)
  assert f.mean == 0.0 and f.se == 0.0
  with pytest.raises(ValueError, match="members only"):
    controller_lib.GateRequest(
        subject="b",
        opponent="init0@0",
        generation=4,
        seed=5,
        deals=32,
        field=sampler.Mix(0.5, {"init0@0": 0.5}),
    )
  with pytest.raises(ValueError, match="members only"):
    lg.gate_cell(
        pytypes.SimpleNamespace(**{
            **vars(req),
            "generation": 4,
            "field": sampler.Mix(0.5, {"init0@0": 0.5}),
        })
    )


# -- in-loop evals -------------------------------------------------------------
def test_vs_uniform_is_a_field_eval_and_vs_mcts_refuses():
  """Vs uniform is a field eval and vs MCTS refuses."""
  env = common.build_env_by_name(KUHN3)
  pol = _policy3()
  ent = match.Entrant(
      "main@0", pol, league_fixtures.init_params(pol, 0), types.ARGMAX
  )
  for design in ("auto", "cyclic+reversed"):
    ev = runner.InLoopEval("vs_uniform", every=1, deals=16, design=design)
    evaluator = runner.build_generic_eval(ev, env)
    est, n, vr = runner.eval_estimate(ev, evaluator, ent, jax.random.key(0))
    assert np.isfinite(est.mean) and est.n == 16 and n == 16 and np.isnan(vr)
  with pytest.raises(ValueError, match="two-player"):
    runner.build_generic_eval(
        runner.InLoopEval("vs_mcts", every=1, deals=4, simulations=2), env
    )
  with pytest.raises(ValueError, match="field design"):
    runner.build_generic_eval(
        runner.InLoopEval("vs_uniform", every=1, deals=4, design="cyclic"),
        league_fixtures.env(),
    )


# -- end to end ----------------------------------------------------------------
def test_kuhn3_league_smoke_runs_end_to_end(tmp_path):
  """Kuhn3 league smoke runs end to end."""
  cfg = kuhn3_league_smoke.get_config(psro_iterations=2, br_updates=3)
  cfg = dataclasses.replace(
      cfg, league=dataclasses.replace(cfg.league, num_envs=64, eval_deals=32)
  )
  res = train.run(cfg, checkpoint_dir=str(tmp_path / "ckpt"))
  lg = res.league
  assert lg.stopped and lg.view_kind == "field"
  names = [m.name for m in lg.members.values()]
  assert names == ["init0@0", "psro1@3", "psro2@3"]
  pm = lg.payoff
  assert (
      pm.mean.shape == (3, 3)
      and np.all(np.isfinite(pm.mean))
      and np.all(pm.se > 0)
  )
  assert res.rows and "meta.init0@0" in res.rows[-1]
  scores = lg.field_scores(n_deals=32)
  assert len(scores) == 3
  print(payoff_views.league_table(pm, lg.view_kind, scores))


def test_kuhn3_smoke_config_shape():
  """Kuhn3 smoke config shape."""
  cfg = common.load_config("kuhn3_league_smoke")
  assert cfg.env == KUHN3 and cfg.controller == "psro"
  assert cfg.controller_kwargs["meta_solver"] == "uniform"
  assert cfg.league.payoff == "evaluator" and cfg.league.lineup is None
  env = common.build_env(cfg)
  assert sides_lib.sides_of(env).num_sides == 3


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

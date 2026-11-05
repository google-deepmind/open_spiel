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

"""Controllers and meta-solvers beyond two interchangeable sides.

* PSRO at a field view solves the symmetric meta-game (view.symmetric):
  replicator on a strict-ESS game gives its sigma (2 - sqrt 2, sqrt 2 - 1); a
  cyclic game raises NotConvergedError; the LP solvers are refused at three
  sides and for general-sum two sides; two uneven constant-sum sides take the
  LP on the centred role-averaged matrix.
* GatedPool, ContinualExploiter, BROracle and _lcb refuse anything but a
  pair view, with a message that says why.
* The League's symmetric game is exact: on kuhn3 with payoff="exact",
  fitness(e_j)[i] is the field cell (i, j), and fitness(sigma) equals a
  brute-force sum over every seat of the subject and every ordered draw of
  its co-players from sigma (GameTree.expected_returns).
* Gate false-admission: 400 trials of the substitution gate on
  nplayer_matrix odd_one_out_3p under an exact null (a pure strategy and
  the uniform one score 0 against a uniform field), SAMPLE everywhere,
  ci_above_zero: the one-sided 95% Clopper-Pearson upper bound of the
  admission rate is <= 0.08.
* End to end: PSRO with fictitious play on kuhn3 (sampled field payoffs)
  and PSRO with the LP and with fictitious play on team_pennies_2v1 (a
  pair against one seat; its sampled field cells agree with the exact
  role-averaged ones within 5 SE).
"""

# pylint: disable=protected-access

import dataclasses
import itertools
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as sps

from open_spiel.python.scale import train
from open_spiel.python.scale.configs import kuhn3_league_smoke
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import nplayer_matrix as nm
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.league import artifact_store
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import controller as controller_lib
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import meta_solvers as ms
from open_spiel.python.scale.league import sampler
from open_spiel.python.scale.league.tests import league_fixtures
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.policies import registry as pol_registry
from open_spiel.python.scale.policies import tabular

IND3 = sides_lib.Sides.individual(3, True)
TWO_V_ONE = sides_lib.Sides((0, 1, 0), True)
GS2 = sides_lib.Sides.individual(2, False)


# -- PSRO on synthetic views ---------------------------------------------------
def _view(
    names,
    kind="field",
    num_sides=3,
    constant_sum=True,
    symmetric=None,
    mean=None,
):
  """A LeagueView of fixed members `names`.

  Args:
    names: The members.
    kind: The view kind.
    num_sides: The sides.
    constant_sum: Whether the game is constant-sum.
    symmetric: The symmetric meta-game, or None.
    mean: The payoff means, or None for zeros.

  Returns:
    The LeagueView.
  """
  k = len(names)
  members = tuple(
      controller_lib.MemberInfo(n, "init" if i == 0 else "psro", "fixed")
      for i, n in enumerate(names)
  )
  pm = controller_lib.payoff_matrix(
      names, np.zeros((k, k)) if mean is None else mean
  )
  return controller_lib.LeagueView(
      iteration=0,
      members=members,
      learners=(),
      payoff=pm,
      kind=kind,
      num_sides=num_sides,
      constant_sum=constant_sum,
      symmetric=symmetric,
  )


def _mix_of(acts):
  """The mix of the one SetMix among `acts`.

  Args:
    acts: Controller actions.

  Returns:
    The sampler.Mix.
  """
  (mix,) = [a.mix for a in acts if isinstance(a, controller_lib.SetMix)]
  return mix


NAMES2 = ("init0@0", "psro1@1")


P_STAR = np.sqrt(2.0) - 1.0


def _ess(names=NAMES2):
  """A three-player game with a strict interior ESS.

  Args:
    names: The two strategies.

  Returns:
    The SymmetricGame.
  """
  # f1 - f0 = (1 - p)^2 - 2 p^2: a strict ESS at p* = sqrt(2) - 1.
  return ms.SymmetricGame.from_function(
      3, names, lambda i, m: 0.0 if i == 0 else (1.0, 0.0, -2.0)[sum(m)]
  )


def test_psro_replicator_at_three_sides_is_the_symmetric_ess():
  """PSRO replicator at three sides is the symmetric ess."""
  ctl = controller_lib.PSRO(meta_solver="replicator", iterations=2)
  assert ctl.supports_sides(IND3) and ctl.needs_symmetric_game
  mix = _mix_of(ctl.step(_view(NAMES2, symmetric=_ess())))
  assert mix.self_play == 0.0
  np.testing.assert_allclose(
      [mix.weights[n] for n in NAMES2], [1.0 - P_STAR, P_STAR], atol=1e-5
  )
  assert ctl.last_regret < 1e-8
  assert ctl.last_meta[0] == NAMES2


def test_psro_at_a_field_view_without_the_symmetric_game_fails_loud():
  """PSRO at a field view without the symmetric game fails loud."""
  with pytest.raises(ValueError, match="needs the symmetric meta-game"):
    controller_lib.PSRO(meta_solver="fictitious_play").step(_view(NAMES2))
  # uniform reads no payoff
  mix = _mix_of(controller_lib.PSRO(meta_solver="uniform").step(_view(NAMES2)))
  assert mix.weights == {n: 0.5 for n in NAMES2}


def test_psro_replicator_on_a_cyclic_meta_game_raises():
  """PSRO replicator on a cyclic meta game raises."""
  big_a = np.array([[0.0, -1.0, 2.0], [1.0, 0.0, -3.0], [-2.0, 3.0, 0.0]])
  names = ("init0@0", "psro1@1", "psro2@2")
  game = ms.SymmetricGame.from_tensor(
      big_a[:, :, None] + big_a[:, None, :], names
  )
  with pytest.raises(ms.NotConvergedError):
    controller_lib.PSRO(
        meta_solver="replicator", solver_kwargs=dict(iterations=40000)
    ).step(_view(names, symmetric=game))


def test_psro_alpha_rank_default_population_is_the_number_of_sides():
  """PSRO alpha rank default population is the number of sides."""
  ctl = controller_lib.PSRO(meta_solver="alpha_rank")
  mix = _mix_of(ctl.step(_view(NAMES2, symmetric=_ess())))
  want = ms.solve_symmetric(
      "alpha_rank", _ess(), alpha=5.0, population_size=3
  ).sigma
  np.testing.assert_allclose([mix.weights[n] for n in NAMES2], want, atol=1e-12)


@pytest.mark.parametrize(
    "name", ["nash", "max_entropy_nash", "prd", "regret_matching"]
)
def test_matrix_solvers_are_refused_at_three_sides_and_general_sum(name):
  """Matrix solvers are refused at three sides and general sum."""
  ctl = controller_lib.PSRO(meta_solver=name)
  with pytest.raises(ms.CapabilityError, match="3 sides"):
    ctl.supports_sides(IND3)
  with pytest.raises(ms.CapabilityError, match="general-sum"):
    ctl.supports_sides(GS2)
  with pytest.raises(ms.CapabilityError, match="general-sum"):
    controller_lib.PSRO(meta_solver=name).step(
        _view(NAMES2, kind="pair", num_sides=2, constant_sum=False)
    )
  assert ctl.supports_sides(sides_lib.Sides.individual(2, True))
  # two uneven constant-sum sides: a zero-sum game
  assert ctl.supports_sides(TWO_V_ONE)


def test_symmetric_solvers_read_a_general_sum_pair_matrix_as_symmetric():
  """Symmetric solvers read a general sum pair matrix as the symmetric game."""
  # A stag hunt, U[i, j] = i's payoff against j (general-sum): the symmetric
  # game is the matrix itself.
  u = np.array([[4.0, 0.0], [3.0, 2.0]])
  ctl = controller_lib.PSRO(meta_solver="replicator")
  assert ctl.supports_sides(GS2)
  mix = _mix_of(
      ctl.step(
          _view(NAMES2, kind="pair", num_sides=2, constant_sum=False, mean=u)
      )
  )
  w = np.array([mix.weights[n] for n in NAMES2])
  sol = ms.solve_symmetric("replicator", ms.SymmetricGame(2, NAMES2, u))
  np.testing.assert_allclose(w, sol.sigma, atol=0)
  assert ctl.last_regret < 1e-6


def test_lp_at_two_uneven_sides_solves_the_centred_role_matrix():
  """Lp at two uneven sides solves the centred role matrix."""
  # u(i; j) with u + u^T = 1 (constant side sum 1): strategy 0 dominates.
  u = np.array([[0.5, 1.5], [-0.5, 0.5]])
  game = ms.SymmetricGame(2, NAMES2, u)
  mix = _mix_of(
      controller_lib.PSRO(meta_solver="nash").step(
          _view(NAMES2, num_sides=2, symmetric=game)
      )
  )
  np.testing.assert_allclose(
      [mix.weights[n] for n in NAMES2], [1.0, 0.0], atol=1e-9
  )
  with pytest.raises(ms.CapabilityError, match="general-sum"):
    controller_lib.PSRO(meta_solver="nash").step(
        _view(NAMES2, num_sides=2, constant_sum=False, symmetric=game)
    )


# -- pairwise controllers ------------------------------------------------------
@pytest.mark.parametrize(
    "make",
    [
        lambda: controller_lib.GatedPool("main"),
        lambda: controller_lib.ContinualExploiter("main"),
        lambda: controller_lib.BROracle("main"),
    ],
)
def test_pairwise_controllers_refuse_field_views(make):
  """Pairwise controllers refuse field views."""
  for sides in (IND3, TWO_V_ONE):
    with pytest.raises(ValueError, match=r"two-player game.*field score"):
      make().supports_sides(sides)
  assert make().supports_sides(sides_lib.Sides.individual(2, True))
  assert make().supports_sides(sides_lib.Sides.partnerships(4, 2, True))


def test_lcb_and_gated_pool_refuse_a_field_view_at_step():
  """Lcb and gated pool refuse a field view at step."""
  names = ("init0@0",)
  view = dataclasses.replace(
      _view(names),
      learners=(controller_lib.LearnerInfo("main", "main", 1),),
      payoff=controller_lib.payoff_matrix(
          ("init0@0", "main"), np.zeros((2, 2))
      ),
  )
  with pytest.raises(
      ValueError, match="_lcb reads the payoff matrix as a two-player game"
  ):
    controller_lib._lcb(view, "main", "init0@0", 1.96)
  with pytest.raises(ValueError, match="GatedPool reads"):
    controller_lib.GatedPool("main").step(view)


def test_view_kind_is_checked():
  """View kind is checked."""
  with pytest.raises(ValueError, match="pair view has two sides"):
    _view(NAMES2, kind="pair", num_sides=3)
  with pytest.raises(ValueError, match="kind"):
    _view(NAMES2, kind="team")


# -- the League's symmetric game is exact --------------------------------------
def _policy3(width=16):
  """The kuhn3 smoke network policy.

  Args:
    width: The hidden width.

  Returns:
    The policy.
  """
  return pol_registry.build(
      "network", kuhn3_league_smoke.policy_kwargs((width,))
  )


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


def _league(tmp_path, env, pol, controller, cfg, members=3):
  """A League of `members` initial members.

  Args:
    tmp_path: The test's temporary directory.
    env: The env.
    pol: The policy.
    controller: The controller.
    cfg: The LeagueConfig.
    members: The initial members.

  Returns:
    The League.
  """
  initial = {
      f"init{i}@0": league_fixtures.init_params(pol, i) for i in range(members)
  }
  return loop.League(
      env,
      pol,
      league_fixtures.br_learner(pol),
      controller,
      artifact_store.ArtifactStore(str(tmp_path / "store")),
      str(tmp_path / "run"),
      cfg,
      initial_members=initial,
  )


def test_kuhn3_symmetric_game_is_the_exact_iid_field_value(tmp_path):
  """Kuhn3 symmetric game is the exact iid field value."""
  env = common.build_env_by_name("kuhn_nplayer")
  lg = _league(
      tmp_path,
      env,
      _policy3(),
      controller_lib.PSRO(meta_solver="alpha_rank"),
      _cfg(payoff="exact"),
  )
  pm = lg.evaluate()
  game = lg._symmetric_game()
  names = list(game.names)
  assert names == ["init0@0", "init1@0", "init2@0"]
  idx = [list(pm.names).index(n) for n in names]
  for j in range(3):
    np.testing.assert_allclose(
        game.fitness(np.eye(3)[j]), pm.mean[idx, idx[j]], atol=1e-12
    )
  sigma = np.array([0.5, 0.3, 0.2])
  tabs = [lg._table(n, 0) for n in names]
  want = np.zeros(3)
  for i in range(3):
    for s in range(3):  # the subject's seat
      for prof in itertools.product(range(3), repeat=2):
        seats = list(prof)
        seats.insert(s, i)
        r = lg._tree.expected_returns([tabs[x] for x in seats])
        want[i] += np.prod(sigma[list(prof)]) * float(r[s]) / 3.0
  np.testing.assert_allclose(game.fitness(sigma), want, atol=1e-12)
  view = lg.view()
  assert (
      view.kind == "field"
      and view.num_sides == 3
      and view.symmetric is not None
  )


# -- gate false admission ------------------------------------------------------
def test_substitution_gate_false_admission_under_an_exact_null():
  """Substitution gate false admission under an exact null."""
  env = nm.NPlayerMatrixEnv("odd_one_out_3p")
  pol = tabular.TabularPolicy(
      "nplayer_matrix", 3, 2, 3, f"{nm.__name__}:seat_index"
  )

  def ent(name, probs):
    """An Entrant playing `probs` at every seat.

    Args:
      name: the entrant's name.
      probs: its distribution over the two actions.

    Returns:
      The Entrant.
    """
    logits = jnp.log(jnp.clip(jnp.asarray(probs, jnp.float32), 1e-30, 1.0))
    return match.Entrant(
        name,
        pol,
        pol.params_from_logits(jnp.broadcast_to(logits, (3, 2))),
        types.SAMPLE,
    )

  subject, incumbent, field = (
      ent("s", [1.0, 0.0]),
      ent("i", [0.5, 0.5]),
      ent("f", [0.5, 0.5]),
  )
  # The null is exact: against two iid uniform co-players every action scores 0.
  for a in ([1.0, 0.0], [0.5, 0.5]):
    v = np.mean([
        nm.exact_returns(
            "odd_one_out_3p",
            [np.array(a) if q == r else np.array([0.5, 0.5]) for q in range(3)],
        )[r]
        for r in range(3)
    ])
    assert v == pytest.approx(0.0, abs=1e-12)
  ev = match.Evaluator(env, match.EvaluatorConfig(max_rows=4096))
  gate = controller_lib.SeededGate(seed=7, predicate="ci_above_zero")
  trials, admits, ses = 400, 0, []
  for t in range(trials):
    res = ev.substitute(
        subject, incumbent, [field], [1.0], 32, jax.random.key(gate.seed(t))
    )
    est = res.block_diff(0)
    cell = controller_lib.GateCell(
        generation=t,
        subject="s",
        opponent="i",
        seed=gate.seed(t),
        n_deals=32,
        mean=float(est.mean),
        se=float(est.se),
        lo=float(est.lo),
        hi=float(est.hi),
    )
    ses.append(cell.se)
    admits += bool(gate.passes("s", "i", cell))
  assert min(ses) > 0.0  # the statistic did vary
  upper = float(sps.beta.ppf(0.95, admits + 1, trials - admits))
  print(
      f"false admissions {admits}/{trials}, one-sided 95% upper bound "
      f"{upper:.4f}"
  )
  assert upper <= 0.08, (admits, upper)


# -- end to end ----------------------------------------------------------------
def test_kuhn3_psro_fictitious_play_runs_end_to_end(tmp_path):
  """Kuhn3 PSRO fictitious play runs end to end."""
  cfg = kuhn3_league_smoke.get_config(psro_iterations=2, br_updates=3)
  cfg = dataclasses.replace(
      cfg,
      controller_kwargs={
          **cfg.controller_kwargs,
          "meta_solver": "fictitious_play",
      },
      league=dataclasses.replace(cfg.league, num_envs=64, eval_deals=32),
  )
  res = train.run(cfg, checkpoint_dir=str(tmp_path / "ckpt"))
  lg = res.league
  assert lg.stopped and lg.view_kind == "field"
  assert [m.name for m in lg.members.values()] == [
      "init0@0",
      "psro1@3",
      "psro2@3",
  ]
  ctl = [
      c
      for c in loop._controller_leaves(lg.controller)
      if isinstance(c, controller_lib.PSRO)
  ][0]
  pop, sigma = ctl.last_meta
  assert pop[:2] == ("init0@0", "psro1@3") and sigma.sum() == pytest.approx(1.0)
  assert np.isfinite(ctl.last_regret) and ctl.last_regret >= 0.0
  game = lg._symmetric_game()
  assert game.num_players == 3 and game.names == (
      "init0@0",
      "psro1@3",
      "psro2@3",
  )
  print(
      "kuhn3 PSRO(fictitious_play) last meta",
      pop,
      sigma,
      "regret",
      ctl.last_regret,
  )


def _two_v_one(width=8):
  """The team_pennies_2v1 env and a small network policy.

  Args:
    width: The hidden width.

  Returns:
    (env, policy).
  """
  env = env_registry.make("nplayer_matrix", variant="team_pennies_2v1")
  enc = enc_registry.get(
      enc_registry.family_name(
          "nplayer_matrix_seat",
          env_registry.spec("nplayer_matrix", variant="team_pennies_2v1"),
      )
  )
  kw = common.network_policy_kwargs(
      env.name,
      enc.name,
      list(enc.shape),
      env.num_actions,
      env.num_players,
      catalog.mlp([width], "relu", "separate"),
  )
  return env, pol_registry.build("network", kw)


def test_two_v_one_field_cells_average_over_the_role(tmp_path):
  """Two v one field cells average over the role."""
  env, pol = _two_v_one()

  def ctl():
    """A fresh pure self-play controller.

    Returns:
      The FixedMix controller.
    """
    return controller_lib.FixedMix("main", sampler.Mix(1.0, {}))

  sampled = _league(
      tmp_path / "s",
      env,
      pol,
      ctl(),
      _cfg(eval_deals=2048, num_envs=8),
      members=2,
  )
  exact = _league(
      tmp_path / "e",
      env,
      pol,
      ctl(),
      _cfg(payoff="exact", num_envs=8),
      members=2,
  )
  assert sampled.view_kind == exact.view_kind == "field"
  assert sampled.eval_design == "cyclic+classes"
  ps, pe = sampled.evaluate(), exact.evaluate()
  assert np.all(ps.se > 0)
  z = np.abs(ps.mean - pe.mean) / ps.se
  assert np.all(z < 5.0), (ps.mean, pe.mean, ps.se)
  # Side-level zero-sum, role-averaged: the self-play diagonal is exactly 0.
  np.testing.assert_allclose(np.diag(pe.mean), 0.0, atol=1e-12)


@pytest.mark.parametrize("solver", ["nash", "fictitious_play"])
def test_two_v_one_psro_runs_end_to_end(tmp_path, solver):
  """Two v one PSRO runs end to end."""
  env, pol = _two_v_one()
  ctl = controller_lib.PSRO(meta_solver=solver, iterations=2, br_updates=2)
  lg = _league(tmp_path, env, pol, ctl, _cfg(eval_deals=64), members=1)
  assert lg.view_kind == "field" and lg.eval_design == "cyclic+classes"
  lg.run(20)
  assert lg.stopped, lg.stop_reason
  assert [m.name for m in lg.members.values()] == [
      "init0@0",
      "psro1@2",
      "psro2@2",
  ]
  pop, sigma = ctl.last_meta
  assert pop[:2] == ("init0@0", "psro1@2") and sigma.sum() == pytest.approx(1.0)
  game = lg._symmetric_game()
  u = game.matrix()
  # side-level zero-sum, role-averaged
  np.testing.assert_allclose(u + u.T, 0.0, atol=1e-9)
  print(f"2v1 PSRO({solver}) last meta", pop, sigma)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

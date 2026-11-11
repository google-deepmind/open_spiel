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

"""The League loop on kuhn (CPU).

The League loop on kuhn (CPU): Arena, learner, sampler, Evaluator,
ArtifactStore and controller composed.

  * no recompile when the mix changes: the training Arena's rollout/reset and
    the learner's update compile once while mixes, admissions, evictions and
    spawns change between iterations (fixed K);
  * admissions are artifacts: an admitted member is an artifact
    (ckpt.artifact) registered in the ArtifactStore under its id, and loading it
    gives the learner's params at admission;
  * the drawn opponents follow the mix (stratified counts recorded per update);
  * determinism: the same seed gives the same params and payoffs;
  * CopyMutate (PBT) is shrink * src + perturb * fresh, and its no-op is one;
  * a tiny PSRO run on kuhn (Nash meta-solver, PPO best responses, Evaluator
    payoffs) reaches exact exploitability < 0.1 (uniform 0.458) in 10
    iterations of 60 updates; pipeline-PSRO runs its
    concurrent best-response slots.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import types
from open_spiel.python.scale.eval import exploitability as expl
from open_spiel.python.scale.league import artifact_store
from open_spiel.python.scale.league import controller as controller_lib
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import mixture
from open_spiel.python.scale.league import sampler
from open_spiel.python.scale.league.tests import league_fixtures

UNIFORM_KUHN = 0.4583333333333333


class Scripted:
  """A controller that plays a fixed action list per iteration (first round).

  Attributes:
    script: iteration -> actions.
    seen: the views stepped, in order.
  """

  def __init__(self, script):
    self.script = script
    self.seen = []
    self._done = set()

  def step(self, view):
    """Returns the scripted actions of an iteration, once.

    Args:
      view: the league view.

    Returns:
      The iteration's actions; none if already returned.
    """
    self.seen.append(view)
    if view.iteration in self._done:
      return []
    self._done.add(view.iteration)
    return list(self.script.get(view.iteration, []))


def _league(tmp_path, controller, cfg=None, initial=None, payoff="evaluator"):
  """A Kuhn league with a BR learner.

  Args:
    tmp_path: The folder.
    controller: The controller.
    cfg: The LeagueConfig (default small).
    initial: The initial members (default one).
    payoff: The payoff mode.

  Returns:
    The League.
  """
  pol = league_fixtures.kuhn_policy()
  learner = league_fixtures.br_learner(pol)
  cfg = cfg or loop.LeagueConfig(
      num_envs=64,
      num_steps=8,
      updates_per_iteration=2,
      pool_capacity=6,
      eval_deals=64,
      payoff=payoff,
      seed=3,
  )
  store = artifact_store.ArtifactStore(str(tmp_path / "store"))
  initial = (
      initial
      if initial is not None
      else {"init@0": league_fixtures.init_params(pol, 0)}
  )
  return loop.League(
      league_fixtures.env(),
      pol,
      learner,
      controller,
      store,
      str(tmp_path / "run"),
      cfg,
      initial_members=initial,
  )


def test_no_recompile_when_the_mix_changes(tmp_path):
  """No recompile when the mix changes."""
  script = {
      0: [controller_lib.SpawnLearner("main", "main")],
      1: [controller_lib.SetMix("main", sampler.Mix(0.5, {"init@0": 0.5}))],
      2: [
          controller_lib.Admit("main", "snap@4", "lagged", "fixed"),
          controller_lib.SetMix(
              "main", sampler.Mix(0.0, {"init@0": 0.25, "snap@4": 0.75})
          ),
      ],
      3: [
          controller_lib.SpawnLearner("other", "exploiter", init_from="snap@4"),
          controller_lib.SetMix("other", sampler.Mix(0.0, {"main": 1.0})),
          controller_lib.SetMix("main", sampler.Mix(0.2, {"other": 0.8})),
          controller_lib.CopyMutate("main", "other", 0.5, 0.1),
      ],
      4: [
          controller_lib.SetLifecycle("snap@4", "evicted"),
          controller_lib.RetireLearner("other"),
          controller_lib.SetMix("main", sampler.Mix(1.0, {})),
      ],
  }
  lg = _league(tmp_path, Scripted(script))
  lg.run(6)
  # Reaches private _cache_size, _rollout_jit of the code under test.
  assert lg.arena._rollout_jit._cache_size() == 1  # pylint: disable=protected-access
  # Reaches private _cache_size, _reset_jit of the code under test.
  assert lg.arena._reset_jit._cache_size() == 1  # pylint: disable=protected-access
  # Reaches private _cache_size, _update_jit of the code under test.
  assert lg.learner._update_jit._cache_size() == 1  # pylint: disable=protected-access
  assert int(lg.learners["main"].updates) == 12  # 6 iterations x 2
  assert isinstance(lg.learners["main"].frames, np.int64)
  assert isinstance(lg.learners["main"].updates, np.int64)
  assert lg.learners["main"].frames == np.int64(12 * 64 * 8)
  # Pool shape never changed (fixed K).
  for leaf in jax.tree_util.tree_leaves(lg.pool.params):
    assert leaf.shape[0] == 6


def test_admission_is_a_registered_artifact(tmp_path):
  """Admission is a registered artifact."""
  script = {
      0: [controller_lib.SpawnLearner("main", "main")],
      2: [controller_lib.Admit("main", "snap@4", "lagged", "fixed")],
  }
  lg = _league(tmp_path, Scripted(script))
  lg.run(3)
  assert set(lg.store.ids()) == {"init@0", "snap@4"}
  art = lg.store.load("snap@4")
  assert art.manifest.arch_name == "network" and art.manifest.step == 4
  assert art.manifest.default_rule == lg.config.pool_rule
  assert league_fixtures.leaves_equal(art.params, lg.member_params("snap@4"))
  assert (
      lg.members["snap@4"].origin == "main"
      and lg.members["snap@4"].admitted_at == 2
  )
  with pytest.raises(ValueError):  # ids are unique
    # Reaches private _apply of the code under test.
    lg._apply([controller_lib.Admit("main", "snap@4", "lagged", "fixed")], 3)  # pylint: disable=protected-access


def test_draws_follow_the_mix(tmp_path):
  """Draws follow the mix."""
  script = {
      0: [
          controller_lib.SpawnLearner("main", "main"),
          controller_lib.SetMix("main", sampler.Mix(0.25, {"init@0": 0.75})),
      ]
  }
  lg = _league(tmp_path, Scripted(script))
  lg.run(1)
  counts = lg.learners["main"].draw_counts
  assert counts and all(c == {"self": 16, "init@0": 48} for c in counts)


def test_set_mix_refuses_unknown_members(tmp_path):
  """Set mix refuses unknown members."""
  lg = _league(
      tmp_path,
      Scripted({
          0: [
              controller_lib.SpawnLearner("main", "main"),
              controller_lib.SetMix("main", sampler.Mix(0.0, {"ghost@1": 1.0})),
          ]
      }),
  )
  with pytest.raises(KeyError):
    lg.run(1)


def test_pool_capacity_is_enforced(tmp_path):
  """Pool capacity is enforced."""
  script = {0: [controller_lib.SpawnLearner("main", "main")]}
  script[1] = [
      controller_lib.Admit("main", f"s{i}@2", "lagged", "fixed")
      for i in range(5)
  ]
  lg = _league(tmp_path, Scripted(script))
  with pytest.raises(RuntimeError, match="pool_capacity"):
    lg.run(2)


def test_league_is_deterministic(tmp_path):
  """League is deterministic."""

  def go(sub):
    """Runs the scripted league under `sub`.

    Args:
      sub: the run's subdirectory.

    Returns:
      The league after its run.
    """
    script = {
        0: [
            controller_lib.SpawnLearner("main", "main"),
            controller_lib.SetMix("main", sampler.Mix(0.5, {"init@0": 0.5})),
        ],
        1: [controller_lib.Admit("main", "snap@2", "lagged", "fixed")],
    }
    lg = _league(tmp_path / sub, Scripted(script))
    lg.run(3)
    return lg

  a, b = go("a"), go("b")
  assert league_fixtures.leaves_equal(
      a.learner_params("main"), b.learner_params("main")
  )
  np.testing.assert_array_equal(a.payoff.mean, b.payoff.mean)
  assert a.payoff.names == b.payoff.names


def test_copy_mutate_math_and_noop(tmp_path):
  """Copy mutate math and noop."""
  script = {
      0: [
          controller_lib.SpawnLearner("a", "pbt"),
          controller_lib.SpawnLearner("b", "pbt"),
      ]
  }
  lg = _league(tmp_path, Scripted(script))
  lg.run(1)
  before = lg.learners["b"].state
  # Reaches private _apply of the code under test.
  lg._apply([controller_lib.CopyMutate("b", "b", 1.0, 0.0)], 1)  # pylint: disable=protected-access
  # copying a learner onto itself changes nothing
  assert lg.learners["b"].state is before
  src = lg.learner_params("a")
  # Reaches private _mutations of the code under test.
  m = lg._mutations  # pylint: disable=protected-access
  # Reaches private _apply of the code under test.
  lg._apply([controller_lib.CopyMutate("a", "b", 0.5, 0.25)], 1)  # pylint: disable=protected-access
  # The fresh params of mutation m: init_params at fold_in64(k_mutate, m).
  fresh = lg.policy.init_params(rng.fold_in64(lg._k_mutate, m))  # pylint: disable=protected-access
  got = lg.learner_params("b")
  want = jax.tree_util.tree_map(lambda s, f: 0.5 * s + 0.25 * f, src, fresh)
  for x, y in zip(
      jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)
  ):
    np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-6)
    assert x.dtype == jnp.float32


def test_view_scores_and_payoff_cover_members_and_learners(tmp_path):
  """View scores and payoff cover members and learners."""
  script = {
      0: [
          controller_lib.SpawnLearner("main", "main"),
          controller_lib.SetMix("main", sampler.Mix(0.0, {"init@0": 1.0})),
      ]
  }
  ctl = Scripted(script)
  lg = _league(tmp_path, ctl, payoff="exact")
  lg.run(3)
  v = ctl.seen[-1]
  assert v.payoff.names == ("init@0", "main")
  # one per iteration after the first training
  assert len(v.learner("main").scores) == 2
  mean, se = v.cell("main", "init@0")
  assert se == 0.0 and v.learner("main").scores[-1] == pytest.approx(mean)


def _meta_exploitability(lg, ctl, tree):
  """The exploitability of the controller's last meta-strategy.

  Args:
    lg: The league.
    ctl: The controller.
    tree: The game tree.

  Returns:
    The exploitability.
  """
  names, w = ctl.last_meta
  tables = [
      tree.policy_table(lg.policy, lg.member_params(n), lg.config.pool_rule)
      for n in names
  ]
  return tree.exploitability(mixture.mixture_table(tree, tables, w))


def test_tiny_psro_kuhn_reaches_low_exploitability(tmp_path):
  """Tiny psro kuhn reaches low exploitability."""
  ctl = controller_lib.PSRO(meta_solver="nash", iterations=10, br_updates=60)
  cfg = loop.LeagueConfig(
      num_envs=256,
      num_steps=8,
      updates_per_iteration=60,
      pool_capacity=13,
      eval_deals=4096,
      payoff="evaluator",
      pool_rule=types.ARGMAX,
      seed=0,
  )
  lg = _league(tmp_path, ctl, cfg=cfg)
  tree = expl.GameTree(league_fixtures.env())
  curve = []

  def on_iteration(league, it):
    del it
    if ctl.last_meta is not None and len(ctl.last_meta[0]) > 1:
      curve.append(_meta_exploitability(league, ctl, tree))

  lg.run(30, on_iteration=on_iteration)
  assert lg.stopped and "psro" in lg.stop_reason
  pop = [n for n, m in lg.members.items() if m.role in ("init", "psro")]
  assert len(pop) == 11
  # Reaches private _cache_size, _update_jit of the code under test.
  assert lg.learner._update_jit._cache_size() == 1  # pylint: disable=protected-access
  final = _meta_exploitability(lg, ctl, tree)
  print(
      f"PSRO kuhn: exploitability curve {[round(x, 4) for x in curve]}, final "
      f"{final:.4f} "
      f"(uniform {UNIFORM_KUHN:.4f}); meta {ctl.last_meta}"
  )
  # The final exploitability varies widely across seeds at this budget; 0.1
  # (22% of uniform's 0.4583) bounds the spread.
  assert final < 0.1
  assert curve and curve[-1] < curve[0]
  # The Evaluator's payoff matrix is antisymmetric with a zero diagonal.
  pm = lg.payoff.mean
  np.testing.assert_allclose(pm, -pm.T, atol=1e-12)
  # Every admitted BR is in the store.
  assert set(pop) <= set(lg.store.ids())


def test_pipeline_psro_kuhn_concurrent_slots(tmp_path):
  """Pipeline psro kuhn concurrent slots."""
  ctl = controller_lib.PSRO(
      meta_solver="nash",
      iterations=3,
      br_updates=6,
      pipeline_depth=2,
      active_weight_share=0.5,
  )
  cfg = loop.LeagueConfig(
      num_envs=64,
      num_steps=8,
      updates_per_iteration=3,
      pool_capacity=8,
      eval_deals=64,
      pool_rule=types.ARGMAX,
      seed=1,
  )
  lg = _league(tmp_path, ctl, cfg=cfg)
  lg.run(20)
  assert lg.stopped
  admitted = [n for n, m in lg.members.items() if m.role == "psro"]
  assert len(admitted) == 3
  # The level-1 worker drew rows against the live level-0 worker.
  assert any("br0" in c for c in lg.retired["br1"].draw_counts)
  # Reaches private _cache_size, _rollout_jit of the code under test.
  assert lg.arena._rollout_jit._cache_size() == 1  # pylint: disable=protected-access
  # Four BR slots were spawned over the run; the learner compiled once.
  assert lg.learner._update_jit._cache_size() == 1  # pylint: disable=protected-access


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

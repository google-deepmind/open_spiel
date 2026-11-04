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

"""league.population: a PSRO run's final population.

league.population: a PSRO run's final population (members + meta-strategy)
read from a run dir, and the exact exploitability of its meta-mixture.

Known answers:
  * a league run (kuhn, PSRO, exact payoffs) read back at its final
    step: the members and weights are the controller's last meta-strategy
    (names and f64 weights), each member's params bitwise equal to the League's,
    final_step(dir) is the iteration count;
  * mixture_exploitability with all weight on one member equals that
    member's exploitability (eval.exploitability.exploitability);
  * an INDEPENDENT best-response computation by pure-strategy enumeration
    (every pure kuhn strategy of the responding seat, 2^6 per seat, against
    the members played whole games with their weights) gives the same
    NashConv as the realization-weighted mixture table, to 1e-12;
  * a mixture of two kuhn Nash equilibria (alpha 0 and 1/3) has
    exploitability 0 (a convex combination of Nash strategies of a two-player
    zero-sum game is Nash).
"""

import itertools
import pickle
import sys

from etils import epath
import numpy as np
import pytest

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.eval import exploitability as expl
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import population
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.league.tests import league_fixtures
from open_spiel.python.scale.testing import populations


def _tab_artifact(tmp_path, name, bet):
  """Writes a Kuhn tabular artifact that bets with probability `bet`.

  Args:
    tmp_path: The folder.
    name: The artifact name.
    bet: The betting probability.

  Returns:
    The loaded Artifact.
  """
  d = str(tmp_path / name)
  tab = populations.kuhn_tabular()
  artifact.write_artifact(
      d,
      1,
      tab,
      tab.params_from_logits(populations.bet_logits(bet)),
      rules_sha="r" * 8,
      config_hash="c",
      algo="test",
      env_name="kuhn_poker",
      default_rule=types.SAMPLE,
  )
  return artifact.load_artifact(d, 1)


def _pop(tmp_path, bets, weights, name="pop"):
  """A population of tabular members.

  Args:
    tmp_path: The folder.
    bets: Each member's betting probability.
    weights: The weights.
    name: The population name.

  Returns:
    The Population.
  """
  arts = tuple(
      _tab_artifact(tmp_path, f"{name}_m{i}", b) for i, b in enumerate(bets)
  )
  return population.Population(
      name=name,
      game="kuhn_poker",
      members=arts,
      member_names=tuple(f"m{i}" for i in range(len(arts))),
      weights=np.asarray(weights, np.float64),
      source="test",
  )


def test_population_validation(tmp_path):
  """Population validation."""
  with pytest.raises(ValueError, match="weights"):
    _pop(tmp_path, [np.zeros(12)], [0.5])
  with pytest.raises(ValueError, match="weights"):
    _pop(tmp_path, [np.zeros(12), np.ones(12)], [1.2, -0.2], name="p2")


def test_one_hot_mixture_is_the_member(tmp_path):
  """One hot mixture is the member."""
  env = kuhn.KuhnEnv()
  tree = expl.GameTree(env)
  bets = [populations.kuhn_nash(0.0), np.full(12, 0.7), np.zeros(12)]
  pop = _pop(tmp_path, bets, [0.0, 1.0, 0.0])
  got = population.mixture_exploitability(env, pop, types.SAMPLE, tree)
  ref = expl.exploitability(
      env, pop.members[1].policy, pop.members[1].params, tree=tree
  )
  assert got["exploitability"] == pytest.approx(ref.exploitability, abs=1e-12)
  assert got["num_members"] == 3 and got["support"] == 1


def _pure_br(tree, tables, weights, p):
  """The best pure-response value of player p against a mixture, by enumeration.

  Args:
    tree: The game tree.
    tables: The mixture's tables.
    weights: Their weights.
    p: The responding player.

  Returns:
    The best value.
  """
  rows = np.flatnonzero(tree.player == p)
  best = -np.inf
  for choice in itertools.product(
      *[np.flatnonzero(tree.legal[r]) for r in rows]
  ):
    s = tree.uniform_table().copy()
    for r, a in zip(rows, choice):
      s[r] = 0.0
      s[r, a] = 1.0
    v = 0.0
    for t, w in zip(tables, weights):
      seats = [t] * tree.num_players
      seats[p] = s
      v += w * float(tree.expected_returns(seats)[p])
    best = max(best, v)
  return best


def test_nash_conv_equals_pure_strategy_enumeration(tmp_path):
  """Nash conv equals pure strategy enumeration."""
  env = kuhn.KuhnEnv()
  tree = expl.GameTree(env)
  rng = np.random.default_rng(0)
  bets = [rng.random(12), rng.random(12), np.zeros(12)]
  w = np.array([0.3, 0.5, 0.2])
  pop = _pop(tmp_path, bets, w)
  tables = [
      tree.policy_table(m.policy, m.params, types.SAMPLE) for m in pop.members
  ]
  on = 0
  for ti, wi in zip(tables, w):
    for tj, wj in zip(tables, w):
      on = on + wi * wj * tree.expected_returns([ti, tj])
  nash_conv = sum(_pure_br(tree, tables, w, p) - float(on[p]) for p in range(2))
  got = population.mixture_exploitability(env, pop, types.SAMPLE, tree)
  assert got["nash_conv"] == pytest.approx(nash_conv, abs=1e-12)
  assert got["exploitability"] == pytest.approx(nash_conv / 2, abs=1e-12)
  assert got["exploitability"] > 0.05


def test_mixture_of_nash_is_nash(tmp_path):
  """Mixture of nash is nash."""
  env = kuhn.KuhnEnv()
  pop = _pop(
      tmp_path,
      [populations.kuhn_nash(0.0), populations.kuhn_nash(1 / 3)],
      [0.4, 0.6],
  )
  got = population.mixture_exploitability(env, pop)
  # tabular f32 logits of the Nash tables: ~3e-9
  assert abs(got["exploitability"]) < 1e-6


def test_league_run_population_reads_back(tmp_path):
  """League run population reads back."""
  pol = league_fixtures.kuhn_policy(16)
  cfg = runner.LeagueRunConfig(
      name="psro_kuhn_tiny",
      env="kuhn_poker",
      policy_kwargs=dict(pol.spec.arch_kwargs),
      learner=kinds.PG(league_fixtures.br_learner(pol).cfg),
      league=loop.LeagueConfig(
          num_envs=32,
          num_steps=8,
          updates_per_iteration=2,
          pool_capacity=6,
          eval_deals=16,
          payoff="exact",
          seed=0,
      ),
      controller="psro",
      controller_kwargs=dict(meta_solver="nash", iterations=2, br_updates=2),
      num_iterations=10,
      num_initial=1,
      artifact_learner="",
  )
  d = str(tmp_path / "run")
  res = runner.run(cfg, d)
  lg = res.league
  assert lg.stopped
  assert population.final_step(d) == lg.iteration
  pop = population.load_population(d, lg.iteration)
  names, sigma = lg.controller.last_meta
  assert pop.member_names == tuple(names) and pop.source == "league"
  np.testing.assert_array_equal(pop.weights, np.asarray(sigma, np.float64))
  # init + 2 BRs
  assert abs(pop.weights.sum() - 1.0) < 1e-12 and len(names) == 3
  for n, art in zip(names, pop.members):
    assert league_fixtures.leaves_equal(art.params, lg.member_params(n)), n
  assert pop.name == f"{d}@{lg.iteration}" and pop.game == "kuhn_poker"
  with pytest.raises(FileNotFoundError):
    population.load_population(d, lg.iteration + 7)
  x = population.mixture_exploitability(kuhn.KuhnEnv(), pop)
  assert 0.0 <= x["exploitability"] <= 0.5


def test_read_state_applies_the_registered_translations(tmp_path, monkeypatch):
  """Reading a state applies the registered translations.

  League.restore and LeagueReader.load read a state through one function,
  which applies every registered state translation.

  Args:
    tmp_path: pytest's tmp_path.
    monkeypatch: pytest's monkeypatch.
  """
  monkeypatch.setattr(loop, "_STATE_TRANSLATIONS", [])
  path = epath.Path(tmp_path) / "state.pkl"
  path.write_bytes(pickle.dumps({"old": 1}))
  loop.register_state_translation(
      lambda d: {"new": d["old"] + 1, "state_version": loop.STATE_VERSION}
  )
  assert loop.read_state(path) == {
      "new": 2,
      "state_version": loop.STATE_VERSION,
  }


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

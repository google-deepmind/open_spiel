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

"""Search-operator conformance and the infoset-invariance test.

* check_operator holds for every operator family: leduc PIMC SearchPolicy,
gin determinized PIMC and the HOST_DRIVEN test operator;
* every registered architecture (and runtime target emitter) that decides
by search or emits targets has an infoset-invariance fixture; a search
architecture registered without one makes walk() report it; every
policies-registry register call site in the package is a module this test
imports, so the walk sees them all;
* every fixture passes, and the negative controls fail: a gin env whose
info_state leaks an opponent card, an operator that reads the side being
evaluated, a no-op hide_fn, and a host operator with a cache across calls.
"""

import ast
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale import plugins
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.envs.connect_four import teacher
from open_spiel.python.scale.envs.gin import endgame
from open_spiel.python.scale.envs.gin import rule_bot
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.networks import policy as network_policy
from open_spiel.python.scale.policies import opening
from open_spiel.python.scale.policies import operator_vocab
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import rollout
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies import search
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import check_operator
from open_spiel.python.scale.testing import fixtures
from open_spiel.python.scale.testing import host_operator_fixtures
from open_spiel.python.scale.testing import infoset_invariance
from open_spiel.python.scale.testing.errors import ContractError
from open_spiel.python.scale.testing.tests import infoset_invariance_fixtures

PKG = subprocess_env.PACKAGE
PKG_DIR = subprocess_env.PACKAGE_DIR
#: Every module of the package (outside tests) that registers an architecture,
#: imported above.
REGISTERING_MODULES = tuple(
    m.__name__
    for m in (
        uniform,
        tabular,
        teacher,
        endgame,
        rule_bot,
        routed,
        opening,
        network_policy,
        search,
        rollout,
        fixtures,
        host_operator_fixtures,
    )
)


def _in_package(module: str) -> bool:
  """Whether `module` is the package or inside it.

  Args:
    module: The module name.

  Returns:
    True when it is.
  """
  return module == PKG or module.startswith(PKG + ".")


_steps = infoset_invariance_fixtures.operator_steps  # pylint: disable=invalid-name


# gin_pimc runs in a shard file (the fixtures' SHARDED_OPERATOR_FAMILY).
@pytest.mark.parametrize("family", ["leduc_pimc", "host_echo"])
def test_check_operator(family):
  """Each operator family passes check_operator.

  Args:
    family: The operator family.
  """
  rep = infoset_invariance_fixtures.check_operator_family(family)
  assert rep["decider"] == "search" and rep["acting"] > 0
  assert rep["value_bias"] in operator_vocab.VALUE_BIASES


def test_check_operator_catches_an_aux_target_value():
  """check_operator catches an operator that emits AUX_TARGET_VALUE."""
  env = leduc.LeducEnv()
  base = infoset_invariance_fixtures.leduc_net()
  bs = base.spec

  class Bad(host_operator_fixtures.HostEchoOperator):
    """An echo operator that emits its value as AUX_TARGET_VALUE."""

    def evaluate(self, params, carry, step, key):
      """Evaluates the acting rows (only those, in the Arena).

      Args:
        params: The params.
        carry: The carry; not updated.
        step: The SeatStep.
        key: The PRNG key.

      Returns:
        The PolicyOutput, with AUX_TARGET_VALUE added.
      """
      out = super().evaluate(params, carry, step, key)
      return out._replace(
          aux=dict(out.aux, **{types.AUX_TARGET_VALUE: out.value})
      )

  op = Bad(dict(arch_name=bs.arch_name, arch_kwargs=dict(bs.arch_kwargs)))
  with pytest.raises(ContractError, match="AUX_TARGET_VALUE"):
    check_operator.check_operator(
        op, base.init_params(jax.random.key(0)), _steps(env)
    )


# -- coverage ------------------------------------------------------------------
def _policies_registry_names(tree) -> set[str]:
  """The local names a module binds to the policies registry module.

  Args:
    tree: The module's AST.

  Returns:
    The names.
  """
  out = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
        ".policies"
    ):
      out |= {a.asname or a.name for a in node.names if a.name == "registry"}
  return out


def test_every_registration_site_is_imported_here():
  """Every module that registers an architecture is imported by this test."""
  sites = set()
  for root, _, files in os.walk(PKG_DIR):
    if "tests" in os.path.relpath(root, PKG_DIR).split(os.sep):
      continue
    for f in files:
      if not f.endswith(".py"):
        continue
      path = os.path.join(root, f)
      tree = ast.parse(open(path, encoding="utf-8").read())
      names = _policies_registry_names(tree)
      for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in names
        ):
          rel = os.path.relpath(path, PKG_DIR)[:-3].replace(os.sep, ".")
          if rel == "__init__":
            sites.add(PKG)
            continue
          if rel.endswith(".__init__"):
            rel = rel[:-9]
          sites.add(f"{PKG}.{rel}")
  missing = sorted(sites - set(REGISTERING_MODULES))
  assert (
      not missing
  ), f"architecture registrations not imported by the walk test: {missing}"


def _open_walk():
  """walk() over the package's own architectures and runtime emitters.

  One that another package registers in this process is that package's to
  check.

  Returns:
    The architectures without a fixture.
  """
  reg = set(registry.registered())
  return [
      a
      for a in infoset_invariance.walk()
      if a not in reg or _in_package(registry.policy_class(a).__module__)
  ]


def test_every_search_or_target_architecture_has_a_fixture():
  """Every search or target-emitting architecture has a fixture."""
  with host_operator_fixtures.registered():
    assert not _open_walk(), _open_walk()
  assert {
      "search",
      host_operator_fixtures.HostEchoOperator.ARCH,
      "DaggerPolicy",
      "BehaviorMixture",
  } <= set(infoset_invariance.FIXTURES)


def test_open_builtins_and_fixtures_load_from_the_package_alone():
  """The built-ins and shipped fixtures import no sibling package.

  The registry's built-ins, the default slot factory and the walk over the
  shipped fixtures import no sibling package, and every lazily named module
  is a module of the package. A fresh interpreter: this process may hold
  other packages.
  """
  # A sibling package is one whose name extends this one's (PKG + "_x").
  code = (
      f"import sys\nPKG = {PKG!r}\n"
      f"from {PKG}.policies import registry\n"
      f"from {PKG}.league import slot_factories\n"
      f"from {PKG}.testing import host_operator_fixtures, infoset_invariance\n"
      f"from {PKG}.testing.tests import infoset_invariance_fixtures\n"
      "registry.registered()\nregistry.build('uniform', "
      "{'game': 'leduc', 'num_actions': 3, 'num_players': 2})\n"
      "slot_factories.get('', {})\n"
      "with host_operator_fixtures.registered():\n"
      "  assert infoset_invariance.walk() == [], infoset_invariance.walk()\n"
      "assert all(m.startswith(PKG + '.') for m in registry.LAZY.values()), "
      "registry.LAZY\nassert all(m.startswith(PKG + '.') for m in "
      "slot_factories.LAZY.values())\n"
      "bad = sorted(m for m in sys.modules if m.startswith(PKG) and m != PKG\n"
      "             and not m.startswith(PKG + '.'))\n"
      "assert not bad, bad\nprint('ok')\n"
  )
  base = {k: v for k, v in os.environ.items() if k != plugins.ENV}
  r = subprocess_env.run_python(["-c", code], base_env=base)
  assert r.returncode == 0 and r.stdout.strip().endswith("ok"), (
      r.stdout,
      r.stderr,
  )


def test_a_search_architecture_without_a_fixture_fails_the_walk():
  """A search architecture without a fixture is listed by walk()."""

  class DummySearch(host_operator_fixtures.HostEchoOperator):
    """A search architecture registered with no fixture."""

    ARCH = "testing.dummy_search_no_fixture"

  with registry.temporary(DummySearch.ARCH, DummySearch):
    assert DummySearch.ARCH in infoset_invariance.walk()
    with_fixture = dict(infoset_invariance.FIXTURES, **{DummySearch.ARCH: None})
    assert not infoset_invariance.walk(with_fixture)


# -- the test and its negative controls ----------------------------------------
# Shard 0 of the fixture items (infoset_invariance_fixtures.NUM_SHARDS).
def pytest_generate_tests(metafunc):
  """Gives the shard test shard 0's items, at collection (not import)."""
  infoset_invariance_fixtures.parametrize_shard(metafunc, 0)


def test_every_fixture_is_infoset_invariant(key, case, plies_):
  """Every item of shard 0 is infoset-invariant."""
  del key
  infoset_invariance_fixtures.check_invariant(case, plies_)


def test_the_shards_cover_every_history_length_of_every_case_once():
  """The shards check each history length of each case once, in real files."""
  num_shards = infoset_invariance_fixtures.NUM_SHARDS
  got = []
  for shard in range(num_shards):
    for key, case, p in infoset_invariance_fixtures.shard_items(shard):
      got.append((key, case.name, p))
  want = []
  for key in sorted(infoset_invariance.FIXTURES):
    for case in infoset_invariance.cases(key):
      want.extend((key, case.name, (t,)) for t in sorted(set(case.plies)))
  assert sorted(got) == sorted(want) and len(got) == len(set(got))
  here = os.path.dirname(os.path.abspath(__file__))
  for k in range(1, num_shards):
    shard_file = os.path.join(here, f"infoset_invariance_shard{k}_test.py")
    assert os.path.exists(shard_file), k


class _LeakyGin(gin.GinEnv):
  """info_state also carries the opponent's lowest hidden card."""

  def info_state(self, s, seat):
    """Returns the info state with the opponent's lowest card as knock_card.

    Args:
      s: The world state.
      seat: The observing seat.

    Returns:
      The leaky info state.
    """
    info = super().info_state(s, seat)
    opp = (
        jnp.where(
            jnp.asarray(seat) == 0,
            s.game["player1_hand"],
            s.game["player0_hand"],
        )
        > 0
    )
    leak = jnp.argmax(opp).astype(jnp.int32)
    return info._replace(knock_card=leak)


def test_an_env_that_leaks_hidden_cards_fails():
  """An env whose info state leaks a hidden card fails the test."""
  case = infoset_invariance_fixtures.gin_case(rows=8)
  with pytest.raises(infoset_invariance.NotInvariantError, match="env leaks"):
    infoset_invariance.check_infoset_invariance(
        case._replace(env=_LeakyGin(), name="leaky_gin")
    )


def test_an_operator_that_reads_the_evaluated_side_fails():
  """An operator that reads which side is evaluated fails the test."""
  env, _, sp = infoset_invariance_fixtures.leduc_search()

  class GlobalReader(type(sp)):
    """A search operator that bumps its logits on the partner side."""

    def evaluate(self, params, carry, step, key):
      """Evaluates the acting rows (only those, in the Arena).

      Args:
        params: The params.
        carry: The carry; not updated.
        step: The SeatStep.
        key: The PRNG key.

      Returns:
        The PolicyOutput, logits plus 1 on the partner side.
      """
      out = super().evaluate(params, carry, step, key)
      # Reaches private _ACTIVE_SIDE of the code under test.
      side = infoset_invariance._ACTIVE_SIDE  # pylint: disable=protected-access
      bump = jnp.float32(1.0 if side == "partner" else 0.0)
      return out._replace(logits=out.logits + bump)

  bad = GlobalReader(sp.base, sp.env, sp.sampler, sp.cfg)
  case = infoset_invariance.InfosetCase(
      "global_reader",
      env,
      bad,
      sp.init_params(jax.random.key(0)),
      leduc.hide_opponent_card,
      plies=(1, 2),
      rows=8,
      seed=1,
  )
  with pytest.raises(
      infoset_invariance.NotInvariantError, match="policy's output"
  ):
    infoset_invariance.check_infoset_invariance(case)


def test_a_noop_hide_fn_fails():
  """A hide_fn that changes nothing fails the test as vacuous."""
  case = infoset_invariance_fixtures.dagger_case()
  with pytest.raises(infoset_invariance.NotInvariantError, match="no-op"):
    infoset_invariance.check_infoset_invariance(
        case._replace(hide_fn=lambda e, s, q, k: s, name="noop")
    )


def test_a_host_operator_with_a_cache_fails():
  """A host operator whose output depends on earlier calls fails the test."""
  case = infoset_invariance_fixtures.host_echo_case()

  class Cached(host_operator_fixtures.HostEchoOperator):
    """An echo operator whose logits count its calls.

    Attributes:
      calls: The number of calls so far.
    """

    def evaluate(self, params, carry, step, key):
      """Evaluates the acting rows (only those, in the Arena).

      Args:
        params: The params.
        carry: The carry; not updated.
        step: The SeatStep.
        key: The PRNG key.

      Returns:
        The PolicyOutput, logits plus the call count.
      """
      out = super().evaluate(params, carry, step, key)
      self.calls = getattr(self, "calls", 0) + 1
      return out._replace(logits=out.logits + np.float32(self.calls))

  op = Cached(dict(case.policy.spec.arch_kwargs["base"]))
  with pytest.raises(infoset_invariance.NotInvariantError):
    infoset_invariance.check_infoset_invariance(
        case._replace(policy=op, name="cached", fresh=None)
    )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

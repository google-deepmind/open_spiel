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

"""The league's control plane runs without jax.

The league's control plane runs without jax, and the loop names no array
framework.

The sampler, the snapshot schedule, the controllers, the meta-solvers, the
artifact store's index, the payoff accounting, core.rng and core.slots import
and pass their own test files in a subprocess where jax cannot be imported.
Also pinned here: the scheduled-promotion seating draws equal jax.random's,
pool storage per policy type (and the registration hook), the driver hooks
the loop calls instead of naming jax, the PBT parameter edit, and ArenaSource
snapshots of jax arrays, typed keys and registered leaf types.
"""

import importlib.util
import inspect
import os
import pickle
import re
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale import deps
from open_spiel.python.scale import plugins
from open_spiel.python.scale.ckpt import snapshot
from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.league import commit
from open_spiel.python.scale.league import controller
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import pool_params

_PKG_DIR = subprocess_env.PACKAGE_DIR
_PKG = subprocess_env.PACKAGE  # the package's import name

BLOCKED = (
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "orbax",
    "orbax.checkpoint",
    "chex",
    "distrax",
    "mctx",
)

# The control plane's own test files (under the package folder): each must
# import and pass without jax.
JAX_FREE_TESTS = tuple(
    os.path.join(_PKG_DIR, f)
    for f in (
        "core/tests/rng_test.py",
        "core/tests/slots_test.py",
        "league/tests/sampler_test.py",
        "league/tests/snapshot_schedule_test.py",
        "league/tests/controller_test.py",
        "league/tests/meta_solvers_test.py",
        "league/tests/artifact_store_index_test.py",
        "eval/tests/stats_test.py",
        "eval/tests/payoff_test.py",
    )
)

# Modules (under the package) that must import with jax unimportable.
JAX_FREE_MODULES = (_PKG,) + tuple(
    f"{_PKG}.{m}"
    for m in (
        "core.types",
        "core.rng",
        "core.slots",
        "core.frameworks",
        "league.sampler",
        "league.controller",
        "league.meta_solvers",
        "league.artifact_store",
        "league.pool_params",
        "eval.stats",
        "eval.payoff",
        "eval.match",
        "ckpt.artifact",
        "ckpt.snapshot",
        "policies.members",
    )
)

_BLOCK = f"import sys\nfor _m in {BLOCKED!r}:\n  sys.modules[_m] = None\n"
_LEAK = (
    "leaked = sorted(m for m in sys.modules if m.split('.')[0] in"
    f" {tuple(b.split('.', maxsplit=1)[0] for b in BLOCKED)!r} and"
    " sys.modules[m] is not None)\nassert not leaked, leaked\n"
)


def _run(code: str, timeout: int = 900) -> subprocess.CompletedProcess[str]:
  """Runs `code` in a child with no plugins.

  Args:
    code: The code.
    timeout: Seconds.

  Returns:
    The completed process.
  """
  # The open library alone: no deployment plugin (plugins.ENV) either.
  return subprocess_env.run_python(
      ["-c", code],
      env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", plugins.ENV: ""},
      timeout=timeout,
  )


def test_control_plane_modules_import_without_jax():
  """Control plane modules import without jax."""
  code = (
      _BLOCK
      + "".join(f"import {m}\n" for m in JAX_FREE_MODULES)
      + _LEAK
      + (
          "try:\n  import jax\nexcept ImportError:\n  pass\nelse:\n  raise "
          "AssertionError('jax')\n"
      )
  )
  r = _run(code)
  assert r.returncode == 0, r.stdout + r.stderr


def test_control_plane_tests_pass_without_jax():
  """Control plane tests pass without jax."""
  for f in JAX_FREE_TESTS:
    assert os.path.exists(f), f
  # The package is imported before pytest, as a test runner does, so pytest
  # binds the test files to it.
  code = (
      _BLOCK
      + (
          f"import {_PKG}\n"
          "import pytest\n"
          "rc = pytest.main(['-q', '-rs', '-p', 'no:cacheprovider', *"
          f"{list(JAX_FREE_TESTS)!r}])\n"
      )
      + _LEAK
      + "sys.exit(int(rc))\n"
  )
  r = _run(code)
  out = r.stdout + r.stderr
  assert r.returncode == 0, out[-6000:]
  summary = [l for l in out.splitlines() if re.search(r"\d+ passed", l)]
  assert summary, out[-3000:]
  last = summary[-1]
  assert "failed" not in last and "error" not in last, last
  assert int(re.search(r"(\d+) passed", last).group(1)) >= 60, last
  # No skips: every control-plane test runs without jax (a test that needs an
  # optional package this install lacks is the only exception).

  absent = [p for p in deps.OPTIONAL if importlib.util.find_spec(p) is None]
  skips = [
      l
      for l in out.splitlines()
      if l.startswith("SKIPPED")
      and not any(f"'{p}'" in l or f"needs {p}" in l for p in absent)
  ]
  assert not skips, (skips, last)  # -rs lists every skip's reason


# -- seating draws -------------------------------------------------------------
def test_scheduled_promotion_seating_equals_the_jax_random_draw():
  """ScheduledPromotion seats what jax.random.choice draws.

  core.rng keeps threefry: the seated members are what
  jax.random.choice(fold_in64(key(seed), round)) picks.
  """
  seed = 11
  ctl = controller.ScheduledPromotion(
      main="main",
      snapshot_every=10,
      harvest_every=10,
      max_pool=9,
      pool_slots=3,
      self_weight=0.9,
      seed=seed,
  )
  members = {}
  for r in range(40):
    learners = (
        () if r == 0 else (controller.LearnerInfo("main", "main", 10 * r),)
    )
    for a in ctl.step(
        controller.LeagueView(r, tuple(members.values()), learners, None)
    ):
      if isinstance(a, controller.Admit):
        members[a.member] = controller.MemberInfo(
            a.member, a.role, a.lifecycle, a.learner, r
        )
      elif isinstance(a, controller.SetLifecycle):
        members[a.member] = controller.MemberInfo(
            a.member, "snapshot", a.lifecycle, "main", r
        )
    pool = ctl.schedule.pool_steps(10 * r)
    m = min(3, len(pool))
    if m == len(pool):
      want = tuple(sorted(pool))
    else:
      k = x64.fold_in64(jax.random.key(seed), np.int64(r))
      idx = np.asarray(jax.random.choice(k, len(pool), (m,), replace=False))
      want = tuple(sorted(pool[int(i)] for i in idx))
    # Reaches private _seated of the code under test.
    assert ctl._seated == tuple(f"main@{s}" for s in want), r  # pylint: disable=protected-access


# -- pool storage --------------------------------------------------------------
class _Spec:

  def __init__(self, cap, fw):
    self.spec = types.PolicySpec("g", "o", 2, 2, False, cap, fw, "stub")


def test_pool_storage_per_policy_type():
  """Pool storage per policy type."""
  jaxp = pool_params.for_policy(_Spec(types.Capability.NATIVE_JAX, "jax"))
  listp = pool_params.for_policy(_Spec(types.Capability.HOST_DRIVEN, "jax"))
  assert isinstance(jaxp, pool_params.JaxStacked) and isinstance(
      listp, pool_params.HostList
  )
  tmpl = {"w": jnp.zeros((3,), jnp.float32), "n": jnp.int64(0)}
  big = jaxp.empty(tmpl, 4)
  assert big["w"].shape == (4, 3) and big["w"].dtype == jnp.float32
  assert big["n"].dtype == jnp.int64 and isinstance(big["w"], jax.Array)
  x = {"w": np.arange(3, dtype=np.float32), "n": np.int64(7)}
  new = jaxp.set(big, 2, x)
  want = jax.tree_util.tree_map(
      lambda b, v: b.at[2].set(jnp.asarray(v, b.dtype)), big, x
  )
  for a, b in zip(
      jax.tree_util.tree_leaves(new), jax.tree_util.tree_leaves(want)
  ):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
  np.testing.assert_array_equal(np.asarray(jaxp.get(new, 2)["w"]), [0, 1, 2])
  # pure: the input is unchanged
  assert float(np.asarray(big["w"]).sum()) == 0.0
  jaxp.check(x, tmpl)
  with pytest.raises(ValueError, match="architecture"):
    jaxp.check({"w": np.zeros(4, np.float32), "n": np.int64(0)}, tmpl)
  assert isinstance(jaxp.to_host(new)["w"], np.ndarray)
  # HOST_DRIVEN: a list of K host objects.
  lb = listp.empty({"sd": 0}, 2)
  assert isinstance(lb, list) and len(lb) == 2
  lb2 = listp.set(lb, 0, {"sd": 1})
  assert (
      lb2[0] == {"sd": 1}
      and lb[0] == {"sd": 0}
      and listp.get(lb2, 0) == {"sd": 1}
  )


def test_pool_storage_registration_is_tried_before_the_built_ins(monkeypatch):
  """Pool storage registration is tried before the built ins."""
  # Reaches private _REGISTERED of the code under test.
  monkeypatch.setattr(pool_params, "_REGISTERED", list(pool_params._REGISTERED))  # pylint: disable=protected-access
  in_graph_other = _Spec(types.Capability.CALLBACK_BATCHED, "torch")
  # Reaches private _REGISTERED of the code under test.
  if not pool_params._REGISTERED:  # pylint: disable=protected-access
    with pytest.raises(ValueError, match="no pool storage"):
      pool_params.for_policy(in_graph_other)

  class Mine(pool_params.HostList):
    """A HostList subclass registered as its own pool storage."""

    pass

  pool_params.register(lambda spec: spec.arch_name == "stub", Mine)
  assert isinstance(pool_params.for_policy(in_graph_other), Mine)
  assert isinstance(
      pool_params.for_policy(_Spec(types.Capability.NATIVE_JAX, "jax")), Mine
  )


# -- driver hooks --------------------------------------------------------------
def test_loop_names_no_jax_sharding_and_hooks_exist():
  """Loop names no jax sharding and hooks exist."""
  src = inspect.getsource(loop)
  for banned in (
      "jax.sharding",
      "NamedSharding",
      "device_put",
      ".at[",
      "jnp.stack",
  ):
    assert banned not in src, banned
  for name in ("commit_state", "retire_state", "freeze", "mutate"):
    assert callable(getattr(kinds.Driver, name, None)), name


def test_pg_driver_mutate_is_the_leafwise_affine_combination():
  """Pg driver mutate is the leafwise affine combination."""
  # Lazy: optional dependency: league_fixtures needs the games package.
  from open_spiel.python.scale.league.tests import league_fixtures  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  pol = league_fixtures.kuhn_policy(8)
  driver = kinds.PGDriver(pol, league_fixtures.br_learner(pol))
  src, fresh = league_fixtures.init_params(pol, 1), league_fixtures.init_params(
      pol, 2
  )
  got = driver.mutate(src, fresh, 0.75, 0.1)
  sh, pt = jnp.float32(0.75), jnp.float32(0.1)
  want = jax.tree_util.tree_map(
      lambda s, f: (sh * s + pt * f).astype(s.dtype), src, fresh
  )
  assert league_fixtures.leaves_equal(got, want)


# -- ArenaSource snapshots -----------------------------------------------------
class _Boxed:
  """A leaf type no snapshot knows natively.

  Attributes:
    v: the boxed array.
  """

  def __init__(self, v):
    self.v = np.asarray(v)


def _is_boxed(x):
  """Whether `x` is a _Boxed.

  Args:
    x: A value.

  Returns:
    True for one.
  """
  return isinstance(x, _Boxed)


def _box_encode(x):
  """Encodes a _Boxed leaf.

  Args:
    x: The _Boxed.

  Returns:
    (value, no metadata).
  """
  return x.v, None


def _box_decode(v, meta):
  """Decodes a _Boxed leaf.

  Args:
    v: The value.
    meta: The metadata (unused).

  Returns:
    The _Boxed.
  """
  del meta
  return _Boxed(v)


def test_source_snapshot_round_trips_jax_keys_and_registered_leaves(
    monkeypatch,
):
  """Source snapshot round trips jax keys and registered leaves."""
  # Reaches private _CODECS of the code under test.
  monkeypatch.setattr(snapshot, "_CODECS", list(snapshot._CODECS))  # pylint: disable=protected-access
  tree = {
      "k": jax.random.key(3),
      "j": jnp.arange(3, dtype=jnp.int64),
      "n": np.int32(4),
      "x": _Boxed([1.5, 2.5]),
  }
  with pytest.raises(TypeError, match="unsupported state leaf"):
    commit.host_tree(tree)
  snapshot.register_leaf_codec(
      _is_boxed, _box_encode, _box_decode, name="jax_free_control_test.boxed"
  )
  host = commit.host_tree(tree)
  # A round trip of bytes this test just pickled.
  # pylint: disable-next=g-unsafe-pickle-load
  back = commit.device_tree(pickle.loads(pickle.dumps(host)))  # noqa: S301
  np.testing.assert_array_equal(
      np.asarray(jax.random.key_data(back["k"])),
      np.asarray(jax.random.key_data(tree["k"])),
  )
  assert back["j"].dtype == jnp.int64 and back["n"] == 4
  assert isinstance(back["x"], _Boxed) and back["x"].v.tolist() == [1.5, 2.5]


def test_rng_is_the_league_key_schedule():
  """The League's keys through core.rng equal jax.random's."""
  root = jax.random.key(5)
  a = rng.split(root)
  b = jax.random.split(root)
  np.testing.assert_array_equal(
      np.asarray(jax.random.key_data(a)), np.asarray(jax.random.key_data(b))
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

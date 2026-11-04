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

"""The 64-bit contract.

the package runs with jax_enable_x64 on, turned on by its enable_x64() (called
by the entry points and the test conftest, never by an import) and asserted by
its components; every counter is int64 and never wraps at 2**31 (or at 2**32 in
a fold-in).
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import chance
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.core.tests import arena_fixtures
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.policies import uniform

_PKG = subprocess_env.PACKAGE
BIG = 2**31 - 1  # the largest int32


def _run(code, **env_over):
  """Runs `code` in a child with JAX_ENABLE_X64 removed then overridden.

  Args:
    code: The Python code.
    **env_over: Environment variables to set.

  Returns:
    The completed process.
  """
  env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
  env.update(env_over)
  return subprocess_env.run_python(["-c", code], base_env=env, timeout=300)


# -- the package switch --------------------------------------------------------
# Importing the package (and its entry-point modules) changes nothing: other
# JAX code in the same process keeps its own setting.
_IMPORTS = "".join(
    f"import {_PKG}{m}\n"
    for m in (
        "",
        ".core.x64",
        ".core.arena",
        ".train",
        ".league.runner",
        ".serve.server",
        ".launch.run",
    )
)
_UNCHANGED = (
    "import os\n"
    "assert 'JAX_ENABLE_X64' not in os.environ, os.environ['JAX_ENABLE_X64']\n"
    "assert not jax.config.jax_enable_x64\n"
    "assert jnp.arange(2).dtype == jnp.int32\n"
)


def test_importing_the_package_after_jax_leaves_x64_unchanged():
  """Importing the package after jax leaves x64 unchanged."""
  r = _run(
      "import jax, jax.numpy as jnp\nassert not jax.config.jax_enable_x64\n"
      + _IMPORTS
      + _UNCHANGED
  )
  assert r.returncode == 0, r.stderr[-3000:]


def test_importing_the_package_before_jax_leaves_x64_unchanged():
  """Importing the package before jax leaves x64 unchanged."""
  r = _run(
      f"import sys, {_PKG}\nassert 'jax' not in sys.modules\n"
      + _IMPORTS
      + "import jax, jax.numpy as jnp\n"
      + _UNCHANGED
  )
  assert r.returncode == 0, r.stderr[-3000:]


def test_importing_the_package_with_x64_off_in_the_env_does_not_raise():
  """Importing with JAX_ENABLE_X64=0 does not raise."""
  r = _run(
      _IMPORTS + "import jax\nassert not jax.config.jax_enable_x64\n",
      JAX_ENABLE_X64="0",
  )
  assert r.returncode == 0, r.stderr[-3000:]


def test_enable_x64_before_jax_turns_x64_on():
  """enable_x64 before jax loads turns x64 on."""
  r = _run(
      f"import sys, {_PKG}\n"
      f"{_PKG}.enable_x64()\n"
      "assert 'jax' not in sys.modules\n"
      "import jax, jax.numpy as jnp\n"
      "assert jax.config.jax_enable_x64\n"
      "assert jnp.zeros(()).dtype == jnp.float64 and jnp.arange(2).dtype == "
      "jnp.int64\n"
  )
  assert r.returncode == 0, r.stderr[-3000:]


def test_enable_x64_after_jax_turns_x64_on():
  """enable_x64 after jax loads turns x64 on."""
  r = _run(
      "import jax, jax.numpy as jnp\n"
      "assert not jax.config.jax_enable_x64\n"
      f"from {_PKG}.core import x64\n"
      "x64.enable_x64()\n"
      "assert jax.config.jax_enable_x64 and jnp.arange(2).dtype == jnp.int64\n"
  )
  assert r.returncode == 0, r.stderr[-3000:]


def test_enable_x64_with_x64_off_in_the_env_fails_loud():
  """enable_x64 with JAX_ENABLE_X64=0 fails loud."""
  r = _run(f"import {_PKG}\n{_PKG}.enable_x64()\n", JAX_ENABLE_X64="0")
  assert r.returncode != 0 and "JAX_ENABLE_X64" in r.stderr, r.stderr[-3000:]


def test_the_entry_points_turn_x64_on():
  """The entry points turn x64 on.

  The trainer's command line (before its flags parse), the league runner's
  and the trainer's run() and the policy server.
  """
  cli = _run(
      "import runpy, sys, jax\n"
      "sys.argv = ['train', '--helpshort']\n"
      "try:\n"
      f"  runpy.run_module('{_PKG}.train', run_name='__main__', "
      "alter_sys=True)\n"
      "except SystemExit:\n"
      "  pass\n"
      "assert jax.config.jax_enable_x64\n"
  )
  assert cli.returncode == 0, cli.stderr[-3000:]
  server = _run(
      f"import jax\nfrom {_PKG}.serve import server\n"
      "assert not jax.config.jax_enable_x64\n"
      "server.PolicyServer()\n"
      "assert jax.config.jax_enable_x64\n"
  )
  assert server.returncode == 0, server.stderr[-3000:]
  for mod in ("train", "league.runner"):
    # run() turns x64 on before it reads its config (a wrong one raises after).
    r = _run(
        "import jax\n"
        f"from {_PKG}.{mod} import run\n"
        "assert not jax.config.jax_enable_x64\n"
        "try:\n"
        "  run(object())\n"
        "except (TypeError, AssertionError):\n"
        "  pass\n"
        "assert jax.config.jax_enable_x64\n"
    )
    assert r.returncode == 0, (mod, r.stderr[-3000:])


_OFF_PROBE = r"""
import jax
import PKG
from PKG.core import x64
raises_never_on = None
try:
  x64.assert_x64()
except x64.NotEnabledError as e:  # a process that never turned it on
  raises_never_on = str(e)
assert raises_never_on and "enable_x64()" in raises_never_on, raises_never_on
PKG.enable_x64()
x64.assert_x64()
jax.config.update("jax_enable_x64", False)
def raises(fn):
  try:
    fn()
  except x64.NotEnabledError as e:
    assert "jax_enable_x64" in str(e), e
    return
  raise SystemExit(f"{fn} did not raise NotEnabledError with x64 off")
raises(x64.assert_x64)
from PKG.core import arena, decide, types
from PKG.envs import kuhn
from PKG.policies import uniform
import jax.numpy as jnp
env = kuhn.KuhnEnv()
pol = uniform.UniformPolicy("kuhn_poker", 2, 2)
raises(lambda: arena.Arena(env, [arena.Slot(pol)]))
raises(lambda: decide.decide(jnp.zeros((1, 2), jnp.float32),
                             jnp.ones((1, 2), bool), types.ARGMAX,
                             jax.random.key(0)))
print("OFF_PROBE_OK")
""".replace("PKG", _PKG)


def test_assert_x64_and_entry_points_raise_when_x64_is_off():
  """assert_x64 and the entry points raise when x64 is off."""
  r = _run(_OFF_PROBE)
  assert r.returncode == 0 and "OFF_PROBE_OK" in r.stdout, (
      r.stdout[-2000:] + r.stderr[-3000:]
  )


def test_x64_is_on_in_this_process():
  """The conftest turned it on."""
  x64.assert_x64()
  assert jax.config.jax_enable_x64


# -- fold-ins ------------------------------------------------------------------
def test_raw_fold_in_wraps_at_2_32_but_fold_in64_does_not():
  """jax.random.fold_in wraps at 2**32; fold_in64 does not."""
  k = jax.random.key(0)

  def kd(key):
    """The key data of `key`.

    Args:
      key: A typed key.

    Returns:
      The numpy key data.
    """
    return np.asarray(jax.random.key_data(key))

  # The hazard: jax.random.fold_in takes data mod 2**32.
  np.testing.assert_array_equal(
      kd(jax.random.fold_in(k, jnp.int64(2**32 + 5))),
      kd(jax.random.fold_in(k, 5)),
  )
  assert not np.array_equal(
      kd(x64.fold_in64(k, jnp.int64(2**32 + 5))), kd(x64.fold_in64(k, 5))
  )
  assert not np.array_equal(
      kd(x64.fold_in64(k, jnp.int64(2**33))), kd(x64.fold_in64(k, 0))
  )
  # Below 2**32 it is jax.random.fold_in exactly (derivations unchanged).
  for n in (0, 5, BIG, BIG + 1, 2**32 - 1):
    np.testing.assert_array_equal(
        kd(x64.fold_in64(k, jnp.int64(n))), kd(jax.random.fold_in(k, n))
    )
  # Jittable and vmappable, over typed and raw keys.
  ns = jnp.array([5, 2**32 + 5], jnp.int64)
  out = jax.jit(jax.vmap(x64.fold_in64, in_axes=(None, 0)))(k, ns)
  assert not np.array_equal(kd(out[0]), kd(out[1]))
  raw = jax.random.PRNGKey(0)
  r5, rbig = x64.fold_in64(raw, 5), x64.fold_in64(raw, jnp.int64(2**32 + 5))
  assert r5.dtype == jnp.uint32 and not np.array_equal(
      np.asarray(r5), np.asarray(rbig)
  )


def test_fold_in64_rejects_non_integer_data():
  """fold_in64 rejects non-integer data."""
  with pytest.raises(AssertionError):
    x64.fold_in64(jax.random.key(0), jnp.float32(1.0))


def test_policy_keys_past_2_32_plies_are_distinct():
  """Policy keys past 2**32 plies are distinct."""
  k = jax.random.key(3)
  a = plies.ply_keys(k, jnp.int64(7))
  b = plies.ply_keys(k, jnp.int64(2**32 + 7))
  for x, y in zip(a, b):
    assert not np.array_equal(
        np.asarray(jax.random.key_data(x)), np.asarray(jax.random.key_data(y))
    )


def test_chance_count_is_int64_and_does_not_wrap():
  """The chance count is int64 and does not wrap."""
  env = arena_fixtures.Leduc()
  key = jax.random.key(1)
  _, n = chance.resolve(env, env.root(), key, jnp.int64(BIG))
  assert (
      n.dtype == jnp.int64 and int(n) == BIG + 2
  )  # leduc deals two private cards
  # Event index 2**32 + i is not event i (raw fold_in would make them equal).
  resolve = jax.jit(
      lambda k, c: chance.resolve(env, env.root(), k, c)[0].private_cards
  )
  differ = 0
  for i in range(8):
    k = jax.random.fold_in(key, i)
    lo = np.asarray(resolve(k, jnp.int64(0)))
    hi = np.asarray(resolve(k, jnp.int64(2**32)))
    differ += int(not np.array_equal(lo, hi))
  assert (
      differ >= 4
  ), f"deals at event index 2**32 equal those at 0 in {8 - differ}/8 keys"


# -- Arena counters ------------------------------------------------------------
def _arena():
  """A two-hand uniform Kuhn arena.

  Returns:
    (arena, slot params).
  """
  env = kuhn.KuhnEnv()
  pol = uniform.UniformPolicy("kuhn_poker", env.num_actions, env.num_players)
  ar = arena_lib.Arena(
      env, [arena_lib.Slot(pol, 1, types.SAMPLE)], hands_per_match=2
  )
  return ar, [jax.tree_util.tree_map(lambda x: x[None], {})]


def test_arena_counters_are_int64():
  """The Arena's counters are int64."""
  ar, params = _arena()
  table = arena_lib.self_play_table(8, 2)
  dk, pk = arena_lib.independent_keys(jax.random.key(0), 8, 2)
  st = ar.reset(params, table, dk, pk)
  for name in (
      "hand_index",
      "hand_in_match",
      "chance_count",
      "row_ply",
      "hand_ply",
  ):
    assert getattr(st, name).dtype == jnp.int64, (name, getattr(st, name).dtype)
  st, _, _ = ar.rollout(params, table, st, 5)
  for name in (
      "hand_index",
      "hand_in_match",
      "chance_count",
      "row_ply",
      "hand_ply",
  ):
    assert getattr(st, name).dtype == jnp.int64, (name, getattr(st, name).dtype)


def test_arena_counters_past_2_31_do_not_wrap():
  """The Arena's counters past 2**31 do not wrap."""
  ar, params = _arena()
  b, t = 8, 9
  table = arena_lib.self_play_table(b, 2)
  dk, pk = arena_lib.independent_keys(jax.random.key(1), b, 2)
  st = ar.reset(params, table, dk, pk)
  big = jnp.full((b,), BIG, jnp.int64)
  st = st._replace(row_ply=big, hand_index=big, chance_count=big)
  st2, batch, log = ar.rollout(params, table, st, t)
  np.testing.assert_array_equal(
      np.asarray(st2.row_ply), np.full((b,), BIG + t, np.int64)
  )
  hi = np.asarray(st2.hand_index)
  assert (
      hi > BIG
  ).all(), hi  # kuhn hands are <= 4 plies, so every row dealt again
  assert (np.asarray(st2.chance_count) >= 0).all()
  assert not np.asarray(log.bad).any()
  # Rows keep sampling distinct policy keys past the int32 range: actions are
  # not one frozen line repeated.
  assert len({tuple(r) for r in np.asarray(batch.action).tolist()}) > 1


def test_arena_source_frame_counter_is_int64_past_2_31():
  """ArenaSource.frames is int64 past 2**31."""
  ar, params = _arena()
  b, t = 8, 3
  src = arena_lib.ArenaSource(
      ar, params, arena_lib.self_play_table(b, 2), num_steps=t
  )
  assert isinstance(src.frames, np.int64) and src.frames == 0
  src.frames = np.int64(BIG)
  src.next(jax.random.key(2))
  assert isinstance(src.frames, np.int64) and src.frames == BIG + b * t
  src.restart()
  src.next(jax.random.key(3))
  assert src.frames == BIG + 2 * b * t, "frames are cumulative across restarts"


def test_arena_source_decision_counter_counts_valid_rows_int64():
  """ArenaSource.decisions counts the batches' valid rows, as int64."""
  ar, params = _arena()
  b, t = 8, 6
  src = arena_lib.ArenaSource(
      ar, params, arena_lib.self_play_table(b, 2), num_steps=t
  )
  assert isinstance(src.decisions, np.int64) and src.decisions == 0
  src.decisions = np.int64(BIG)
  batch = src.next(jax.random.key(4))
  n = int(np.asarray(batch.valid).sum())
  assert 0 < n < b * t  # terminal plies are never decisions
  assert isinstance(src.decisions, np.int64) and src.decisions == BIG + n


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

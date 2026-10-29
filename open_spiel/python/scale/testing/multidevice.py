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

"""Multi-device test helpers: probes under N forced host devices, tree checks.

A claim such as "the 8-device update equals the 1-device update" is tested
on CPU by forcing N host devices. XLA reads the device count before jax
loads, so the multi-device side runs as a probe module in a subprocess.

  out =
  multidevice.run_probe(subprocess_env.module("parallel.tests.probe_learner"),
                              8, "pg")

The probe calls expect_devices(8), runs its checks and prints PROBE_OK.

Data-parallel cases: the learner probe runs every registered case, so a
package that adds a loss variant also adds its data-parallel equivalence
check with register_dp_case(name, fn), where fn maps the kuhn PGConfig to the
PGConfig under test.
"""

import contextlib
import io
import os
import re
import runpy
import sys
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.launch import subprocess_env

OK = "PROBE_OK"

_DP_CASES: dict[str, Callable[[Any], Any]] = {}


def register_dp_case(name: str, fn: Callable[[Any], Any]) -> None:
  """Adds a data-parallel learner case.

  Args:
    name: The case name.
    fn: fn(kuhn PGConfig) -> the PGConfig to check.

  Raises:
    ValueError: `name` is already registered.
  """
  if name in _DP_CASES:
    raise ValueError(f"data-parallel case {name!r} is already registered")
  _DP_CASES[name] = fn


def dp_cases() -> tuple[tuple[str, Callable[[Any], Any]], ...]:
  """Returns the registered data-parallel cases, in order.

  Returns:
    ((name, fn), ...).
  """
  return tuple(_DP_CASES.items())


def run_probe(
    module: str, num_devices: int, *args: str, timeout: int = 1500
) -> str:
  """Runs `python -m module args` with num_devices forced CPU devices.

  With no interpreter to start a child (subprocess_env.python_available), the
  probe runs in this process when this process already has exactly
  num_devices CPU devices (a runner that sets
  XLA_FLAGS=--xla_force_host_platform_device_count=N for the test binary), and
  the test is skipped otherwise.

  Args:
    module: The probe module.
    num_devices: The number of forced host devices.
    *args: The probe's arguments.
    timeout: The timeout in seconds.

  Returns:
    The probe's stdout.

  Raises:
    AssertionError: the probe did not exit 0 or did not print PROBE_OK.
  """
  if not subprocess_env.python_available() and _has_cpu_devices(num_devices):
    out = _run_in_process(module, args)
    assert OK in out, f"stdout:\n{out[-3000:]}"
    return out
  flags = [
      f
      for f in os.environ.get("XLA_FLAGS", "").split()
      if not f.startswith("--xla_force_host_platform_device_count")
  ]
  env = {
      "XLA_FLAGS": " ".join(
          flags + [f"--xla_force_host_platform_device_count={num_devices}"]
      ),
      "JAX_PLATFORMS": "cpu",
  }
  r = subprocess_env.run_python(["-m", module, *args], env=env, timeout=timeout)
  tail = f"stdout:\n{r.stdout[-3000:]}\nstderr:\n{r.stderr[-6000:]}"
  assert r.returncode == 0, tail
  assert OK in r.stdout, tail
  return r.stdout


def _has_cpu_devices(n: int) -> bool:
  """Whether the backend is CPU with exactly n devices.

  Args:
    n: The device count.

  Returns:
    True when it is.
  """
  return jax.default_backend() == "cpu" and len(jax.devices()) == n


def _run_in_process(module: str, args: Sequence[str]) -> str:
  """`python -m module args` in this process; returns its stdout.

  Args:
    module: The module.
    args: Its arguments.

  Returns:
    The stdout.
  """
  out = io.StringIO()
  saved_argv = sys.argv
  sys.argv = [module, *args]
  try:
    with contextlib.redirect_stdout(out):
      runpy.run_module(module, run_name="__main__", alter_sys=True)
  except SystemExit as e:
    assert e.code in (
        None,
        0,
    ), f"probe exit {e.code}; stdout:\n{out.getvalue()[-3000:]}"
  finally:
    sys.argv = saved_argv
  return out.getvalue()


def expect_devices(n: int) -> None:
  """Asserts this process sees n devices (call it first in a probe).

  Args:
    n: The expected device count.
  """
  assert len(jax.devices()) == n, (
      f"probe wants {n} devices, has {jax.devices()}: run it through "
      "testing.multidevice.run_probe"
  )


def host(x):
  """Returns a numpy view of an array leaf.

  A typed PRNG key is returned as its key data.

  Args:
    x: The leaf.

  Returns:
    The numpy array.
  """
  x = jnp.asarray(x) if not isinstance(x, jax.Array) else x
  if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
    return np.asarray(jax.random.key_data(x))
  return np.asarray(x)


def assert_trees_equal(a, b, what: str = "") -> None:
  """Asserts equal structure and bit-identical leaves, dtype and shape too.

  Args:
    a: A tree.
    b: Another tree.
    what: A label for the failure message.
  """
  la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
  assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(
      b
  ), what
  for i, (x, y) in enumerate(zip(la, lb)):
    x, y = host(x), host(y)
    assert x.dtype == y.dtype and x.shape == y.shape, (
        what,
        i,
        x.dtype,
        x.shape,
        y.dtype,
        y.shape,
    )
    np.testing.assert_array_equal(x, y, err_msg=f"{what} leaf {i}")


def assert_trees_close(a, b, rtol: float, atol: float, what: str = "") -> float:
  """Asserts allclose leaf by leaf.

  Args:
    a: A tree.
    b: The reference tree.
    rtol: The relative tolerance.
    atol: The absolute tolerance.
    what: A label for the failure message.

  Returns:
    The max absolute difference.
  """
  assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(
      b
  ), what
  worst = 0.0
  for i, (x, y) in enumerate(
      zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
  ):
    x, y = host(x), host(y)
    assert x.dtype == y.dtype and x.shape == y.shape, (what, i)
    np.testing.assert_allclose(
        x, y, rtol=rtol, atol=atol, err_msg=f"{what} leaf {i}"
    )
    if x.size and np.issubdtype(x.dtype, np.floating):
      worst = max(
          worst,
          float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))),
      )
  return worst


def rel_l2(a, b) -> float:
  """Returns ||a - b|| / ||b|| over every leaf.

  Integer leaves are included; each leaf is cast to f64 (f64 accumulation).

  Args:
    a: A tree.
    b: The reference tree.

  Returns:
    The relative L2 difference.
  """
  num = den = 0.0
  for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
    x, y = host(x).astype(np.float64), host(y).astype(np.float64)
    num += float(np.sum((x - y) ** 2))
    den += float(np.sum(y**2))
  return (num / max(den, 1e-300)) ** 0.5


COLLECTIVES = (
    "all-gather",
    "all-reduce",
    "all-to-all",
    "collective-permute",
    "reduce-scatter",
    "collective-broadcast",
)


def collectives_in(hlo_text: str) -> dict[str, int]:
  """Counts the COLLECTIVES ops in compiled HLO text.

  Args:
    hlo_text: The HLO text.

  Returns:
    {collective: count}.
  """
  return {
      c: len(re.findall(rf"\b{c}(?:-start)?\(", hlo_text)) for c in COLLECTIVES
  }

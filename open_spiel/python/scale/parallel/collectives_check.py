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

"""A pre-training check that every device in the mesh can run a collective.

A mismatched NCCL installation lets a sharded program compile and run its
collective-free rollout, then fail at the first gradient all-reduce or
all-to-all, often with an error that names no cause. This check runs psum,
all_gather, all_to_all and a sharded argsort across the mesh on known inputs
before training starts and reports the library versions in use, so the
failure appears in seconds.

  collectives_check.check_collectives(jax.devices())

Preflights. The train entry point calls run_preflights(devices) before its first
compile. It runs the built-in "collectives" preflight when asked
(`--collectives_preflight`) and then every preflight registered with
register_preflight(name, fn), in registration order; a deployment registers
the checks its machines need. The built-in preflight skips one device or a
non-GPU platform (there is nothing to talk to); on GPUs it sets NCCL_DEBUG
to WARN unless the run set it, so a later NCCL failure names its cause, logs
diagnostics() and raises Error if a check fails.

  python -m open_spiel.python.scale.parallel.collectives_check

runs diagnostics() and check_collectives() on every visible device of any
platform and exits 1 on a failed check
(XLA_FLAGS=--xla_force_host_platform_device_count=8 gives 8 CPU devices).
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import resource
import subprocess
import sys
from typing import Any, Callable, Optional, Sequence

from absl import logging
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np

NCCL_DEBUG_DEFAULT = "WARN"
CHECKS = ("psum", "all_gather", "all_to_all", "sharded_argsort")
COLLECTIVES = "collectives"  # the built-in preflight's name

_PREFLIGHTS: dict[str, Callable[[Sequence[Any]], None]] = {}


class Error(RuntimeError):
  """A multi-device collective failed before training."""


def _open_fds() -> object:
  """The process's open file descriptor count.

  Returns:
    The count, or why it is unavailable.
  """
  try:
    return len(os.listdir("/proc/self/fd"))
  except OSError as e:
    return f"<unavailable: {e}>"


def _run(argv: Sequence[str], timeout: float = 20.0) -> str:
  """The stdout + stderr of argv, or the reason it could not be read.

  Args:
    argv: The command.
    timeout: Seconds.

  Returns:
    The output, or the reason.
  """
  try:
    p = subprocess.run(
        list(argv), capture_output=True, text=True, timeout=timeout, check=False
    )
  except (OSError, subprocess.SubprocessError) as e:
    return f"<unavailable: {type(e).__name__}: {e}>"
  return (p.stdout + p.stderr).strip() + (
      "" if p.returncode == 0 else f"\n<rc={p.returncode}>"
  )


def _nccl_library() -> dict[str, Any]:
  """The installed NCCL wheels' versions and library files.

  Returns:
    dist -> version for each installed NCCL wheel, plus lib_dir (and libs,
    its files) when nvidia.nccl imports.
  """
  out: dict[str, Any] = {}
  for dist in ("nvidia-nccl-cu12", "nvidia-nccl-cu13"):
    try:
      out[dist] = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
      continue
  try:
    # Lazy: optional dependency (the NCCL wheel exists only on GPU installs).
    import nvidia.nccl  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    lib = os.path.join(list(nvidia.nccl.__path__)[0], "lib")
    out["lib_dir"] = lib
    out["libs"] = sorted(os.listdir(lib)) if os.path.isdir(lib) else []
  except ImportError as e:
    out["lib_dir"] = f"<unavailable: {e}>"
  return out


def _net_plugins() -> list[str]:
  """The NCCL network plugins on LD_LIBRARY_PATH.

  Returns:
    Their paths.
  """
  found = []
  for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    if d and os.path.isdir(d):
      found += [
          os.path.join(d, f)
          for f in sorted(os.listdir(d))
          if f.startswith("libnccl-net")
      ]
  return found


def diagnostics() -> dict[str, Any]:
  """What multi-GPU collectives depend on, read without side effects.

  Returns a JSON-ready dict: the NCCL distribution versions and library dir,
  the NCCL_* / XLA_* / CUDA_* / JAX_* environment, /dev/shm size,
  RLIMIT_NOFILE and RLIMIT_MEMLOCK, open descriptors, network interfaces,
  NCCL net plugins on LD_LIBRARY_PATH, `nvidia-smi topo -m` and the fabric
  manager's state. A read that fails records why; nothing is guessed.

  Returns:
    the JSON-ready dict.
  """
  try:
    st = os.statvfs("/dev/shm")
    shm = {
        "total_bytes": st.f_blocks * st.f_frsize,
        "free_bytes": st.f_bavail * st.f_frsize,
    }
  except OSError as e:
    shm = {"error": f"{type(e).__name__}: {e}"}
  try:
    ifaces = sorted(os.listdir("/sys/class/net"))
  except OSError as e:
    ifaces = [f"<unavailable: {e}>"]
  return {
      "nccl": _nccl_library(),
      "env": {
          k: v
          for k, v in sorted(os.environ.items())
          if k.startswith(("NCCL_", "XLA_", "CUDA_", "JAX_", "LD_LIBRARY_PATH"))
      },
      "dev_shm": shm,
      "rlimit_nofile": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
      "open_fds": _open_fds(),
      "rlimit_memlock": list(resource.getrlimit(resource.RLIMIT_MEMLOCK)),
      "net_interfaces": ifaces,
      "nccl_net_plugins": _net_plugins(),
      "nvidia_smi_topo": _run(["nvidia-smi", "topo", "-m"]),
      "fabricmanager": _run(["systemctl", "is-active", "nvidia-fabricmanager"]),
  }


def check_collectives(devices: Sequence[Any]) -> dict[str, str]:
  """Runs CHECKS over `devices` on known inputs and compares each to numpy.

  Args:
    devices: at least 2 devices of one platform.

  Returns:
    {check: "ok"} for every name in CHECKS.

  Raises:
    ValueError: fewer than 2 devices.
    AssertionError: a collective returned a wrong result.
    Exception: the runtime's own error for a collective that failed.
  """
  devices = list(devices)
  n = len(devices)
  if n < 2:
    raise ValueError(f"check_collectives needs at least 2 devices, got {n}")
  mesh = Mesh(np.asarray(devices), ("d",))
  rows = NamedSharding(mesh, PartitionSpec("d"))
  smap = jax.shard_map
  out = {}

  def put(x):
    """Places x with its rows split over the devices.

    Args:
      x: an array.

    Returns:
      the placed array.
    """
    return jax.device_put(jnp.asarray(x), rows)

  a = np.arange(n * n, dtype=np.float32).reshape(n, n)
  f = jax.jit(
      smap(
          lambda x: jax.lax.psum(x, "d"),
          mesh=mesh,
          in_specs=PartitionSpec("d"),
          out_specs=PartitionSpec("d"),
      )
  )
  got = np.asarray(f(put(a)))
  np.testing.assert_array_equal(got, np.tile(a.sum(0, keepdims=True), (n, 1)))
  out["psum"] = "ok"

  f = jax.jit(
      smap(
          lambda x: jax.lax.all_gather(x, "d", tiled=True),
          mesh=mesh,
          in_specs=PartitionSpec("d"),
          out_specs=PartitionSpec("d"),
      )
  )
  got = np.asarray(f(put(a)))
  np.testing.assert_array_equal(got, np.tile(a, (n, 1)))
  out["all_gather"] = "ok"

  # Device i holds row i; after the all-to-all device j holds column j.
  f = jax.jit(
      smap(
          lambda x: jax.lax.all_to_all(
              x, "d", split_axis=1, concat_axis=0, tiled=True
          ),
          mesh=mesh,
          in_specs=PartitionSpec("d"),
          out_specs=PartitionSpec("d"),
      )
  )
  got = np.asarray(f(put(a)))
  np.testing.assert_array_equal(got, a.T.reshape(n * n, 1))
  out["all_to_all"] = "ok"

  # A stable argsort of a row-sharded array, the pattern a learner update uses
  # to compact plies; XLA lowers it to an all-to-all.
  keys = np.random.default_rng(0).permutation(n * 64).astype(np.int32)
  f = jax.jit(lambda k: jnp.argsort(k, stable=True), out_shardings=rows)
  got = np.asarray(f(put(keys)))
  np.testing.assert_array_equal(got, np.argsort(keys, kind="stable"))
  out["sharded_argsort"] = "ok"
  assert tuple(out) == CHECKS, out
  return out


def collectives_preflight(devices: Sequence[Any]) -> dict[str, Any]:
  """The built-in preflight (module doc).

  Raises:
    Error: a check failed; the message carries diagnostics().

  Args:
    devices: the devices to check.

  Returns:
    the record it logged.
  """
  devices = list(devices)
  platform = devices[0].platform
  if platform != "gpu" or len(devices) < 2:
    rec = {
        "skipped": (
            f"{len(devices)} {platform} device(s): no collectives to check"
        )
    }
    logging.info("collectives preflight: %s", rec["skipped"])
    return rec
  os.environ.setdefault("NCCL_DEBUG", NCCL_DEBUG_DEFAULT)
  rec = {
      "devices": len(devices),
      "platform": platform,
      "diagnostics": diagnostics(),
  }
  logging.info(
      "collectives preflight: %s", json.dumps(rec, sort_keys=True, default=str)
  )
  try:
    rec["checks"] = check_collectives(devices)
  # Any failure is re-raised as Error with the diagnostics.
  except Exception as e:  # pylint: disable=broad-except
    raise Error(
        f"multi-device collectives failed on {len(devices)} {platform} devices "
        "before "
        f"training ({type(e).__name__}: {e}); diagnostics: "
        f"{json.dumps(rec['diagnostics'], sort_keys=True, default=str)}"
    ) from e
  logging.info(
      "collectives preflight: %d %s devices: %s",
      len(devices),
      platform,
      json.dumps(rec["checks"]),
  )
  return rec


def register_preflight(name: str, fn: Callable[[Sequence[Any]], None]) -> None:
  """Adds fn(devices) to the preflights run_preflights runs.

  fn raises to stop the run. Registering a name twice raises ValueError.

  Args:
    name: the preflight's name.
    fn: devices -> None; raises to stop the run.

  Raises:
    TypeError: fn is not callable.
    ValueError: name is already registered.
  """
  if not callable(fn):
    raise TypeError(f"preflight {name!r}: {fn!r} is not callable")
  if name == COLLECTIVES or name in _PREFLIGHTS:
    raise ValueError(f"preflight {name!r} is already registered")
  _PREFLIGHTS[name] = fn


def preflights() -> tuple[str, ...]:
  """The registered preflights' names, in the order they run.

  Returns:
    the names.
  """
  return tuple(_PREFLIGHTS)


def run_preflights(
    devices: Optional[Sequence[Any]] = None, *, collectives: bool = False
) -> None:
  """Runs the preflights over `devices` (default every device).

  The built-in collectives preflight when `collectives`, then every registered
  one.

  Args:
    devices: the devices, or None for every device.
    collectives: run the built-in collectives preflight first.
  """
  if devices is None:
    devices = jax.devices()
  devices = list(devices)
  if collectives:
    collectives_preflight(devices)
  for name, fn in _PREFLIGHTS.items():
    logging.info(
        "preflight %s: %d %s devices", name, len(devices), devices[0].platform
    )
    fn(devices)


def main() -> int:
  """Runs diagnostics() and check_collectives() on every visible device.

  Returns:
    the exit code: 0 when the check passes (or does not apply), else 1.
  """
  devices = jax.devices()
  print(
      f"collectives: {len(devices)} {devices[0].platform} devices "
      f"({sorted({d.device_kind for d in devices})})",
      flush=True,
  )
  print(
      "diagnostics: " + json.dumps(diagnostics(), sort_keys=True, default=str),
      flush=True,
  )
  if len(devices) < 2:
    print("collectives: one device, nothing to check", flush=True)
    return 0
  try:
    res = check_collectives(devices)
  # Any failure is reported with the diagnostics and exits 1.
  except Exception as e:  # pylint: disable=broad-except
    print(
        f"collectives: FAILED: {type(e).__name__}: {e} (open fds "
        f"{_open_fds()}, "
        f"RLIMIT_NOFILE {resource.getrlimit(resource.RLIMIT_NOFILE)})",
        flush=True,
    )
    return 1
  for k, v in res.items():
    print(f"collectives: {k} {v}", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())

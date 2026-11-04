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

"""collectives_check.

collectives on known inputs, the built-in preflight and the preflight registry.
"""

import os
import sys

import jax
import pytest

from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.parallel import collectives_check


def test_main_runs_every_collective_on_8_host_devices():
  """The module's main on 8 forced CPU devices.

  every check passes against numpy (the same shard_map and sharded-argsort
  programs a GPU run takes).
  """
  env = {
      "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
      "JAX_PLATFORMS": "cpu",
  }
  p = subprocess_env.run_python(["-m", collectives_check.__name__], env=env)
  assert p.returncode == 0, p.stdout + p.stderr
  assert "collectives: 8 cpu devices" in p.stdout, p.stdout
  for name in collectives_check.CHECKS:
    assert f"collectives: {name} ok" in p.stdout, p.stdout
  assert "diagnostics: " in p.stdout


def test_diagnostics_record_what_collectives_depend_on():
  """Diagnostics records what collectives depend on."""
  d = collectives_check.diagnostics()
  assert {
      "nccl",
      "env",
      "dev_shm",
      "rlimit_nofile",
      "rlimit_memlock",
      "net_interfaces",
      "nccl_net_plugins",
      "nvidia_smi_topo",
      "fabricmanager",
  } <= set(d)
  assert len(d["rlimit_nofile"]) == 2
  # A read that cannot run says so.
  run = collectives_check._run  # pylint: disable=protected-access
  assert run(["/nonexistent/binary"]).startswith("<unavailable")


def test_check_needs_two_devices():
  """check_collectives needs at least two devices."""
  with pytest.raises(ValueError, match="at least 2"):
    collectives_check.check_collectives(jax.devices()[:1])


def test_one_device_or_cpu_skips_the_builtin_preflight():
  """One device, or CPU devices, skip the built-in preflight."""
  assert "skipped" in collectives_check.collectives_preflight(jax.devices()[:1])
  assert "skipped" in collectives_check.collectives_preflight(
      jax.devices()[:1] * 2
  )


class _Gpu:
  platform = "gpu"
  device_kind = "fake"


def test_builtin_preflight_on_gpus_sets_nccl_debug_and_fails_loud(monkeypatch):
  """On GPUs the built-in preflight sets NCCL_DEBUG and fails loud."""
  monkeypatch.delenv("NCCL_DEBUG", raising=False)
  monkeypatch.setattr(
      collectives_check,
      "check_collectives",
      lambda d: {k: "ok" for k in collectives_check.CHECKS},
  )
  rec = collectives_check.collectives_preflight([_Gpu(), _Gpu()])
  assert os.environ["NCCL_DEBUG"] == collectives_check.NCCL_DEBUG_DEFAULT
  assert (
      rec["checks"] == {k: "ok" for k in collectives_check.CHECKS}
      and rec["devices"] == 2
  )
  monkeypatch.setenv("NCCL_DEBUG", "INFO")  # an explicit setting is kept
  collectives_check.collectives_preflight([_Gpu(), _Gpu()])
  assert os.environ["NCCL_DEBUG"] == "INFO"

  def boom(devices):
    raise RuntimeError("ncclAlltoAll failed: unhandled system error")

  monkeypatch.setattr(collectives_check, "check_collectives", boom)
  with pytest.raises(
      collectives_check.Error, match="unhandled system error"
  ) as ei:
    collectives_check.collectives_preflight([_Gpu(), _Gpu()])
  assert "dev_shm" in str(ei.value) and isinstance(
      ei.value.__cause__, RuntimeError
  )


def test_run_preflights_runs_the_builtin_only_when_asked_then_the_registered(
    monkeypatch,
):
  """run_preflights runs the built-in only when asked, then the rest."""
  monkeypatch.setattr(collectives_check, "_PREFLIGHTS", {})
  calls = []
  monkeypatch.setattr(
      collectives_check,
      "collectives_preflight",
      lambda d: calls.append(("collectives", len(d))),
  )
  collectives_check.register_preflight(
      "a", lambda d: calls.append(("a", len(d)))
  )
  collectives_check.register_preflight(
      "b", lambda d: calls.append(("b", len(d)))
  )
  assert collectives_check.preflights() == ("a", "b")
  collectives_check.run_preflights(jax.devices()[:1])
  assert calls == [("a", 1), ("b", 1)]
  calls.clear()
  collectives_check.run_preflights(jax.devices()[:1], collectives=True)
  assert calls == [("collectives", 1), ("a", 1), ("b", 1)]
  with pytest.raises(ValueError, match="already registered"):
    collectives_check.register_preflight("a", lambda d: None)
  with pytest.raises(ValueError, match="already registered"):
    collectives_check.register_preflight(
        collectives_check.COLLECTIVES, lambda d: None
    )


def test_a_failing_preflight_stops_the_run(monkeypatch):
  """A failing registered preflight stops the run."""
  monkeypatch.setattr(collectives_check, "_PREFLIGHTS", {})

  def refuse(devices):
    raise collectives_check.Error("refused")

  collectives_check.register_preflight("refuse", refuse)
  with pytest.raises(collectives_check.Error, match="refused"):
    collectives_check.run_preflights(jax.devices()[:1])


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

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

"""The hardware a run actually got, recorded by the run itself.

A launch may take the first of several machine shapes with capacity, so only
the running process knows which one it got. Every trainer process
records it, logs it and appends it to the run's hardware.json, one row per
process start (a resumed run on other hardware adds a row).

  rec = hardware.record(checkpoint_dir, num_devices=n)

The record holds the JAX devices (platform, device kinds, counts), the host
name and time, whatever the registered machine-info providers return
(register_machine_info: for example a cloud metadata read,
launch.gcp.machine_info), and the accelerator name the registered namer
gives (register_accelerator_namer; None without one). It also emits one
series record, hw/devices/n at step 0 (metrics.emit_series).

check_platform(rec): a launcher that knows what the machine is sets
$OPEN_SPIEL_SCALE_EXPECT_PLATFORM (EXPECT_PLATFORM_ENV); a run that sees another
JAX platform (a CPU jax on a GPU machine) raises before it trains.
"""

from __future__ import annotations

import json
import os
import platform
import time
from typing import Any, Callable, Mapping, Optional

from absl import logging

from open_spiel.python.scale import plugins

FILE_NAME = "hardware.json"
EXPECT_PLATFORM_ENV = plugins.ENV_PREFIX + "EXPECT_PLATFORM"
DEVICES_PARAM = "hw/devices/n"
#: The keys describe() sets itself; a machine-info provider may not return them.
DEVICE_KEYS = (
    "utc",
    "host",
    "platform",
    "device_kinds",
    "devices_visible",
    "num_devices",
    "accelerator",
)

MachineInfo = Callable[[], Mapping[str, Any]]
_PROVIDERS: list[MachineInfo] = []
_NAMER: list[Callable[[Mapping[str, Any]], Optional[str]]] = []


def register_machine_info(provider: MachineInfo) -> None:
  """Adds provider() -> {key: JSON value} to every hardware record.

  Providers run in registration order; two providers may not return the same
  key.

  Args:
    provider: provider() -> {key: JSON value}.
  """
  assert callable(provider), provider
  _PROVIDERS.append(provider)


def register_accelerator_namer(
    fn: Callable[[Mapping[str, Any]], Optional[str]],
) -> None:
  """fn(record) -> the accelerator's name, or None.

  One namer; a second registration replaces the first.

  Args:
    fn: fn(record) -> the accelerator's name, or None.
  """
  assert callable(fn), fn
  _NAMER[:] = [fn]


def accelerator_name(rec: Mapping[str, Any]) -> Optional[str]:
  """The registered namer's name for a record, or None.

  The registered namer's name for a record (devices and machine info), or
  None without a namer.

  Args:
    rec: a hardware record (devices and machine info).

  Returns:
    the name, or None.
  """
  if not _NAMER:
    return None
  return _NAMER[0](rec)


def machine_info() -> dict[str, Any]:
  """Every registered provider's info, merged.

  Returns:
    the merged info.

  Raises:
    ValueError: two providers, or a provider and describe(), share a key.
  """
  out: dict[str, Any] = {}
  for provider in _PROVIDERS:
    got = dict(provider())
    clash = sorted((set(got) & set(out)) | (set(got) & set(DEVICE_KEYS)))
    if clash:
      raise ValueError(
          f"machine-info provider {provider!r} returns keys already set: "
          f"{clash}"
      )
    out.update(got)
  return out


def describe(
    num_devices: Optional[int] = None,
    devices=None,
    info: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
  """The hardware record of this process (module doc): what it runs on.

  Args:
    num_devices: the devices the run shards over (default every device).
    devices: the devices to describe (default jax.devices()).
    info: machine info to use instead of the registered providers'.

  Returns:
    the record.

  Raises:
    ValueError: num_devices is not between 1 and the visible devices.
  """
  # Lazy: an optional dependency (gcp imports this module on the launcher
  # host, which needs no JAX).
  import jax  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  devs = list(jax.devices() if devices is None else devices)
  n = len(devs) if num_devices is None else int(num_devices)
  if not 1 <= n <= len(devs):
    raise ValueError(f"num_devices {n}: {len(devs)} devices visible")
  kinds = sorted({d.device_kind for d in devs})
  rec = {
      "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "host": platform.node(),
      "platform": devs[0].platform,
      "device_kinds": kinds,
      "devices_visible": len(devs),
      "num_devices": n,
  }
  rec.update(machine_info() if info is None else dict(info))
  rec["accelerator"] = accelerator_name(rec)
  return rec


def check_platform(
    rec: Mapping[str, Any], environ: Optional[Mapping[str, str]] = None
) -> None:
  """Checks the record's platform against EXPECT_PLATFORM_ENV.

  Raises RuntimeError when EXPECT_PLATFORM_ENV names a platform other
  than the record's; unset, there is nothing to check.

  Args:
    rec: the hardware record.
    environ: the environment (default os.environ).

  Raises:
    RuntimeError: EXPECT_PLATFORM_ENV names a platform other than the record's.
  """
  want = (
      (os.environ if environ is None else environ)
      .get(EXPECT_PLATFORM_ENV, "")
      .strip()
  )
  if want and rec["platform"] != want:
    raise RuntimeError(
        f"this machine should give JAX the {want!r} platform but JAX sees "
        f"{rec['platform']!r} ({rec['device_kinds']}): install the jax build "
        "for "
        "it (e.g. jax[cuda12] on a GPU machine) in the backend's setup lines"
    )


def write(run_dir: str, rec: Mapping[str, Any]) -> str:
  """Appends rec to <run_dir>/hardware.json.

  Appends rec to <run_dir>/hardware.json (rewritten atomically, so a
  preemption mid-write leaves the previous file); returns its path.

  Args:
    run_dir: the run dir.
    rec: the record.

  Returns:
    the file's path.
  """
  # Lazy: optional dependencies (gcp imports this module on the launcher
  # host, which needs neither etils nor the checkpoint package).
  from etils import epath  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  from open_spiel.python.scale.ckpt import artifact  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  d = epath.Path(run_dir)
  if not str(run_dir).startswith("gs://"):
    d.mkdir(parents=True, exist_ok=True)
  p = d / FILE_NAME
  doc = json.loads(p.read_text()) if p.exists() else {"attempts": []}
  assert isinstance(doc.get("attempts"), list), doc
  doc["attempts"].append(rec)
  artifact.atomic_write(
      p, (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
  )
  return str(p)


def record(
    run_dir: Optional[str],
    *,
    num_devices: Optional[int] = None,
    devices=None,
    info: Optional[Mapping[str, Any]] = None,
    emit: bool = True,
) -> dict[str, Any]:
  """Describes, logs, emits and records this machine's hardware.

  describe(), logged, emitted as one series record (when emit) and
  appended to <run_dir>/hardware.json (when run_dir is given).

  Args:
    run_dir: the run dir, or None to write nothing.
    num_devices: the devices the run shards over (default every device).
    devices: the devices to describe (default jax.devices()).
    info: machine info to use instead of the registered providers'.
    emit: whether to emit the series record.

  Returns:
    the record.
  """
  # Lazy: an optional dependency (gcp imports this module on the launcher
  # host, which needs neither numpy nor etils, both of which metrics imports).
  from open_spiel.python.scale import metrics  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  rec = describe(num_devices, devices, info)
  logging.info("hardware: %s", json.dumps(rec, sort_keys=True))
  if emit:
    metrics.emit_series(0, {DEVICES_PARAM: rec["num_devices"]})
  if run_dir:
    rec = dict(rec, file=write(run_dir, rec))
  return rec

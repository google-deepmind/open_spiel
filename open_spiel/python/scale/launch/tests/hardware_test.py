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

"""Tests of the hardware record.

The hardware record (launch.hardware): JAX devices, registered
machine-info providers, the registered accelerator namer, hardware.json and
the series record; and launch.gcp's metadata-server provider.
"""

import io
import json
import sys
import urllib.error

import pytest

from open_spiel.python.scale import metrics
from open_spiel.python.scale.launch import gcp
from open_spiel.python.scale.launch import hardware as hardware_lib

_TABLE = {
    "a3-highgpu-8g": "h100x8",
    "a3-highgpu-1g": "h100",
    "e2-standard-8": "cpu",
}


def _namer(rec):
  """A launcher's machine table, for the tests: two shapes, spot told apart.

  Args:
    rec: The attempt record.

  Returns:
    The machine name, or None.
  """
  machine = rec.get("machine", {})
  base = _TABLE.get(machine.get("type"))
  if base is None or base == "cpu":
    return base
  return base if machine.get("spot") else base + "_std"


@pytest.fixture(name="registries", autouse=True)
def _registries_fixture(monkeypatch):
  """Each test starts with the test namer and no providers.

  Args:
    monkeypatch: pytest's monkeypatch.
  """
  monkeypatch.setattr(hardware_lib, "_NAMER", [_namer])
  monkeypatch.setattr(hardware_lib, "_PROVIDERS", [])


class _Dev:
  """A fake JAX device.

  Attributes:
    device_kind: the device kind.
    platform: the JAX platform.
  """

  def __init__(self, kind="NVIDIA H100 80GB HBM3", plat="gpu"):
    self.device_kind, self.platform = kind, plat


INFO = {"machine": {"type": "a3-highgpu-8g", "spot": True}}


def test_obtained_known_answer():
  """`describe` has its known answer."""
  rec = hardware_lib.describe(devices=[_Dev()] * 8, info=INFO)
  assert rec["platform"] == "gpu" and rec["device_kinds"] == [
      "NVIDIA H100 80GB HBM3"
  ]
  assert rec["devices_visible"] == rec["num_devices"] == 8
  assert rec["machine"] == INFO["machine"] and rec["accelerator"] == "h100x8"
  assert (
      hardware_lib.describe(1, devices=[_Dev()] * 8, info=INFO)["num_devices"]
      == 1
  )
  with pytest.raises(ValueError, match="devices visible"):
    hardware_lib.describe(9, devices=[_Dev()] * 8, info=INFO)


def test_machine_info_comes_from_the_registered_providers():
  """Machine info comes from the registered providers."""
  hardware_lib.register_machine_info(lambda: INFO)
  hardware_lib.register_machine_info(lambda: {"rack": "r1"})
  rec = hardware_lib.describe(devices=[_Dev()] * 2)
  assert rec["machine"] == INFO["machine"] and rec["rack"] == "r1"
  assert rec["accelerator"] == "h100x8"
  hardware_lib.register_machine_info(lambda: {"rack": "r2"})
  with pytest.raises(ValueError, match="rack"):
    hardware_lib.describe(devices=[_Dev()])


def test_a_provider_may_not_overwrite_device_keys():
  """A provider may not overwrite device keys."""
  hardware_lib.register_machine_info(lambda: {"platform": "tpu"})
  with pytest.raises(ValueError, match="platform"):
    hardware_lib.describe(devices=[_Dev()])


def test_accelerator_names_come_from_the_registered_namer(monkeypatch):
  """Accelerator names come from the registered namer."""

  def rec(t, spot):
    """A record of a machine type, spot or not.

    Args:
      t: the machine type.
      spot: whether it is spot.

    Returns:
      the record.
    """
    return {"machine": {"type": t, "spot": spot}}

  assert [
      hardware_lib.accelerator_name(rec(m, p))
      for m, p in (
          ("a3-highgpu-8g", True),
          ("a3-highgpu-8g", False),
          ("a3-highgpu-1g", True),
          ("e2-standard-8", False),
          ("n2-x", None),
      )
  ] == ["h100x8", "h100x8_std", "h100", "cpu", None]
  monkeypatch.setattr(hardware_lib, "_NAMER", [])
  # no namer: never guessed
  assert hardware_lib.accelerator_name(rec("a3-highgpu-8g", True)) is None
  hardware_lib.register_accelerator_namer(_namer)
  assert hardware_lib.accelerator_name(rec("a3-highgpu-1g", True)) == "h100"


def test_gce_reads_the_metadata_server_and_reports_its_absence():
  """`gce` reads the metadata server and reports its absence."""
  seen = []

  def opener(req, timeout):
    """A metadata server that records each request.

    Args:
      req: the request.
      timeout: its timeout.

    Returns:
      the response.
    """
    seen.append((req.full_url, req.get_header("Metadata-flavor"), timeout))
    val = {
        "machine-type": "projects/1/machineTypes/a2-highgpu-1g",
        "zone": "projects/1/zones/zone-b",
        "scheduling/preemptible": "FALSE",
    }
    return io.BytesIO(val[req.full_url[len(gcp.METADATA) :]].encode())

  assert gcp.gce(opener=opener) == {
      "machine_type": "a2-highgpu-1g",
      "zone": "zone-b",
      "preemptible": "FALSE",
  }
  assert [u for u, _, _ in seen] == [
      gcp.METADATA + p for _, p in gcp.METADATA_KEYS
  ]
  assert all(h == "Google" for _, h, _ in seen)

  def down(req, timeout):
    raise urllib.error.URLError("no route")

  g = gcp.gce(opener=down)
  assert list(g) == ["error"] and "machine-type" in g["error"]
  assert (
      hardware_lib.describe(devices=[_Dev("cpu", "cpu")], info={"gce": g})[
          "accelerator"
      ]
      is None
  )


def test_record_appends_one_row_per_start_and_emits_the_series_record(
    tmp_path, monkeypatch
):
  """`record` appends one row per start and emits the series record."""
  got = []
  monkeypatch.setattr(metrics, "_SERIES_EMITTERS", [got.append])
  a = hardware_lib.record(str(tmp_path), devices=[_Dev()] * 8, info=INFO)
  assert got == [{"hw/devices/n": 8, "step": 0}]
  hardware_lib.record(
      str(tmp_path),
      num_devices=1,
      devices=[_Dev()],
      info={"machine": {"type": "a3-highgpu-1g", "spot": True}},
      emit=False,
  )
  assert len(got) == 1
  doc = json.loads((tmp_path / hardware_lib.FILE_NAME).read_text())
  assert [r["accelerator"] for r in doc["attempts"]] == ["h100x8", "h100"]
  assert a["file"] == str(tmp_path / hardware_lib.FILE_NAME)


def test_check_platform_refuses_a_machine_jax_does_not_use():
  """A GPU machine whose setup installed a CPU jax fails before training."""
  rec = {"platform": "cpu", "device_kinds": ["cpu"]}
  hardware_lib.check_platform(
      rec, environ={}
  )  # nothing promised: nothing to check
  hardware_lib.check_platform(
      rec, environ={hardware_lib.EXPECT_PLATFORM_ENV: "cpu"}
  )
  with pytest.raises(RuntimeError, match="'gpu' platform but JAX sees 'cpu'"):
    hardware_lib.check_platform(
        rec, environ={hardware_lib.EXPECT_PLATFORM_ENV: "gpu"}
    )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

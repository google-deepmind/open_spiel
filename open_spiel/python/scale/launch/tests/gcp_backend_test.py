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

"""Tests of the reference GCP backend without GCP.

The reference GCP backend without GCP: the one config file (every field
required, no defaults), the ordered fallback over machines and zones, the
create command, the startup script, poll / teardown over a fake gcloud, and
the dry run of launch.run.
"""

import dataclasses
import subprocess
import sys

import pytest
import yaml

from open_spiel.python.scale.launch import backend as backend_lib
from open_spiel.python.scale.launch import config as config_lib
from open_spiel.python.scale.launch import gcp
from open_spiel.python.scale.launch import hardware
from open_spiel.python.scale.launch import run as run_cli
from open_spiel.python.scale.launch import storage as storage_lib
from open_spiel.python.scale.launch import supervisor as supervisor_lib

CFG = {
    "backend": "gcp",
    "project": "my-project-123",
    "storage": "gs://my-bucket/runs",
    "image": {
        "family": "ubuntu-2404-lts-amd64",
        "name": None,
        "project": "ubuntu-os-cloud",
    },
    "network": {
        "network": "default",
        "subnetwork": "default",
        "external_ip": False,
    },
    "service_account": {
        "email": "vm@my-project-123.iam.gserviceaccount.com",
        "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
    },
    "labels": {"owner": "me", "ttl": "2h"},
    "code": {"root": ".", "paths": ["pkg"]},
    "setup": ["echo setup-one", "echo setup-two"],
    "python": "/opt/venv/bin/python",
    "create_lock": None,
    "machines": [
        {
            "name": "small-spot",
            "machine_type": "e2-standard-4",
            "accelerator": None,
            "platform": "cpu",
            "zones": ["zone-a", "zone-b"],
            "provisioning": "SPOT",
            "boot_disk_gb": 50,
            "max_run_duration": None,
        },
        {
            "name": "gpu-flex",
            "machine_type": "a3-highgpu-1g",
            "accelerator": None,
            "platform": "gpu",
            "zones": ["zone-c"],
            "provisioning": "FLEX_START",
            "boot_disk_gb": 200,
            "max_run_duration": "24h",
        },
        {
            "name": "t4",
            "machine_type": "n1-standard-8",
            "accelerator": {"type": "nvidia-tesla-t4", "count": 1},
            "platform": "gpu",
            "zones": ["zone-d"],
            "provisioning": "STANDARD",
            "boot_disk_gb": 100,
            "max_run_duration": None,
        },
    ],
}
RUN = backend_lib.RunSpec(
    "kuhn-a", train_args=("--config=mmd_kuhn_poker_jax", "--seed=0")
)
NO_CAP = (
    "ERROR: (gcloud.compute.instances.create) ZONE_RESOURCE_POOL_EXHAUSTED: no"
)
# A stockout as gcloud really prints it: several lines, the reason in the
# middle.
STOCKOUT = (
    "ERROR: (gcloud.compute.instances.create) Could not fetch resource:\n"
    " - code: ZONE_RESOURCE_POOL_EXHAUSTED\n"
    "   message: The zone 'projects/my-project-123/zones/zone-a' does not "
    "have enough resources available to fulfill the request.\n"
    "   errorDetails: domain: compute.googleapis.com metadatas: {} "
    "reason: resource_availability"
)
NO_QUOTA = (
    "ERROR: (gcloud.compute.instances.create) Could not fetch resource:\n"
    " - Quota 'NVIDIA_T4_GPUS' exceeded.  Limit: 0.0 in region zone."
)


def _cfg(tmp_path, **over):
  """A GcpConfig with `over` applied, its code in tmp_path.

  Args:
    tmp_path: The folder.
    **over: Config overrides.

  Returns:
    The GcpConfig.
  """
  (tmp_path / "pkg").mkdir(exist_ok=True)
  (tmp_path / "pkg" / "m.py").write_text("X = 1\n")
  body = {k: v for k, v in {**CFG, **over}.items() if k != "backend"}
  return gcp.GcpConfig.parse(body, str(tmp_path))


class FakeGcloud:
  """A fake gcloud keyed by argv.

  gcloud by argv: create outcomes in order (None succeeds, "full" is NO_CAP,
  any other string is the error itself), storage objects in a dict.

  Attributes:
    creates: create outcomes still to come, in order.
    objects: the storage objects, uri -> text.
    describe: the VM status describe returns.
    ops: the operations list JSON.
    calls: every argv called, in order.
  """

  def __init__(self, creates=(), objects=None, describe="RUNNING", ops="[]"):
    self.creates, self.objects = list(creates), dict(objects or {})
    self.describe, self.ops, self.calls = describe, ops, []

  # `input` is the keyword the callers pass.
  def __call__(self, argv, input=None):  # pylint: disable=redefined-builtin
    self.calls.append(list(argv))

    def ok(out=""):
      """A successful gcloud result.

      Args:
        out: The stdout.

      Returns:
        The completed process.
      """
      return subprocess.CompletedProcess(argv, 0, out, "")

    def err(e):
      """A failed gcloud result.

      Args:
        e: The stderr.

      Returns:
        The completed process.
      """
      return subprocess.CompletedProcess(argv, 1, "", e)

    if argv[1:4] == ["compute", "instances", "create"]:
      outcome = self.creates.pop(0) if self.creates else None
      return (
          ok("[]")
          if outcome is None
          else err(NO_CAP if outcome == "full" else outcome)
      )
    if argv[1:3] == ["storage", "cat"]:
      return (
          ok(self.objects[argv[3]])
          if argv[3] in self.objects
          else err("ERROR: One or more URLs matched no objects.")
      )
    if argv[1:3] == ["storage", "cp"]:
      self.objects[argv[4]] = input if argv[3] == "-" else "<file>"
      return ok()
    if argv[1:4] == ["compute", "instances", "describe"]:
      zone = next(a for a in argv if a.startswith("--zone=")).split("=", 1)[1]
      project = next(a for a in argv if a.startswith("--project=")).split(
          "=", 1
      )[1]
      if self.describe:
        return ok(self.describe)
      return err(
          "ERROR: (gcloud.compute.instances.describe) Could not fetch "
          "resource:\n - The "
          f"resource 'projects/{project}/zones/{zone}/instances/{argv[4]}' "
          "was not found"
      )
    if argv[1:4] == ["compute", "instances", "delete"]:
      return ok()
    if argv[1:4] == ["compute", "operations", "list"]:
      return ok(self.ops)
    raise AssertionError(f"unexpected gcloud {argv}")


def test_every_field_is_required_and_nothing_defaults(tmp_path):
  """Every config field is required; nothing defaults."""
  _cfg(tmp_path)  # the complete file parses
  assert all(
      f.default is dataclasses.MISSING
      and f.default_factory is dataclasses.MISSING
      for f in dataclasses.fields(gcp.GcpConfig)
  )
  for key in [k for k in CFG if k != "backend"]:
    with pytest.raises(
        config_lib.BackendFileError, match=key if key != "code" else "code"
    ):
      gcp.GcpConfig.parse(
          {k: v for k, v in CFG.items() if k not in ("backend", key)},
          str(tmp_path),
      )
  for sub, key in (
      ("image", "project"),
      ("network", "external_ip"),
      ("service_account", "email"),
  ):
    bad = dict(CFG[sub])
    del bad[key]
    with pytest.raises(config_lib.BackendFileError, match=key):
      _cfg(tmp_path, **{sub: bad})
  m = dict(CFG["machines"][0])
  del m["zones"]
  with pytest.raises(config_lib.BackendFileError, match="zones"):
    _cfg(tmp_path, machines=[m])
  with pytest.raises(config_lib.BackendFileError, match="unknown"):
    _cfg(tmp_path, region="zone")
  with pytest.raises(config_lib.BackendFileError, match="max_run_duration"):
    _cfg(tmp_path, machines=[dict(CFG["machines"][1], max_run_duration=None)])
  with pytest.raises(config_lib.BackendFileError, match="provisioning"):
    _cfg(
        tmp_path,
        machines=[dict(CFG["machines"][0], provisioning="PREEMPTIBLE")],
    )
  m = dict(CFG["machines"][0])
  del m["platform"]
  with pytest.raises(config_lib.BackendFileError, match="platform"):
    _cfg(tmp_path, machines=[m])
  with pytest.raises(config_lib.BackendFileError, match="platform"):
    _cfg(tmp_path, machines=[dict(CFG["machines"][0], platform="metal")])
  with pytest.raises(config_lib.BackendFileError, match="accelerator"):
    _cfg(tmp_path, machines=[dict(CFG["machines"][2], platform="cpu")])


def test_max_run_duration_on_every_provisioning_model(tmp_path):
  """max_run_duration applies on every provisioning model."""
  spot, flex, t4 = CFG["machines"]
  # optional on SPOT and STANDARD: the backstop if the supervisor dies
  for m in (
      spot,
      t4,
  ):
    assert (
        _cfg(tmp_path, machines=[dict(m, max_run_duration="24h")])
        .machines[0]
        .max_run_duration
        == "24h"
    )
    assert _cfg(tmp_path, machines=[m]).machines[0].max_run_duration is None
  for m, bad, match in (
      (flex, None, "needs max_run_duration"),  # required on FLEX_START
      (t4, "24 h", "want <N>s"),
      (t4, "1.5h", "want <N>s"),
      (t4, "0s", "30s to 120d"),
      (t4, "29s", "30s to 120d"),
      (t4, "121d", "30s to 120d"),
      (flex, "8d", "30s to 7d"),
  ):
    with pytest.raises(config_lib.BackendFileError, match=match):
      _cfg(tmp_path, machines=[dict(m, max_run_duration=bad)])
  assert _cfg(tmp_path, machines=[dict(t4, max_run_duration="120d")]).machines[
      0
  ]
  assert _cfg(tmp_path, machines=[dict(flex, max_run_duration="7d")]).machines[
      0
  ]
  assert gcp.duration_s("30s") == 30 and gcp.duration_s("2d") == 172800


def test_one_file_with_environment_references(tmp_path, monkeypatch):
  """One config file with environment references."""
  (tmp_path / "pkg").mkdir()
  doc = dict(CFG, project="${MY_PROJECT}", storage="gs://${MY_BUCKET}/runs")
  path = tmp_path / "backend.yaml"
  path.write_text(yaml.safe_dump(doc))
  monkeypatch.setenv("MY_PROJECT", "proj-from-env")
  monkeypatch.delenv("MY_BUCKET", raising=False)
  with pytest.raises(config_lib.BackendFileError, match="MY_BUCKET"):
    config_lib.load(str(path))
  monkeypatch.setenv("MY_BUCKET", "bucket-from-env")
  be = config_lib.make_backend(config_lib.load(str(path)))
  assert isinstance(be, gcp.GcpBackend) and be.cfg.project == "proj-from-env"
  assert be.run_uri("kuhn-a") == "gs://bucket-from-env/runs/kuhn-a"


def test_ordered_fallback_over_machines_and_zones(tmp_path):
  """Provision falls back over machines and zones in order."""
  fake = FakeGcloud(creates=["full", "full", None])
  be = gcp.GcpBackend(_cfg(tmp_path), runner=fake)
  be.prepare(RUN)
  h = be.provision(RUN, 0)
  assert (h["candidate"], h["zone"], h["provisioning"]) == (
      "gpu-flex",
      "zone-c",
      "FLEX_START",
  )
  creates = [
      c for c in fake.calls if c[1:4] == ["compute", "instances", "create"]
  ]
  assert [next(a for a in c if a.startswith("--zone=")) for c in creates] == [
      "--zone=zone-a",
      "--zone=zone-b",
      "--zone=zone-c",
  ]
  fake = FakeGcloud(creates=["full"] * 4)
  be = gcp.GcpBackend(_cfg(tmp_path), runner=fake)
  be.prepare(RUN)
  with pytest.raises(backend_lib.NoCapacityError) as e:
    be.provision(RUN, 0)
  assert [(t["candidate"], t["zone"]) for t in e.value.tried] == [
      ("small-spot", "zone-a"),
      ("small-spot", "zone-b"),
      ("gpu-flex", "zone-c"),
      ("t4", "zone-d"),
  ]

  class Denied(FakeGcloud):
    """Every create is denied (not a capacity error)."""

    # `input` is the keyword the callers pass.
    def __call__(self, argv, input=None):  # pylint: disable=redefined-builtin
      if argv[1:4] == ["compute", "instances", "create"]:
        return subprocess.CompletedProcess(
            argv, 1, "", "ERROR: PERMISSION_DENIED on image"
        )
      return super().__call__(argv, input)

  be = gcp.GcpBackend(_cfg(tmp_path), runner=Denied())
  be.prepare(RUN)
  with pytest.raises(RuntimeError, match="not for capacity"):
    be.provision(RUN, 0)


def test_create_command_known_answer(tmp_path):
  """The create command has its known argv."""
  be = gcp.GcpBackend(_cfg(tmp_path))
  spot, flex, t4 = be.cfg.machines
  a = be.create_argv(spot, "zone-a", "train-kuhn-a-a0", RUN, 0, "/tmp/s.sh")
  assert a[:4] == ["compute", "instances", "create", "train-kuhn-a-a0"]
  for flag in (
      "--project=my-project-123",
      "--zone=zone-a",
      "--machine-type=e2-standard-4",
      "--provisioning-model=SPOT",
      "--instance-termination-action=DELETE",
      "--image-family=ubuntu-2404-lts-amd64",
      "--image-project=ubuntu-os-cloud",
      "--network=default",
      "--subnet=default",
      "--no-address",
      "--service-account=vm@my-project-123.iam.gserviceaccount.com",
      "--labels=owner=me,train-attempt=0,train-run=kuhn-a,ttl=2h",
      "--metadata-from-file=startup-script=/tmp/s.sh",
      "--boot-disk-size=50GB",
  ):
    assert flag in a, flag
  f = be.create_argv(flex, "zone-c", "n", RUN, 1, "s")
  assert {
      "--provisioning-model=FLEX_START",
      "--max-run-duration=24h",
      "--reservation-affinity=none",
      "--maintenance-policy=TERMINATE",
  } <= set(f)
  t = be.create_argv(t4, "zone-d", "n", RUN, 1, "s")
  assert (
      "--accelerator=type=nvidia-tesla-t4,count=1" in t
      and "--maintenance-policy=TERMINATE" in t
  )
  assert not any(x.startswith("--instance-termination-action") for x in t)
  assert not any(x.startswith("--max-run-duration") for x in t)
  # STANDARD with a max run duration: GCE deletes the VM at that age, whoever is
  # watching
  t = be.create_argv(
      dataclasses.replace(t4, max_run_duration="12h"),
      "zone-d",
      "n",
      RUN,
      1,
      "s",
  )
  assert {
      "--provisioning-model=STANDARD",
      "--max-run-duration=12h",
      "--instance-termination-action=DELETE",
      "--maintenance-policy=TERMINATE",
  } <= set(t)
  assert not any(x.startswith("--reservation-affinity") for x in t)
  s = be.create_argv(
      dataclasses.replace(spot, max_run_duration="2h"),
      "zone-a",
      "n",
      RUN,
      1,
      "s",
  )
  assert (
      "--max-run-duration=2h" in s
      and s.count("--instance-termination-action=DELETE") == 1
  )
  assert gcp.vm_name("kuhn-a", 3) == "train-kuhn-a-a3"


def test_machine_label_names_the_gpu():
  """machine_label names the GPU, else the machine type."""
  m = gcp.Machine(
      name="x",
      machine_type="g2-standard-8",
      accelerator=None,
      platform="gpu",
      zones=("z",),
      provisioning="STANDARD",
      boot_disk_gb=50,
      max_run_duration=None,
  )
  assert gcp.machine_label(m) == "L4"
  assert (
      gcp.machine_label(dataclasses.replace(m, machine_type="a3-highgpu-1g"))
      == "H100"
  )
  assert (
      gcp.machine_label(
          dataclasses.replace(
              m,
              machine_type="n1-standard-8",
              accelerator={"type": "nvidia-tesla-t4", "count": 1},
          )
      )
      == "T4"
  )
  assert (
      gcp.machine_label(
          dataclasses.replace(
              m,
              machine_type="n1-standard-8",
              accelerator={"type": "nvidia-tesla-k80", "count": 1},
          )
      )
      == "K80"
  )
  assert (
      gcp.machine_label(
          dataclasses.replace(m, machine_type="e2-standard-4", platform="cpu")
      )
      == "e2-standard-4"
  )


class _RecordingLog:
  """Stands in for the backend's absl logging: records (level, message)."""

  def __init__(self, records):
    self._records = records

  def __getattr__(self, level):
    def log(msg, *args, **kwargs):
      del kwargs
      self._records.append((level, msg % args if args else msg))

    return log


def test_no_capacity_is_one_line_and_keeps_the_raw_error(tmp_path, monkeypatch):
  """NoCapacityError is one line, and the raw error is logged."""
  fake = FakeGcloud(creates=[STOCKOUT, STOCKOUT, "full", NO_QUOTA])
  be = gcp.GcpBackend(_cfg(tmp_path), runner=fake)
  be.prepare(RUN)
  # (level, message) of the backend's log calls, whatever absl's handlers do.
  logged = []
  monkeypatch.setattr(gcp, "logging", _RecordingLog(logged))
  with pytest.raises(backend_lib.NoCapacityError) as e:
    be.provision(RUN, 0)
  assert str(e.value) == (
      "no e2-standard-4 capacity in zone-a, zone-b; no H100 capacity in "
      "zone-c; no T4 quota in zone-d"
  )
  assert "\n" not in str(e.value) and "resource_availability" not in str(
      e.value
  )
  assert [(t["what"], t["reason"]) for t in e.value.tried] == [
      ("e2-standard-4", "capacity"),
      ("e2-standard-4", "capacity"),
      ("H100", "capacity"),
      ("T4", "quota"),
  ]
  assert (
      "resource_availability" in e.value.tried[0]["error"]
  )  # the record keeps it
  debug = [m for level, m in logged if level == "debug"]
  assert any(
      "resource_availability" in d for d in debug
  )  # and so does the debug log
  above = [m for level, m in logged if level != "debug"]
  assert not any("resource_availability" in m for m in above)


def test_supervisor_says_no_capacity_in_one_line(tmp_path):
  """The supervisor reports no capacity in one line."""
  l4 = {
      "name": "l4",
      "machine_type": "g2-standard-8",
      "accelerator": None,
      "platform": "gpu",
      "zones": ["us-central1-a", "us-central1-b"],
      "provisioning": "STANDARD",
      "boot_disk_gb": 100,
      "max_run_duration": "24h",
  }
  fake = FakeGcloud(creates=[STOCKOUT, STOCKOUT, None])
  be = gcp.GcpBackend(_cfg(tmp_path, machines=[l4]), runner=fake)
  orig = fake.__call__

  # `input` is the keyword the callers pass.
  def done_on_create(argv, input=None):  # pylint: disable=redefined-builtin
    """The fake gcloud, with the attempt done once its VM is created.

    Args:
      argv: gcloud's argv.
      input: its stdin.

    Returns:
      its CompletedProcess.
    """
    out = orig(argv, input)
    if argv[1:4] == ["compute", "instances", "create"] and out.returncode == 0:
      fake.objects["gs://my-bucket/runs/kuhn-a/attempts/0/exit_code"] = "0\n"
    return out

  # Reaches private _runner of the code under test.
  be._runner = done_on_create  # pylint: disable=protected-access
  said = []
  sup = supervisor_lib.Supervisor(
      be,
      poll_s=1,
      capacity_wait_s=600,
      sleep=lambda s: None,
      on_event=said.append,
  )
  assert sup.run(RUN) == 0
  assert (
      "kuhn-a attempt 0: no L4 capacity in us-central1-a, us-central1-b; "
      "retrying in 600 s"
  ) in said
  st = sup.read_state("kuhn-a")
  assert "resource_availability" in st["capacity_waits"][0]["tried"][0]["error"]


def test_startup_script_installs_runs_watches_and_copies_back(tmp_path):
  """The startup script installs, runs, watches and copies back."""
  cfg = _cfg(tmp_path)
  s = gcp.startup_script(
      cfg,
      RUN,
      2,
      "gs://my-bucket/runs/kuhn-a",
      "gs://my-bucket/runs/kuhn-a/code/abc.tar.gz",
      cfg.machines[0],
  )
  assert s.startswith("#!/bin/bash")
  assert "A=gs://my-bucket/runs/kuhn-a/attempts/2" in s
  assert (
      "gcloud storage cp gs://my-bucket/runs/kuhn-a/code/abc.tar.gz code.tar.gz"
      in s
  )
  assert (
      "step bash -c 'echo setup-one'" in s
      and "step bash -c 'echo setup-two'" in s
  )
  assert "instance/preempted?wait_for_change=true" in s
  assert f"pkill -TERM -f {backend_lib.TRAINER_MODULE}" in s
  assert (
      f"{hardware.EXPECT_PLATFORM_ENV}=cpu /opt/venv/bin/python -u -m "
      f"{backend_lib.TRAINER_MODULE} "
      "--config=mmd_kuhn_poker_jax --seed=0 "
      "--checkpoint_dir=gs://my-bucket/runs/kuhn-a/ckpt"
  ) in s
  # The started marker is a step (a VM that cannot write it fails, not "lost");
  # the notice watcher long-polls with the metadata ETag (a flip between polls
  # is seen at once).
  assert (
      "step put started" in s
      and "last_etag=$etag" in s
      and 'tolower($1) == "etag:"' in s
  )
  gpu = gcp.startup_script(
      cfg,
      RUN,
      2,
      "gs://my-bucket/runs/kuhn-a",
      "gs://my-bucket/runs/kuhn-a/code/abc.tar.gz",
      cfg.machines[2],
  )
  assert (
      f"{hardware.EXPECT_PLATFORM_ENV}=gpu /opt/venv/bin/python -u -m "
      f"{backend_lib.TRAINER_MODULE}"
  ) in gpu
  assert (
      'put exit_code "$rc"' in s
      and '"$A/train.log"' in s
      and "shutdown -h now" in s
  )
  assert (
      s.index("put started")
      < s.index("code.tar.gz")
      < s.index("setup-one")
      < s.index("watch_preemption &")
      < s.index(f"{backend_lib.TRAINER_MODULE} --config")
  )


def test_poll_and_teardown(tmp_path):
  """`poll` reads the attempt's state; `teardown` deletes the VM."""
  uri = "gs://my-bucket/runs/kuhn-a/attempts/0/"
  h = {
      "backend": "gcp",
      "attempt": 0,
      "name": "train-kuhn-a-a0",
      "zone": "zone-a",
      "run_id": "kuhn-a",
  }
  started = {uri + "started": "t host"}
  cases = [
      (
          FakeGcloud(objects={uri + "exit_code": "0\n"}),
          backend_lib.SUCCEEDED,
          0,
      ),
      (
          FakeGcloud(objects={uri + "exit_code": "143\n"}),
          backend_lib.PREEMPTED,
          143,
      ),
      (FakeGcloud(objects={uri + "exit_code": "2\n"}), backend_lib.FAILED, 2),
      (FakeGcloud(describe="RUNNING"), backend_lib.RUNNING, None),
      # GCE's preemption grace: the trainer is checkpointing; do not delete the
      # VM yet.
      (
          FakeGcloud(describe="STOPPING", objects={uri + "preempted": "t"}),
          backend_lib.RUNNING,
          None,
      ),
      (FakeGcloud(describe="SUSPENDING"), backend_lib.RUNNING, None),
      (
          FakeGcloud(describe="", objects={uri + "preempted": "t"}),
          backend_lib.PREEMPTED,
          None,
      ),
      (
          FakeGcloud(
              describe="TERMINATED",
              ops='[{"targetLink": ".../instances/train-kuhn-a-a0"}]',
          ),
          backend_lib.PREEMPTED,
          None,
      ),
  ]
  for fake, state, rc in cases:
    st = gcp.GcpBackend(_cfg(tmp_path), runner=fake).poll(h)
    assert (st.state, st.rc) == (state, rc), (st, fake.objects)
  lost = gcp.GcpBackend(
      _cfg(tmp_path), runner=FakeGcloud(describe="", objects=started)
  ).poll(h)
  assert lost.state == backend_lib.PREEMPTED and "lost" in lost.detail
  never = gcp.GcpBackend(
      _cfg(tmp_path), runner=FakeGcloud(describe="TERMINATED")
  ).poll(h)
  assert never.state == backend_lib.FAILED and "started marker" in never.detail
  fake = FakeGcloud()
  be = gcp.GcpBackend(_cfg(tmp_path), runner=fake)
  be.teardown(h)
  assert fake.calls[-1][:5] == [
      "gcloud",
      "compute",
      "instances",
      "delete",
      "train-kuhn-a-a0",
  ]
  assert be.adopt(h) == h
  assert (
      gcp.GcpBackend(_cfg(tmp_path), runner=FakeGcloud(describe="")).adopt(h)
      is None
  )


def test_only_the_vm_s_own_not_found_reads_as_gone(tmp_path):
  """A describe error that names another resource (a wrong zone) raises.

  It is not read as a VM that no longer exists.

  Args:
    tmp_path: pytest's per-test directory.
  """
  h = {
      "backend": "gcp",
      "attempt": 0,
      "name": "train-kuhn-a-a0",
      "zone": "zone-a",
      "run_id": "kuhn-a",
  }

  class WrongZone(FakeGcloud):
    """describe fails naming another resource."""

    # `input` is the keyword the callers pass.
    def __call__(self, argv, input=None):  # pylint: disable=redefined-builtin
      if argv[1:4] == ["compute", "instances", "describe"]:
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(
            argv,
            1,
            "",
            "ERROR: The resource 'projects/p/zones/zone-a' was not found",
        )
      return super().__call__(argv, input)

  with pytest.raises(RuntimeError, match="describe"):
    gcp.GcpBackend(_cfg(tmp_path), runner=WrongZone()).poll(h)


def test_storage_and_supervisor_state_through_gcloud(tmp_path):
  """Storage and the supervisor's state go through gcloud."""
  fake = FakeGcloud(creates=[None], objects={})
  be = gcp.GcpBackend(_cfg(tmp_path), runner=fake)
  orig = fake.__call__

  # `input` is the keyword the callers pass.
  def done_on_create(argv, input=None):  # pylint: disable=redefined-builtin
    """The fake gcloud, with the attempt done once its VM is created.

    Args:
      argv: gcloud's argv.
      input: its stdin.

    Returns:
      its CompletedProcess.
    """
    out = orig(argv, input)
    if argv[1:4] == ["compute", "instances", "create"]:
      fake.objects["gs://my-bucket/runs/kuhn-a/attempts/0/exit_code"] = "0\n"
    return out

  # Reaches private _runner of the code under test.
  be._runner = done_on_create  # pylint: disable=protected-access
  assert (
      supervisor_lib.Supervisor(be, poll_s=1, sleep=lambda s: None).run(RUN)
      == 0
  )
  st = supervisor_lib.Supervisor(be).read_state("kuhn-a")
  assert (
      st["final"] == "done"
      and st["attempts"][0]["handle"]["candidate"] == "small-spot"
  )
  assert any(
      k.startswith("gs://my-bucket/runs/kuhn-a/code/") for k in fake.objects
  )
  assert ["gcloud", "compute", "instances", "delete"] == [
      c for c in fake.calls if c[1:4] == ["compute", "instances", "delete"]
  ][0][:4]


def test_dry_run_prints_the_plan_and_creates_nothing(tmp_path, capsys):
  """A dry run prints the plan and creates nothing."""
  (tmp_path / "pkg").mkdir()
  (tmp_path / "pkg" / "m.py").write_text("X = 1\n")
  path = tmp_path / "backend.yaml"
  path.write_text(yaml.safe_dump(CFG))
  rc = run_cli.main([
      "prog",
      f"--backend_config={path}",
      "--run_id=kuhn-a",
      "--dry_run",
      "--",
      "--config=mmd_kuhn_poker_jax",
      "--seed=0",
  ])
  out = capsys.readouterr().out
  assert rc == 0 and "nothing created" in out
  assert out.count("gcloud compute instances create train-kuhn-a-a0") == 4
  assert (
      "#!/bin/bash" in out
      and "--checkpoint_dir=gs://my-bucket/runs/kuhn-a/ckpt" in out
  )


class Flaky(FakeGcloud):
  """Storage calls fail `fails` times (a transient error), then work.

  Attributes:
    fails: the failures still to come.
  """

  def __init__(self, fails, **kw):
    super().__init__(**kw)
    self.fails = fails

  # `input` is the keyword the callers pass.
  def __call__(self, argv, input=None):  # pylint: disable=redefined-builtin
    if argv[1] == "storage" and self.fails:
      self.fails -= 1
      if self.fails % 2:
        raise subprocess.TimeoutExpired(argv, 900)
      return subprocess.CompletedProcess(argv, 1, "", "ERROR: 503 backendError")
    return super().__call__(argv, input)


def test_storage_retries_transient_failures_and_raises_on_persistent_ones(
    tmp_path,
):
  """Storage retries transient failures and raises on persistent ones."""
  waits = []
  uri = "gs://my-bucket/runs/kuhn-a/supervisor.json"
  be = gcp.GcpBackend(
      _cfg(tmp_path), runner=Flaky(3, objects={uri: "{}"}), sleep=waits.append
  )
  assert be.read_text(uri) == "{}" and waits == [2.0, 4.0, 8.0]
  waits.clear()
  be.write_text(uri, "x")  # works at once now
  assert not waits
  be = gcp.GcpBackend(_cfg(tmp_path), runner=Flaky(99), sleep=waits.append)
  with pytest.raises(RuntimeError, match="failed 5x"):
    be.write_text(uri, "x")
  assert waits == [2.0, 4.0, 8.0, 16.0]
  waits.clear()
  # a missing object is an answer, not a failure: no retry
  assert (
      gcp.GcpBackend(
          _cfg(tmp_path), runner=FakeGcloud(), sleep=waits.append
      ).read_text(uri)
      is None
      and not waits
  )
  calls = []  # GcsStorage (storage without a GcpBackend) retries the same way

  # `input` is the keyword the callers pass.
  def runner(argv, input=None):  # pylint: disable=redefined-builtin
    """A gcloud that fails per the script, else succeeds.

    Args:
      argv: gcloud's argv.
      input: its stdin.

    Returns:
      its CompletedProcess.
    """
    del input
    calls.append(argv)
    if len(calls) < 3:
      return subprocess.CompletedProcess(argv, 1, "", "ERROR: 429 rate")
    return subprocess.CompletedProcess(argv, 0, "z", "")

  assert (
      storage_lib.GcsStorage(runner, sleep=waits.append).read_text(uri) == "z"
      and len(calls) == 3
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

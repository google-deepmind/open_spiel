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

"""Tests of the cloud launcher against a fake gcloud.

The cloud launcher against a fake gcloud: preflight with fixes, the dry run
creates nothing, a launch walks zones on capacity errors, waits for its own
attempt's exit code and deletes only its own labelled VM; the VM's startup
script, run under a fake VM, restores, publishes and keeps the VM on a failed
upload. Nothing touches GCP.
"""

import base64
import dataclasses
import io
import json
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import textwrap
import time
from typing import Any, Optional

import pytest

from open_spiel.python.scale import plugins
from open_spiel.python.scale.launch import cloud
from open_spiel.python.scale.launch import subprocess_env

OK_REGION = {
    "quotas": [
        {"metric": "CPUS", "limit": 24.0, "usage": 0.0},
        {"metric": "PREEMPTIBLE_CPUS", "limit": 0.0, "usage": 0.0},
    ]
}


class FakeGcloud:
  """Answers gcloud calls from a table of (argument prefix -> reply).

  Attributes:
    calls: every call's arguments, in order.
    over: replies that override the defaults, argument prefix -> reply.
    exit_code_after: the exit_code poll from which the exit code exists.
    polls: exit_code polls so far.
  """

  def __init__(self, over=None, exit_code_after=1):
    self.calls = []
    self.over = dict(over or {})
    self.exit_code_after = exit_code_after
    self.polls = 0

  def __call__(self, args):
    args = list(args)
    self.calls.append(args)
    j = " ".join(args)

    def reply(out="", rc=0, err=""):
      """A CompletedProcess of this call.

      Args:
        out: its stdout.
        rc: its return code.
        err: its stderr.

      Returns:
        the CompletedProcess.
      """
      return subprocess.CompletedProcess(args, rc, out, err)

    for key, val in self.over.items():
      if j.startswith(key):
        return reply(*val) if isinstance(val, tuple) else reply(val)
    if j.startswith("auth list"):
      return reply("me@example.com\n")
    if j.startswith("auth application-default"):
      return reply("token")
    if j.startswith("projects describe"):
      return reply("my-project")
    if j.startswith("services list"):
      return reply("compute.googleapis.com\n")
    if j.startswith("storage buckets describe"):
      return reply("my-bucket")
    if j.startswith("compute regions describe"):
      return reply(json.dumps(OK_REGION))
    if j.startswith("compute zones list"):
      return reply("us-central1-b\nus-central1-a\n")
    if j.startswith("compute instances list"):
      created = any(
          c[:3] == ["compute", "instances", "create"] for c in self.calls
      )
      deleted = any(
          c[:3] == ["compute", "instances", "delete"] for c in self.calls
      )
      return reply("r1 us-central1-a\n" if created and not deleted else "")
    if j.startswith("storage cat"):
      self.polls += 1
      return (
          reply("0\n") if self.polls >= self.exit_code_after else reply("", 1)
      )
    return reply()


def settings(**kw):
  """Resolved settings for my-project, my-bucket and run r1.

  Args:
    **kw: further flags.

  Returns:
    the Settings.
  """
  cli = {
      "project": "my-project",
      "bucket": "gs://my-bucket",
      "run_id": "r1",
      **kw,
  }
  return cloud.resolve_settings(cli, {})


def which(name):
  """Finds gcloud.

  Args:
    name: the program asked for (unused).

  Returns:
    gcloud's path.
  """
  del name
  return "/usr/bin/gcloud"


def test_defaults_and_bucket_forms():
  """Defaults fill unset flags; the bucket may omit gs://."""
  s = settings()
  assert (s.bucket, s.region, s.machine_type, s.spot, s.max_run_hours) == (
      "my-bucket",
      "us-central1",
      "e2-standard-8",
      True,
      2.0,
  )
  assert s.run_uri == f"gs://my-bucket/{cloud.PREFIX}/r1"
  auto = cloud.resolve_settings({"project": "p", "bucket": "b-1"}, {})
  assert cloud.RUN_ID_RE.fullmatch(auto.run_id) and auto.run_id.startswith(
      cloud.SHORT
  )


@pytest.mark.parametrize(
    "cli,match",
    [
        ({"bucket": "b"}, "--project"),
        ({"project": "p"}, "--bucket"),
        ({"project": "p", "bucket": "b-1", "run_id": "Bad_Id"}, "run_id"),
    ],
)
def test_bad_settings_are_refused(cli, match):
  """Bad settings are refused with a message saying why."""
  with pytest.raises(cloud.LaunchError, match=match):
    cloud.resolve_settings(cli, {})


def test_preflight_passes_when_everything_is_there():
  """Preflight finds no problem when every prerequisite is there."""
  assert not cloud.preflight(settings(), FakeGcloud(), which)


def test_preflight_prints_the_fix_for_each_missing_piece():
  """Preflight gives the fix of each missing prerequisite."""
  probs = cloud.preflight(
      settings(),
      FakeGcloud({
          "auth application-default": ("", 1),
          "storage buckets describe": ("", 1),
      }),
      which,
  )
  fixes = [p.fix for p in probs]
  assert "gcloud auth application-default login" in fixes
  assert any("gcloud storage buckets create gs://my-bucket" in f for f in fixes)
  assert cloud.preflight(settings(), FakeGcloud(), lambda _: None)[
      0
  ].fix.startswith("install")
  api = cloud.preflight(settings(), FakeGcloud({"services list": ""}), which)
  assert (
      api[-1].fix
      == "gcloud services enable compute.googleapis.com --project=my-project"
  )


def test_preflight_looks_up_which_at_call_time(monkeypatch):
  """Preflight's default `which` is shutil.which as it is when called.

  A default bound when the module loads would ignore the patch, and the
  result would depend on whether gcloud is installed on the test machine.

  Args:
    monkeypatch: patches cloud.shutil.which.
  """
  monkeypatch.setattr(cloud.shutil, "which", which)
  assert not cloud.preflight(settings(), FakeGcloud())
  monkeypatch.setattr(cloud.shutil, "which", lambda _: None)
  (prob,) = cloud.preflight(settings(), FakeGcloud())
  assert "gcloud CLI is not on PATH" in prob.what


def test_preflight_reports_short_quota():
  """Preflight reports a quota too small for the VM."""
  region = {"quotas": [{"metric": "CPUS", "limit": 4.0, "usage": 0.0}]}
  probs = cloud.preflight(
      settings(),
      FakeGcloud({"compute regions describe": json.dumps(region)}),
      which,
  )
  assert (
      len(probs) == 1 and "CPUS" in probs[0].what and "quotas" in probs[0].fix
  )
  gpu = cloud.preflight(
      settings(accelerator="nvidia-tesla-t4:1"), FakeGcloud(), which
  )
  assert any("NVIDIA_T4_GPUS" in p.what for p in gpu)


def test_dry_run_creates_and_uploads_nothing(capsys):
  """A dry run creates and uploads nothing."""
  g = FakeGcloud()
  assert (
      cloud.launch(
          settings(),
          ["--config=ppo_tic_tac_toe"],
          dry_run=True,
          detach=False,
          gcloud=g,
      )
      == 0
  )
  out = capsys.readouterr().out
  assert "DRY RUN" in out and "--provisioning-model=SPOT" in out
  assert (
      f"--labels={cloud.LABEL_KEY}=r1" in out
      and "--max-run-duration=120m" in out
  )
  assert "--instance-termination-action=DELETE" in out
  assert not [c for c in g.calls if c[:3] == ["compute", "instances", "create"]]
  assert not [c for c in g.calls if c[:2] == ["storage", "cp"]]


def test_launch_walks_zones_waits_and_deletes_its_vm(capsys):
  """A launch walks zones on capacity errors, waits, and deletes its VM."""

  class Capacity(FakeGcloud):
    """No capacity in us-central1-a."""

    def __call__(self, args):
      if (
          list(args[:3]) == ["compute", "instances", "create"]
          and "--zone=us-central1-a" in args
      ):
        self.calls.append(list(args))
        return subprocess.CompletedProcess(
            args, 1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"
        )
      return super().__call__(args)

  g = Capacity(exit_code_after=2)
  rc = cloud.launch(
      settings(),
      ["--config=ppo_tic_tac_toe"],
      dry_run=False,
      detach=False,
      gcloud=g,
      sleep=lambda s: None,
  )
  assert rc == 0
  creates = [c for c in g.calls if c[:3] == ["compute", "instances", "create"]]
  assert [next(a for a in c if a.startswith("--zone")) for c in creates] == [
      "--zone=us-central1-a",
      "--zone=us-central1-b",
  ]
  deletes = [c for c in g.calls if c[:3] == ["compute", "instances", "delete"]]
  assert len(deletes) == 1 and deletes[0][3] == "r1"
  assert "finished with exit code 0" in capsys.readouterr().out


def test_a_create_command_replaces_gcloud_create_and_keeps_zone_fallback(
    capsys,
):
  """A create command replaces gcloud's create and keeps the zone fallback."""
  del capsys
  g = FakeGcloud(exit_code_after=2)
  ran = []

  def run(argv):
    """The create command: a stockout in us-central1-a, else a VM.

    Args:
      argv: the create command's argv.

    Returns:
      its CompletedProcess.
    """
    ran.append(list(argv))
    if "--zone=us-central1-a" in argv:
      return subprocess.CompletedProcess(
          argv, 75, "", "STOCKOUT in us-central1-a"
      )
    # VM exists
    g.calls.append(["compute", "instances", "create", "(by the wrapper)"])
    return subprocess.CompletedProcess(argv, 0, "", "")

  s = settings(create_command="my-wrapper --team t")
  rc = cloud.launch(
      s,
      ["--config=ppo_tic_tac_toe"],
      dry_run=False,
      detach=False,
      gcloud=g,
      sleep=lambda s: None,
      run=run,
  )
  assert rc == 0
  assert [a[:4] for a in ran] == [["my-wrapper", "--team", "t", "r1"]] * 2
  assert [next(x for x in a if x.startswith("--zone")) for a in ran] == [
      "--zone=us-central1-a",
      "--zone=us-central1-b",
  ]
  assert (
      f"--labels={cloud.LABEL_KEY}=r1,created-by={cloud.CREATED_BY}" in ran[1]
  )
  creates = [c for c in g.calls if c[:3] == ["compute", "instances", "create"]]
  assert creates == [["compute", "instances", "create", "(by the wrapper)"]]
  deletes = [c for c in g.calls if c[:3] == ["compute", "instances", "delete"]]
  assert len(deletes) == 1 and deletes[0][3] == "r1"


def test_a_create_command_shows_in_the_dry_run(capsys):
  """The dry run shows the create command."""
  g = FakeGcloud()
  s = settings(create_command="my-wrapper")
  assert (
      cloud.launch(
          s,
          ["--config=ppo_tic_tac_toe"],
          dry_run=True,
          detach=False,
          gcloud=g,
          run=lambda a: pytest.fail("ran in a dry run"),
      )
      == 0
  )
  assert "  my-wrapper r1 --project=my-project" in capsys.readouterr().out


def test_down_deletes_only_labelled_vms():
  """--down deletes only the VMs labelled for the run."""
  g = FakeGcloud({"compute instances list": "r9 us-central1-f\n"})
  assert cloud.down("my-project", "r9", g) == 1
  listing = next(
      c for c in g.calls if c[:3] == ["compute", "instances", "list"]
  )
  flt = next(a for a in listing if a.startswith("--filter="))
  assert (
      f"labels.{cloud.LABEL_KEY}=r9" in flt
      and f"labels.created-by={cloud.CREATED_BY}" in flt
  )
  none = FakeGcloud({"compute instances list": ""})
  assert cloud.down("my-project", "r9", none) == 0
  assert not [
      c for c in none.calls if c[:3] == ["compute", "instances", "delete"]
  ]


def test_example_config_loads_with_its_placeholders(monkeypatch):
  """The example config loads with its ${VAR} placeholders set."""
  monkeypatch.setenv("PROJECT", "my-project")
  monkeypatch.setenv("BUCKET", "my-bucket")
  file = cloud.load_config_file(str(cloud.EXAMPLE_CONFIG))
  s = cloud.resolve_settings({}, file)
  assert (s.project, s.bucket) == ("my-project", "my-bucket")
  assert file["train_args"] == ["--config=ppo_tic_tac_toe"]
  monkeypatch.delenv("BUCKET")
  with pytest.raises(cloud.LaunchError, match="BUCKET"):
    cloud.load_config_file(str(cloud.EXAMPLE_CONFIG))


def test_main_requires_trainer_flags_and_stops_on_preflight(
    capsys, monkeypatch
):
  """`main` requires trainer flags and stops on a preflight problem."""
  monkeypatch.setattr(cloud.shutil, "which", which)
  assert (
      cloud.main(["--project=p", "--bucket=my-bucket"], gcloud=FakeGcloud())
      == 2
  )
  rc = cloud.main(
      ["--project=p", "--bucket=my-bucket", "--dry_run", "--", "--config=x"],
      gcloud=FakeGcloud({"auth list": ""}),
  )
  assert rc == 2 and "gcloud auth login" in capsys.readouterr().out


def test_a_requirements_file_replaces_the_local_versions(tmp_path, capsys):
  """A requirements file replaces the locally installed versions."""
  req = tmp_path / "r.txt"
  req.write_text("# pins\nopen-spiel\njax==9.9\n")
  assert cloud.read_requirements(str(req)) == ["open-spiel", "jax==9.9"]
  cloud.launch(
      settings(),
      ["--config=x"],
      dry_run=True,
      detach=False,
      gcloud=FakeGcloud(),
      reqs=cloud.read_requirements(str(req)),
  )
  assert "requirements: open-spiel jax==9.9" in capsys.readouterr().out


def test_code_bundle_holds_the_packages_and_pins():
  """The code bundle holds the shipped packages and the pins."""
  data = cloud.code_bundle(["jax==1"])
  names = tarfile.open(fileobj=io.BytesIO(data)).getnames()
  assert cloud.shipped_packages()[0] == cloud.FRAMEWORK
  for pkg in cloud.shipped_packages():
    assert f"pkgs/{pkg.replace('.', '/')}/__init__.py" in names
  assert "requirements.txt" in names and not any(
      "__pycache__" in n for n in names
  )


def test_startup_script_waits_out_a_held_apt_lock():
  """The startup script waits out a held apt lock.

  The VM's boot-time apt may hold /var/lib/apt/lists/lock, which
  DPkg::Lock::Timeout does not wait for (a real run exited 22 on it).
  """
  s = cloud.STARTUP_SCRIPT.replace("\\\n  ", "")  # joins continued lines
  for cmd in (
      "apt-get -o DPkg::Lock::Timeout=60 update -qq",
      "apt-get -o DPkg::Lock::Timeout=60 install -yqq python3-venv",
  ):
    assert f"apt_retry {cmd} || finish 22" in s
  assert "\napt-get " not in s  # every apt call goes through the retry


def test_the_games_package_is_shipped_only_when_importable(monkeypatch):
  """The games package is shipped only when it is importable."""
  monkeypatch.setattr(cloud, "GAMES", "no_such_games_package_xyz.sub")
  assert cloud.shipped_packages() == (cloud.FRAMEWORK,)


def _metadata(args, key):
  """The base64-JSON metadata value `key` of a create call.

  Args:
    args: The create call's arguments.
    key: The metadata key.

  Returns:
    The decoded value.
  """
  meta = next(a for a in args if a.startswith("--metadata="))[
      len("--metadata=") :
  ]
  value = dict(kv.split("=", 1) for kv in meta.split(","))[key]
  return json.loads(base64.b64decode(value))


def test_plugins_reach_the_trainer_in_the_plugins_variable(monkeypatch):
  """Plugins reach the trainer in the plugins variable."""
  monkeypatch.delenv(plugins.ENV, raising=False)
  args = cloud.create_args(settings(), "z", ["--config=x"], "s.sh", "a1")
  assert not _metadata(args, "train-env")
  assert _metadata(args, "train-args") == ["--config=x"]
  s = settings(plugins="acme.sinks, acme.kinds")
  assert s.plugins == ("acme.sinks", "acme.kinds")
  args = cloud.create_args(s, "z", ["--config=x"], "s.sh", "a1")
  assert _metadata(args, "train-env") == {plugins.ENV: "acme.sinks,acme.kinds"}
  env = cloud.resolve_settings(
      {"project": "p", "bucket": "b-1"}, {}, environ={plugins.ENV: "acme.env"}
  )
  assert env.plugins == ("acme.env",)
  assert 'eval "$ENV_EXPORTS"' in cloud.STARTUP_SCRIPT
  with pytest.raises(cloud.LaunchError, match="dotted module"):
    settings(plugins="not a module")


# -- attempts: only this launch's exit code ends it -------------------------

RUN_URI = f"gs://my-bucket/{cloud.PREFIX}/r1"


def _attempt_of(calls):
  """The attempt id of the create call in `calls`.

  Args:
    calls: The gcloud calls.

  Returns:
    The attempt id.
  """
  create = next(c for c in calls if c[:3] == ["compute", "instances", "create"])
  meta = next(a for a in create if a.startswith("--metadata="))[
      len("--metadata=") :
  ]
  return dict(kv.split("=", 1) for kv in meta.split(","))["attempt-id"]


class Bucket(FakeGcloud):
  """storage cat answers only the objects present.

  This attempt's exit_code appears at the `finish_at`-th read.

  Attributes:
    objects: the bucket's objects, uri -> text.
    finish_at: the read at which this attempt's exit_code appears.
    marker: the exit_code written then.
    reads: every object read, in order.
    deleted_at: the read count when the exit code was deleted, or None.
    calls: every call's arguments, in order (FakeGcloud's).
  """

  def __init__(self, objects, finish_at, marker):
    super().__init__()
    self.objects = dict(objects)
    self.finish_at, self.marker = finish_at, marker
    self.reads, self.deleted_at = [], None

  def __call__(self, args):
    args = list(args)
    if args[:2] == ["storage", "cat"]:
      self.calls.append(args)
      self.reads.append(args[2])
      assert len(self.reads) < 50, "the launcher never stopped polling"
      if len(self.reads) >= self.finish_at:
        self.objects[
            f"{RUN_URI}/attempts/{_attempt_of(self.calls)}/exit_code"
        ] = self.marker
      found = self.objects.get(args[2])
      return subprocess.CompletedProcess(
          args, 0 if found else 1, found or "", ""
      )
    if (
        args[:3] == ["compute", "instances", "delete"]
        and self.deleted_at is None
    ):
      self.deleted_at = len(self.reads)
    return super().__call__(args)


def test_an_earlier_attempts_exit_code_never_ends_a_new_launch(capsys):
  """An earlier attempt's exit code never ends a new launch.

  Relaunching a run (a retry, or a resume) leaves the earlier launch's
  exit marker in the bucket; the new VM runs until its own attempt ends.

  Args:
    capsys: pytest's captured output.
  """
  old = {
      f"{RUN_URI}/exit_code": "0\n",
      f"{RUN_URI}/attempts/20260101-000000-deadbeef/exit_code": "1\n",
  }
  g = Bucket(old, finish_at=4, marker="7\n")
  rc = cloud.launch(
      settings(),
      ["--config=x"],
      dry_run=False,
      detach=False,
      gcloud=g,
      sleep=lambda s: None,
  )
  # not before its own marker
  assert g.deleted_at is not None and g.deleted_at >= 4
  assert rc == 7 and "finished with exit code 7" in capsys.readouterr().out
  attempt = _attempt_of(g.calls)
  assert attempt != "20260101-000000-deadbeef"
  assert set(g.reads) == {f"{RUN_URI}/attempts/{attempt}/exit_code"}
  ckpt = [
      a
      for a in next(
          c for c in g.calls if c[:3] == ["compute", "instances", "create"]
      )
      if a.startswith("--metadata=")
  ][0]
  # the checkpoint lineage stays under the run
  assert f"run-uri={RUN_URI}," in ckpt


def test_a_failed_final_upload_keeps_the_vm_and_is_not_success(capsys):
  """A failed final upload keeps the VM and is not a success."""
  g = Bucket({}, finish_at=2, marker="storage-error\n")
  rc = cloud.launch(
      settings(),
      ["--config=x"],
      dry_run=False,
      detach=False,
      gcloud=g,
      sleep=lambda s: None,
  )
  assert rc == 30 == cloud.STORAGE_ERROR_RC
  assert not [c for c in g.calls if c[:3] == ["compute", "instances", "delete"]]
  out = capsys.readouterr().out
  assert (
      "the VM is kept" in out and "/opt/run/ckpt" in out and "--down=r1" in out
  )


# -- the VM's startup script, run under a fake VM ---------------------------

FAKE_GCLOUD = r"""
import json, os, pathlib, shutil, sys, time
root = pathlib.Path(os.environ["FAKE_ROOT"])
args = sys.argv[1:]

def local(uri):
  assert uri.startswith("gs://"), uri
  return root / "gcs" / uri[len("gs://"):]

def log(*entry):
  with open(root / "log", "a") as f:
    f.write(json.dumps([time.time(), *entry]) + "\n")

def fails(op, uri):
  rules = json.loads((root / "fail.json").read_text())
  return any(op == o and part in uri for o, part in rules)

def files(d):
  d = pathlib.Path(d)
  return ({p.relative_to(d).as_posix(): p.read_bytes()
            for p in sorted(d.rglob("*")) if p.is_file()}
           if d.is_dir() else {})

def write(d, snap):
  for rel, data in snap.items():
    out = pathlib.Path(d) / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)

def fail(op):
  log(op, "fail")
  print("ERROR: (gcloud.storage) HTTPError 503: backend error", file=sys.stderr)
  sys.exit(1)

if args[:2] == ["storage", "rsync"]:
  src, dst = args[-2], args[-1]
  op = "rsync-up" if dst.startswith("gs://") else "rsync-down"
  if fails(op, dst if op == "rsync-up" else src):
    fail(op)
  flight = root / "inflight" / str(os.getpid())
  flight.parent.mkdir(exist_ok=True)
  flight.touch()
  snap = files(local(src) if op == "rsync-down" else src)  # as the sync began
  if op == "rsync-up":
    queue = json.loads((root / "delays.json").read_text())
    if queue:
      (root / "delays.json").write_text(json.dumps(queue[1:]))
      time.sleep(queue[0])
  write(local(dst) if op == "rsync-up" else dst, snap)
  log(op, "done", snap.get("ckpt.txt", b"").decode())
  flight.unlink()
elif args[:2] == ["storage", "ls"]:
  uri = args[2][:-len("/**")] if args[2].endswith("/**") else args[2]
  if fails("ls", uri):
    fail("ls")
  found = files(local(uri))
  if not found:
    print("ERROR: (gcloud.storage.ls) One or more URLs matched no objects.",
          file=sys.stderr)
    sys.exit(1)
  print("\n".join(f"{uri}/{k}" for k in found))
elif args[:2] == ["storage", "cp"]:
  src, dst = args[2], args[3]
  if dst.startswith("gs://"):
    if fails("cp", dst):
      fail("cp")
    local(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, local(dst))
    log("cp", dst)
  else:
    shutil.copyfile(local(src), dst)
elif args[:3] == ["compute", "instances", "delete"]:
  log("delete", args[3])
else:
  print(f"fake gcloud: unexpected {args}", file=sys.stderr)
  sys.exit(2)
"""

FAKE_TRAINER = r"""
import json, os, pathlib, sys, time
root = pathlib.Path(os.environ["FAKE_ROOT"])
ckpt = pathlib.Path(
    next(a for a in sys.argv if a.startswith("--checkpoint_dir="))
    .split("=", 1)[1])
seen = ckpt / "ckpt.txt"
(root / "trainer_saw.txt").write_text(seen.read_text() if seen.exists() else "")
for pause, content in json.loads((root / "trainer.json").read_text()):
  time.sleep(pause)
  (ckpt / "tmp").write_text(content)
  os.replace(ckpt / "tmp", seen)
"""


def _executable(path, text):
  path.write_text(text)
  path.chmod(0o755)


@dataclasses.dataclass
class VmRun:
  """What a run of the startup script under a fake VM did.

  Attributes:
    rc: the script's exit code.
    out: its output.
    log: the fake gcloud's calls.
    exit_code: this attempt's exit code, in the bucket.
    remote: the run's checkpoint, in the bucket.
    local: the VM's checkpoint.
    trainer_saw: the checkpoint the trainer started from (None: it never ran).
  """

  rc: int
  out: str
  log: list[Any]
  exit_code: Optional[str]  # this attempt's, in the bucket
  remote: Optional[str]  # the run's checkpoint, in the bucket
  local: Optional[str]  # the VM's checkpoint
  trainer_saw: Optional[str]  # None: the trainer never ran


def run_vm(
    root, *, remote_ckpt=None, fail=(), trainer=((0, "ckpt-b"),), delays=()
):
  """Runs STARTUP_SCRIPT with fake gcloud, curl, apt, venv and trainer.

  The bucket is a directory.

  Args:
    root: the test's temp dir.
    remote_ckpt: the run's checkpoint in the bucket (None: no checkpoint).
    fail: (op, uri part) whose gcloud calls fail.
    trainer: the fake trainer's (pause, checkpoint content) steps.
    delays: seconds each background sync's upload takes.

  Returns:
    the VmRun.
  """
  # Reaches private _b64json of the code under test.
  # pylint: disable=protected-access
  # The real interpreter as a shell command (it may be a launcher with args).
  real_py = " ".join(shlex.quote(a) for a in subprocess_env.python_command())
  run = root / "gcs" / RUN_URI[len("gs://") :]
  bin_dir, meta, site = root / "bin", root / "meta", root / "site"
  for d in (bin_dir, meta, site, run):
    d.mkdir(parents=True)
  buf = io.BytesIO()
  with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for name, data in (
        ("pkgs/fakepkg/__init__.py", b""),
        ("requirements.txt", b""),
    ):
      info = tarfile.TarInfo(name)
      info.size = len(data)
      tar.addfile(info, io.BytesIO(data))
  (run / "code.tgz").write_bytes(buf.getvalue())
  if remote_ckpt is not None:
    (run / "ckpt").mkdir()
    (run / "ckpt" / "ckpt.txt").write_text(remote_ckpt)
  for key, value in {
      "run-uri": RUN_URI,
      "attempt-id": "a2",
      "train-module": "fake_trainer",
      "train-args": cloud._b64json(["--config=x"]),
      "train-env": cloud._b64json({}),
  }.items():
    (meta / key).write_text(value)
  (root / "fail.json").write_text(json.dumps(list(fail)))
  (root / "delays.json").write_text(json.dumps(list(delays)))
  (root / "trainer.json").write_text(json.dumps(list(trainer)))
  (site / "fake_trainer.py").write_text(FAKE_TRAINER)
  (bin_dir / "gcloud.py").write_text(FAKE_GCLOUD)
  gcloud_py = shlex.quote(str(bin_dir / "gcloud.py"))
  _executable(
      bin_dir / "gcloud", f'#!/bin/bash\nexec {real_py} {gcloud_py} "$@"\n'
  )
  _executable(
      bin_dir / "curl",
      textwrap.dedent("""\
      #!/bin/bash
      url="${@: -1}"
      case "$url" in
        */attributes/*)
          f="$FAKE_ROOT/meta/${url##*/attributes/}"
          [ -f "$f" ] || exit 22; cat "$f" ;;
        */name) echo r1 ;;
        */zone) echo projects/1/zones/us-central1-a ;;
        *) exit 22 ;;
      esac
      """),
  )
  _executable(bin_dir / "apt-get", "#!/bin/bash\nexit 0\n")
  _executable(
      bin_dir / "poweroff",
      '#!/bin/bash\necho \'[0, "poweroff"]\' >> "$FAKE_ROOT/log"\n',
  )
  # `python3 -m venv venv` makes a venv whose python imports from site/ and
  # names site/ as its site-packages; any other python3 is the real one.
  venv_bin = root / "venv-bin"
  venv_bin.mkdir()
  _executable(venv_bin / "pip", "#!/bin/bash\nexit 0\n")
  _executable(
      venv_bin / "python",
      "#!/bin/bash\n"
      'case "$*" in *sysconfig*) echo "$FAKE_ROOT/site"; exit 0 ;; esac\n'
      f'PYTHONPATH="$FAKE_ROOT/site" exec {real_py} "$@"\n',
  )
  _executable(
      bin_dir / "python3",
      "#!/bin/bash\n"
      'if [ "$1 $2" = "-m venv" ]; then\n'
      '  mkdir -p "$3"; cp -r "$FAKE_ROOT/venv-bin" "$3/bin"; exit 0\n'
      "fi\n"
      f'exec {real_py} "$@"\n',
  )
  script = cloud.STARTUP_SCRIPT
  assert "W=/opt/run\n" in script
  # The fake VM's directory, and fast timings (each replaced where present).
  for old, new in (
      ("W=/opt/run\n", f"W={root}/run\n"),
      ("sleep 300", "sleep 0.2"),
      ("sleep 15", "sleep 0"),
  ):
    script = script.replace(old, new)
  (root / "startup.sh").write_text(script)
  env = {
      **os.environ,
      "PATH": f"{bin_dir}:{os.environ['PATH']}",
      "FAKE_ROOT": str(root),
  }
  with open(root / "out.txt", "w", encoding="utf-8") as out:
    proc = subprocess.Popen(
        ["bash", str(root / "startup.sh")],
        cwd=root,
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
      rc = proc.wait(timeout=120)
      deadline = time.monotonic() + 20  # let an upload still in flight land
      while time.monotonic() < deadline and any((root / "inflight").glob("*")):
        time.sleep(0.05)
    finally:
      try:
        os.killpg(proc.pid, signal.SIGKILL)  # whatever the script left running
      except ProcessLookupError:
        pass  # nothing was left

  def read(p):
    """The file's stripped text, or None when it does not exist.

    Args:
      p: the file.

    Returns:
      the text, or None.
    """
    return p.read_text().strip() if p.exists() else None

  log = (
      [json.loads(l) for l in (root / "log").read_text().splitlines()]
      if (root / "log").exists()
      else []
  )
  return VmRun(
      rc=rc,
      out=(root / "out.txt").read_text(),
      log=log,
      exit_code=read(run / "attempts" / "a2" / "exit_code"),
      remote=read(run / "ckpt" / "ckpt.txt"),
      local=read(root / "run" / "ckpt" / "ckpt.txt"),
      trainer_saw=read(root / "trainer_saw.txt"),
  )


def _ops(vm):
  """The operations the fake VM logged.

  Args:
    vm: The fake VM.

  Returns:
    The operation names.
  """
  return [e[1] for e in vm.log]


@pytest.mark.parametrize("remote", [None, "ckpt-a"])
def test_vm_trains_from_scratch_or_resumes_and_deletes_itself(tmp_path, remote):
  """The VM trains from scratch or resumes, then deletes itself."""
  vm = run_vm(tmp_path, remote_ckpt=remote, trainer=((0, "ckpt-b"),))
  assert vm.rc == 0, vm.out
  assert vm.trainer_saw == (remote or "")  # no checkpoint yet is fine
  assert (vm.remote, vm.exit_code) == ("ckpt-b", "0")
  assert "delete" in _ops(vm)


@pytest.mark.parametrize("op", ["rsync-down", "ls"])
def test_vm_never_trains_fresh_when_the_restore_fails(tmp_path, op):
  """The VM never trains fresh when the restore fails.

  A checkpoint exists but cannot be pulled (permissions, an outage): the
  attempt fails before training and leaves the run's checkpoint alone.

  Args:
    tmp_path: pytest's per-test directory.
    op: the restore step that fails (parametrized).
  """
  vm = run_vm(
      tmp_path,
      remote_ckpt="ckpt-a",
      fail=[(op, "/ckpt")],
      trainer=((0, "ckpt-b"),),
  )
  assert vm.trainer_saw is None, "trained without its checkpoint"
  assert vm.rc == 26 and vm.exit_code == "26", vm.out
  assert vm.remote == "ckpt-a"
  assert "rsync-up" not in _ops(vm)


def test_vm_keeps_itself_when_the_final_upload_fails(tmp_path):
  """The VM keeps itself when the final upload fails."""
  vm = run_vm(
      tmp_path,
      fail=[("rsync-up", "/ckpt")],
      trainer=((0, "ckpt-b"), (0.1, "ckpt-c")),
  )
  assert not vm.trainer_saw
  assert (vm.rc, vm.exit_code) == (30, "storage-error"), vm.out
  assert "delete" not in _ops(vm) and "poweroff" not in _ops(vm)
  assert vm.local == "ckpt-c"  # the only current copy is still on the kept VM
  assert "keeping this VM" in vm.out


def test_a_stale_background_sync_never_lands_after_the_final_upload(tmp_path):
  """A stale background sync never lands after the final upload.

  A background sync that began before training ended (and so carries an
  older snapshot) is still uploading when the trainer exits: the final upload
  must land after it, and the exit marker after both.

  Args:
    tmp_path: pytest's per-test directory.
  """
  vm = run_vm(tmp_path, trainer=((0, "ckpt-b"), (1.0, "ckpt-c")), delays=[2.5])
  assert vm.rc == 0, vm.out
  marker = next(
      i
      for i, e in enumerate(vm.log)
      if e[1] == "cp" and e[2].endswith("/exit_code")
  )
  uploads = [i for i, e in enumerate(vm.log) if e[1:3] == ["rsync-up", "done"]]
  assert len(uploads) >= 2  # the slow background one, then the final one
  assert max(uploads) < marker, "an upload landed after the exit marker"
  assert vm.log[max(uploads)][3] == "ckpt-c" and vm.remote == "ckpt-c"
  assert vm.exit_code == "0"


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

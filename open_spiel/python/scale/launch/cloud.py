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

r"""Train on a Google Cloud VM in your own project, and take it down again.

  python -m open_spiel.python.scale.launch.cloud \
      --project=my-project --bucket=my-bucket --dry_run -- \
      --config=ppo_tic_tac_toe
  python -m open_spiel.python.scale.launch.cloud \
      --project=my-project --bucket=my-bucket -- --config=ppo_tic_tac_toe
  python -m open_spiel.python.scale.launch.cloud --project=my-project \
      --down=<run_id>
  python -m open_spiel.python.scale.launch.cloud --project=my-project --list

Everything after `--` is passed to the trainer
(`python -m open_spiel.python.scale.train`); the launcher adds
`--checkpoint_dir`. Settings come from flags, or from a YAML file
(`--config_file`, see cloud_example.yaml beside this file) whose keys are
the flag names and whose values may read the environment as ${VAR}; a flag
given on the command line wins over the file.

WHAT YOU NEED. The gcloud CLI, logged in (`gcloud auth login`) with
application default credentials (`gcloud auth application-default login`),
a project with the Compute Engine API enabled, a bucket you can write, and
quota for the machine. Your account needs roles/compute.instanceAdmin.v1,
roles/iam.serviceAccountUser on the VM's service account and
roles/storage.objectAdmin on the bucket. The VM's service account (the
Compute Engine default unless --service_account names one) needs
roles/storage.objectAdmin on the bucket and roles/compute.instanceAdmin.v1
to delete its own VM at the end. Before anything is created the launcher
checks each of these it can see and prints the command that fixes what is
missing. It never creates or deletes a bucket.

WHAT IT CREATES. One VM per run, named after the run and labelled
`<label key>=<run_id>` (LABEL_KEY), SPOT by default, with a maximum run
duration after which Compute Engine deletes it (--max_run_hours). Objects
under gs://<bucket>/<prefix>/<run_id>/: code.tgz (this package and, when it
is importable here, the JAX games package, plus a requirements.txt pinned to
the versions installed here; a run whose game needs the games package
when it is not importable here is refused before anything is created),
ckpt/ (checkpoints and metrics, synced every few minutes and at the end;
one lineage per run, shared by every launch of it), and for each launch its
own attempts/<attempt_id>/ with train.log, startup.log and exit_code. The
launcher waits only for its own attempt's exit_code: a marker left by an
earlier launch of the same run never ends or takes down a new one.

ON THE VM. The startup script installs Python, the pinned requirements
(OpenSpiel from its released wheel), unpacks the packages into the venv's
site-packages, pulls any earlier checkpoint of the run (a preempted SPOT VM
is resumed by launching the same --run_id again), trains, uploads the
results and deletes its own VM. A run with no checkpoint yet starts from
scratch; a checkpoint that exists but cannot be pulled ends the attempt with
exit code 26 before training. The VM deletes itself only once the final
checkpoint upload has succeeded: if it fails (after retries) the attempt's
exit_code reads `storage-error`, the launcher exits STORAGE_ERROR (30) and
the VM is kept, its checkpoint in /opt/run/ckpt, until you copy it (or
rerun the upload on it) and take it down with --down; Compute Engine still
deletes it at the maximum run duration. The trainer's plugin modules (--plugins,
default $OPEN_SPIEL_SCALE_PLUGINS) are passed on to it in that variable; a
plugin outside the shipped packages must be installed by --requirements.

WHAT IT DELETES. Only VMs carrying its label: the run's own VM when the run
ends with its checkpoint uploaded (the VM deletes itself; the launcher, while
it waits, deletes it too),
Compute Engine at the maximum run duration, and `--down <run_id>`. Never a
bucket and never a bucket object.

YOUR OWN VM POLICY. --create_command names a program run instead of
`gcloud compute instances create`, with the same arguments (the VM name,
then the gcloud flags): an organisation that wraps VM creation (labels,
network rules) points it at its wrapper. The wrapper's stderr is read as
gcloud's is, so a capacity error in it still moves on to the next zone.

COST. A SPOT e2-standard-8 costs a small fraction of its on-demand price;
see https://cloud.google.com/compute/vm-instance-pricing. New projects have
no GPU quota: request it before using --accelerator.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime
import importlib.metadata
import importlib.util
import io
import json
import os
import pathlib
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from open_spiel.python.scale import deps
from open_spiel.python.scale import plugins
from open_spiel.python.scale.launch import preflight as preflight_lib
from open_spiel.python.scale.launch import subprocess_env

# This package (the framework) and the JAX games package it trains on, which
# is optional: both are shipped to the VM when importable here. Derived from
# the package name (subprocess_env.PACKAGE, from that module's own __name__,
# which also holds when this module runs as __main__), so the launcher
# follows the package wherever it is installed.
FRAMEWORK = subprocess_env.PACKAGE
GAMES = deps.GAMES_PACKAGE
TRAIN_MODULE = f"{FRAMEWORK}.train"
SHORT = FRAMEWORK.rsplit(".", 1)[-1].replace("_", "-")
LABEL_KEY = f"{SHORT}-run"
CREATED_BY = f"{SHORT}-launcher"
PREFIX = f"{SHORT}-runs"
EXAMPLE_CONFIG = pathlib.Path(__file__).with_name("cloud_example.yaml")

# What the VM installs, pinned to the version installed here when there is
# one (the run uses what you tested locally).
VM_DISTRIBUTIONS = (
    "open-spiel",
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "chex",
    "mctx",
    "orbax-checkpoint",
    "etils",
    "PyYAML",
    "ml-collections",
    "ml-dtypes",
    "numpy",
    "scipy",
    "absl-py",
    "distrax",
)
DEFAULTS = {
    "region": "us-central1",
    "zone": "auto",
    "machine_type": "e2-standard-8",
    "accelerator": "none",
    "spot": True,
    "max_run_hours": 2.0,
    "service_account": "default",
    "network": "default",
    "external_ip": True,
    "image_family": "ubuntu-2404-lts-amd64",
    "image_project": "ubuntu-os-cloud",
    "boot_disk_gb": 50,
    "run_id": "auto",
    "create_command": "gcloud",
}
SETTINGS = ("project", "bucket", "plugins", *DEFAULTS)
# Substrings of a failed create's stderr that mean "no capacity or quota in
# this zone now": the next zone is tried.
CAPACITY_ERRORS = (
    "ZONE_RESOURCE_POOL_EXHAUSTED",
    "does not have enough resources",
    "QUOTA_EXCEEDED",
    "Quota '",
    "quota exceeded",
    "STOCKOUT",
    "currently unavailable in the",
    "is not available in zone",
)
RUN_ID_RE = re.compile(r"[a-z]([-a-z0-9]{0,61}[a-z0-9])?")
BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]")
POLL_S = 30
# The VM's exit_code when its final checkpoint upload failed (the VM is then
# kept), and the launcher's exit code for it.
STORAGE_ERROR = "storage-error"
STORAGE_ERROR_RC = 30


class LaunchError(Exception):
  """A refused launch; the message says why and what to do."""


@dataclasses.dataclass(frozen=True)
class Problem:
  """A missing prerequisite and the command that fixes it.

  Attributes:
    what: the missing prerequisite.
    fix: the command that fixes it.
  """

  what: str
  fix: str

  def __str__(self) -> str:
    return f"  - {self.what}\n    fix: {self.fix}"


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def run_command(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
  """Runs argv, capturing its output; never raises on failure.

  Args:
    argv: the command.

  Returns:
    its CompletedProcess (returncode 127 when it cannot be started).
  """
  try:
    return subprocess.run(
        list(argv), capture_output=True, text=True, check=False
    )
  except OSError as e:
    return subprocess.CompletedProcess(
        list(argv), 127, "", f"cannot run {argv[0]}: {e}"
    )


def run_gcloud(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
  """Runs `gcloud args...`, capturing its output; never raises on failure.

  Args:
    args: gcloud's arguments.

  Returns:
    its CompletedProcess.
  """
  return subprocess.run(
      ["gcloud", *args], capture_output=True, text=True, check=False
  )


# -- settings ---------------------------------------------------------------


def _expand(value: Any, where: str) -> Any:
  """Expands the ${VAR} references in a string from the environment.

  Args:
    value: the value; a non-string is returned unchanged.
    where: where the value came from, for the error message.

  Returns:
    the expanded value.

  Raises:
    LaunchError: a referenced variable is not set.
  """
  if not isinstance(value, str):
    return value

  def sub(m: re.Match[str]) -> str:
    """The environment's value of the ${VAR} that m matched.

    Args:
      m: the match.

    Returns:
      the value.

    Raises:
      LaunchError: the variable is not set.
    """
    name = m.group(1)
    if name not in os.environ:
      raise LaunchError(f"{where}: ${{{name}}} is not set in the environment")
    return os.environ[name]

  return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", sub, value)


def load_config_file(path: str) -> dict[str, Any]:
  """The YAML file's settings, ${VAR} expanded; unknown keys are refused.

  Args:
    path: the YAML file.

  Returns:
    the settings.

  Raises:
    LaunchError: the file is not a mapping, or has an unknown key.
  """
  # Lazy: an optional dependency (PyYAML reads --config_file; the launcher
  # host needs it only for one).
  import yaml  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  with open(path, encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
  if not isinstance(data, dict):
    raise LaunchError(f"{path}: expected a mapping of settings")
  unknown = sorted(set(data) - set(SETTINGS) - {"train_args"})
  if unknown:
    raise LaunchError(
        f"{path}: unknown settings {unknown}; known: {sorted(SETTINGS)}"
    )
  return {k: _expand(v, f"{path}:{k}") for k, v in data.items()}


@dataclasses.dataclass(frozen=True)
class Settings:
  """The launcher's resolved settings, one per flag.

  Attributes:
    project: the GCP project.
    bucket: the bucket name.
    region: the region.
    zone: a zone, or auto: every zone of the region in turn.
    machine_type: the VM's machine type.
    accelerator: none, or TYPE[:COUNT].
    spot: whether the VM is a Spot VM.
    max_run_hours: the VM's maximum run duration.
    service_account: default, or a service account email.
    network: the VPC network.
    external_ip: whether the VM gets an external IP.
    image_family: the boot image family.
    image_project: the boot image project.
    boot_disk_gb: the boot disk size.
    run_id: the run's name.
    create_command: gcloud, or a program run instead of `gcloud compute
        instances create`.
    plugins: the trainer's plugin modules.
  """

  project: str
  bucket: str
  region: str
  zone: str
  machine_type: str
  accelerator: str
  spot: bool
  max_run_hours: float
  service_account: str
  network: str
  external_ip: bool
  image_family: str
  image_project: str
  boot_disk_gb: int
  run_id: str
  create_command: str = "gcloud"
  plugins: tuple[str, ...] = ()

  @property
  def run_uri(self) -> str:
    """The run's root in the bucket, gs://<bucket>/<PREFIX>/<run_id>."""
    return f"gs://{self.bucket}/{PREFIX}/{self.run_id}"

  def attempt_uri(self, attempt_id: str) -> str:
    """One launch's logs and exit_code (the checkpoints stay at run_uri).

    Args:
      attempt_id: the launch's attempt id.

    Returns:
      <run_uri>/attempts/<attempt_id>.
    """
    return f"{self.run_uri}/attempts/{attempt_id}"


def new_attempt_id() -> str:
  """A fresh identity for one launch of a run: UTC time and a random tag.

  Returns:
    the attempt id.
  """
  stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
  return f"{stamp}-{secrets.token_hex(4)}"


def resolve_settings(
    cli: dict[str, Any],
    file: dict[str, Any],
    need_bucket: bool = True,
    environ: Optional[dict[str, str]] = None,
) -> Settings:
  """Flags over the file over DEFAULTS, validated.

  The trainer's plugins: --plugins, else the file's, else plugins.ENV of this
  environment.

  Args:
    cli: the flags given on the command line.
    file: the config file's settings.
    need_bucket: whether a bucket is required.
    environ: the environment (default os.environ).

  Returns:
    the Settings.

  Raises:
    LaunchError: a setting is missing or invalid.
  """
  merged = dict(DEFAULTS)
  merged.update({k: v for k, v in file.items() if k in SETTINGS})
  merged.update(
      {k: v for k, v in cli.items() if v is not None and k in SETTINGS}
  )
  if not merged.get("project"):
    raise LaunchError("--project is required (your GCP project id)")
  bucket = str(merged.get("bucket") or "")
  bucket = bucket[len("gs://") :] if bucket.startswith("gs://") else bucket
  bucket = bucket.rstrip("/")
  if need_bucket and not BUCKET_RE.fullmatch(bucket):
    raise LaunchError(
        "--bucket is required: a bucket name you can write (e.g. my-bucket)"
    )
  run_id = str(merged["run_id"])
  if run_id == "auto":
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%d-%H%M%S"
    )
    run_id = f"{SHORT}-{stamp}-{secrets.token_hex(2)}"
  if not RUN_ID_RE.fullmatch(run_id):
    raise LaunchError(
        f"--run_id {run_id!r}: lower-case letters, digits and '-', starting "
        "with"
        " a letter, at most 63 characters (it names the VM)"
    )
  if float(merged["max_run_hours"]) <= 0:
    raise LaunchError("--max_run_hours must be positive")
  return Settings(
      project=str(merged["project"]),
      bucket=bucket,
      region=str(merged["region"]),
      zone=str(merged["zone"]),
      machine_type=str(merged["machine_type"]),
      accelerator=str(merged["accelerator"]),
      spot=_bool(merged["spot"]),
      max_run_hours=float(merged["max_run_hours"]),
      service_account=str(merged["service_account"]),
      network=str(merged["network"]),
      external_ip=_bool(merged["external_ip"]),
      image_family=str(merged["image_family"]),
      image_project=str(merged["image_project"]),
      boot_disk_gb=int(merged["boot_disk_gb"]),
      run_id=run_id,
      create_command=str(merged["create_command"]).strip() or "gcloud",
      plugins=_plugins(merged, os.environ if environ is None else environ),
  )


def _plugins(
    merged: dict[str, Any], environ: Mapping[str, str]
) -> tuple[str, ...]:
  """The plugin modules: the config's `plugins`, else `plugins.ENV`.

  Args:
    merged: the merged launch config.
    environ: the environment.

  Returns:
    the dotted module names.

  Raises:
    LaunchError: a name is not a dotted module name.
  """
  value = merged.get("plugins")
  if value is None:
    value = environ.get(plugins.ENV, "")
  if isinstance(value, (list, tuple)):
    value = ",".join(str(v) for v in value)
  mods = tuple(m.strip() for m in str(value).split(",") if m.strip())
  bad = [
      m for m in mods if not all(part.isidentifier() for part in m.split("."))
  ]
  if bad:
    raise LaunchError(f"--plugins: {bad} are not dotted module names")
  return mods


def _bool(v: Any) -> bool:
  """A flag value as a bool.

  Args:
    v: The value.

  Returns:
    The bool.

  Raises:
    LaunchError: The value is not a boolean.
  """
  if isinstance(v, bool):
    return v
  s = str(v).strip().lower()
  if s in ("1", "true", "yes", "on"):
    return True
  if s in ("0", "false", "no", "off"):
    return False
  raise LaunchError(f"not a boolean: {v!r}")


# -- preflight --------------------------------------------------------------


def _accelerator(spec: str) -> Optional[tuple[str, int]]:
  """Parses --accelerator TYPE[:COUNT].

  Args:
    spec: The flag value.

  Returns:
    (type, count), or None for none.

  Raises:
    LaunchError: The value is malformed.
  """
  if spec in ("", "none", None):
    return None
  m = re.fullmatch(r"([a-z0-9-]+)(?::(\d+))?", spec)
  if not m:
    raise LaunchError(
        f"--accelerator {spec!r}: want TYPE[:COUNT], e.g. nvidia-tesla-t4:1"
    )
  return m.group(1), int(m.group(2) or 1)


def cpu_quota_metric(machine_type: str) -> Optional[str]:
  """The regional vCPU quota a machine family draws on; None if unknown.

  Args:
    machine_type: the machine type.

  Returns:
    the quota metric, or None.
  """
  family = machine_type.split("-", 1)[0]
  return {
      "e2": "CPUS",
      "n1": "CPUS",
      "n2": "N2_CPUS",
      "n2d": "N2D_CPUS",
      "c2": "C2_CPUS",
      "c2d": "C2D_CPUS",
      "c3": "C3_CPUS",
      "n4": "N4_CPUS",
      "t2d": "T2D_CPUS",
  }.get(family)


def gpu_quota_metric(accel_type: str) -> str:
  """The regional quota metric of a GPU type, e.g. NVIDIA_T4_GPUS.

  Args:
    accel_type: the accelerator type.

  Returns:
    the metric.
  """
  name = re.sub(r"^nvidia-(tesla-)?", "", accel_type).replace("-", "_").upper()
  return f"NVIDIA_{name}_GPUS"


def _vcpus(machine_type: str) -> Optional[int]:
  """The vCPUs of a machine type, from its suffix.

  Args:
    machine_type: The machine type.

  Returns:
    The count, or None.
  """
  m = re.search(r"-(\d+)$", machine_type)
  return int(m.group(1)) if m else None


def preflight(
    s: Settings,
    gcloud: Runner = run_gcloud,
    which: Optional[Callable[[str], Optional[str]]] = None,
    need_bucket: bool = True,
) -> list[Problem]:
  """Every missing prerequisite the launcher can see, with its fix.

  Reads only; creates nothing.

  Args:
    s: the settings.
    gcloud: runs gcloud.
    which: finds a program on PATH; default: shutil.which, looked up at call
      time (so a patch of shutil.which takes effect).
    need_bucket: whether the bucket is checked.

  Returns:
    the problems (empty when every check passes).
  """
  if which is None:
    which = shutil.which
  p = s.project
  if which("gcloud") is None:
    return [
        Problem(
            "the gcloud CLI is not on PATH",
            "install it: https://cloud.google.com/sdk/docs/install",
        )
    ]
  out = []
  r = gcloud(
      ["auth", "list", "--filter=status:ACTIVE", "--format=value(account)"]
  )
  if r.returncode != 0 or not r.stdout.strip():
    out.append(Problem("gcloud has no active account", "gcloud auth login"))
  r = gcloud(["auth", "application-default", "print-access-token"])
  if r.returncode != 0:
    out.append(
        Problem(
            "no application default credentials",
            "gcloud auth application-default login",
        )
    )
  r = gcloud(["projects", "describe", p, "--format=value(projectId)"])
  if r.returncode != 0:
    out.append(
        Problem(
            f"project {p!r} cannot be read with this account",
            "check --project against: gcloud projects list",
        )
    )
    return out
  r = gcloud([
      "services",
      "list",
      "--enabled",
      f"--project={p}",
      "--filter=config.name=compute.googleapis.com",
      "--format=value(config.name)",
  ])
  if r.returncode != 0 or "compute.googleapis.com" not in r.stdout:
    out.append(
        Problem(
            "the Compute Engine API is not enabled",
            f"gcloud services enable compute.googleapis.com --project={p}",
        )
    )
    return out
  if need_bucket:
    r = gcloud([
        "storage",
        "buckets",
        "describe",
        f"gs://{s.bucket}",
        "--format=value(name)",
    ])
    if r.returncode != 0:
      out.append(
          Problem(
              f"bucket gs://{s.bucket} does not exist or you cannot read it"
              " (the launcher never creates one)",
              f"gcloud storage buckets create gs://{s.bucket} --project={p}"
              f" --location={s.region} --uniform-bucket-level-access",
          )
      )
  r = gcloud([
      "compute",
      "regions",
      "describe",
      s.region,
      f"--project={p}",
      "--format=json",
  ])
  if r.returncode != 0:
    out.append(
        Problem(
            f"region {s.region!r} cannot be read",
            f"gcloud compute regions list --project={p}",
        )
    )
    return out
  quotas = {q["metric"]: q for q in json.loads(r.stdout).get("quotas", [])}
  quota_url = f"https://console.cloud.google.com/iam-admin/quotas?project={p}"

  def short(metric: str, need: float) -> None:
    for m in (f"PREEMPTIBLE_{metric}", metric) if s.spot else (metric,):
      q = quotas.get(m)
      if q is not None and float(q["limit"]) > 0:
        if float(q["limit"]) - float(q["usage"]) < need:
          out.append(
              Problem(
                  f"quota {m} in {s.region}: {q['usage']:.0f} of "
                  f"{q['limit']:.0f} used,"
                  f" {need:.0f} needed",
                  f"request more at {quota_url}",
              )
          )
        return
    out.append(
        Problem(
            f"no {metric} quota in {s.region}",
            f"request it at {quota_url} (metric {metric})",
        )
    )

  metric, cpus = cpu_quota_metric(s.machine_type), _vcpus(s.machine_type)
  if metric and cpus:
    short(metric, cpus)
  acc = _accelerator(s.accelerator)
  if acc:
    short(gpu_quota_metric(acc[0]), acc[1])
  return out


# -- the run's VM -----------------------------------------------------------


def zones(s: Settings, gcloud: Runner = run_gcloud) -> list[str]:
  """The zones to try in order: --zone, or every zone of the region.

  Args:
    s: the settings.
    gcloud: runs gcloud.

  Returns:
    the zones.
  """
  if s.zone != "auto":
    return [s.zone]
  r = gcloud([
      "compute",
      "zones",
      "list",
      f"--project={s.project}",
      f"--filter=region:{s.region} AND status:UP",
      "--format=value(name)",
  ])
  found = sorted(r.stdout.split()) if r.returncode == 0 else []
  return found or [f"{s.region}-{z}" for z in "abcf"]


def requirements(accelerator: bool) -> list[str]:
  """The VM's pip requirements: each distribution at its local version.

  Args:
    accelerator: whether the VM has a GPU.

  Returns:
    the requirement lines.
  """
  out = []
  for dist in VM_DISTRIBUTIONS:
    try:
      v = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
      if dist == "open-spiel":
        out.append(dist)
      continue
    name = "jax[cuda12]" if accelerator and dist == "jax" else dist
    if accelerator and dist == "jaxlib":
      continue  # jax[cuda12] brings its own
    out.append(f"{name}=={v}")
  return out


def package_dir(name: str) -> pathlib.Path:
  """The directory of the importable package `name`.

  Args:
    name: the package's import name.

  Returns:
    its directory.

  Raises:
    LaunchError: the package is not importable here.
  """
  spec = importlib.util.find_spec(name)
  if spec is None or not spec.submodule_search_locations:
    raise LaunchError(f"package {name} is not importable here")
  return pathlib.Path(list(spec.submodule_search_locations)[0])


def shipped_packages() -> tuple[str, ...]:
  """The code packages shipped to the VM.

  The code packages shipped to the VM: the framework, and the games package
  when it is importable here (only the configs on its games need it).

  Returns:
    the import names.
  """
  try:
    games = importlib.util.find_spec(GAMES) is not None
  except ModuleNotFoundError:  # a parent package of GAMES is missing
    games = False
  return (FRAMEWORK, GAMES) if games else (FRAMEWORK,)


def code_bundle(reqs: Sequence[str]) -> bytes:
  """The bytes of code.tgz.

  pkgs/<package path>/... for each shipped package, and requirements.txt.

  Args:
    reqs: the requirement lines.

  Returns:
    the bundle's bytes.
  """
  buf = io.BytesIO()
  with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for name in shipped_packages():
      root = package_dir(name)
      arc = "pkgs/" + name.replace(".", "/")
      for path in sorted(root.rglob("*")):
        if (
            "__pycache__" in path.parts
            or path.suffix == ".pyc"
            or not path.is_file()
        ):
          continue
        tar.add(path, arcname=f"{arc}/{path.relative_to(root).as_posix()}")
    data = ("\n".join(reqs) + "\n").encode()
    info = tarfile.TarInfo("requirements.txt")
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
  return buf.getvalue()


STARTUP_SCRIPT = r"""#!/bin/bash
# Startup script of one training run (written by the launcher).
set -uo pipefail
MD=http://metadata.google.internal/computeMetadata/v1/instance
attr() { curl -sf -H Metadata-Flavor:Google "$MD/attributes/$1"; }
RUN_URI="$(attr run-uri)"
MODULE="$(attr train-module)"
# This launch's own directory: its logs and exit_code. The checkpoints stay
# under the run ($RUN_URI/ckpt), shared by every launch of it.
ATTEMPT="$(attr attempt-id)" || ATTEMPT=missing-attempt-id
ATTEMPT_URI="$RUN_URI/attempts/$ATTEMPT"
W=/opt/run
mkdir -p "$W/ckpt"
cd "$W"
exec > >(tee -a "$W/startup.log") 2>&1
TRIES=4
# retry CMD...: runs CMD up to $TRIES times, a pause between tries.
retry() {
  local i
  for i in $(seq 1 "$TRIES"); do
    "$@" && return 0
    echo "failed (try $i/$TRIES): $*"
    [ "$i" -lt "$TRIES" ] && sleep 15
  done
  return 1
}
# The background checkpoint sync. TERM ends it between uploads only: bash runs
# the trap once a running rsync has returned, so stop_sync, which waits for it,
# leaves no stale upload that could land after the final one.
sync_loop() {
  trap 'exit 0' TERM
  while :; do
    sleep 300 & wait $!
    gcloud storage rsync -r "$W/ckpt" "$RUN_URI/ckpt" >/dev/null 2>&1
  done
}
SYNC_PID=
stop_sync() {
  [ -n "$SYNC_PID" ] || return 0
  kill -TERM "$SYNC_PID" 2>/dev/null
  wait "$SYNC_PID" 2>/dev/null
  SYNC_PID=
}
# Pulls the run's checkpoint. No checkpoint yet is fine (a new run); one that
# exists but cannot be listed or pulled fails, so a resume never silently
# trains from scratch.
restore() {
  local out
  if ! out="$(gcloud storage ls "$RUN_URI/ckpt/**" 2>&1)"; then
    case "$out" in
      *"matched no objects"*)
        echo "no checkpoint under $RUN_URI/ckpt yet"; return 0 ;;
    esac
    echo "listing $RUN_URI/ckpt failed: $out"
    return 1
  fi
  [ -n "$out" ] || { echo "no checkpoint under $RUN_URI/ckpt yet"; return 0; }
  echo "resuming from $RUN_URI/ckpt"
  retry gcloud storage rsync -r "$RUN_URI/ckpt" "$W/ckpt"
}
PUBLISH=0  # set once training starts: from then on $W/ckpt holds new state
finish() {
  local rc="$1" status="$1"
  stop_sync
  if [ "$PUBLISH" = 1 ] &&
      ! retry gcloud storage rsync -r "$W/ckpt" "$RUN_URI/ckpt"; then
    status=STORAGE_ERROR_STATUS
  fi
  gcloud storage cp "$W/startup.log" "$ATTEMPT_URI/startup.log" || true
  [ -f "$W/train.log" ] &&
    gcloud storage cp "$W/train.log" "$ATTEMPT_URI/train.log"
  echo "$status" > "$W/exit_code"
  retry gcloud storage cp "$W/exit_code" "$ATTEMPT_URI/exit_code"
  if [ "$status" = STORAGE_ERROR_STATUS ]; then
    echo "run $RUN_URI: the final checkpoint upload failed" \
      "(trainer exit code $rc);"
    echo "keeping this VM: its checkpoint is in $W/ckpt"
    exit STORAGE_ERROR_RC
  fi
  NAME="$(curl -sf -H Metadata-Flavor:Google "$MD/name")"
  ZONE="$(curl -sf -H Metadata-Flavor:Google "$MD/zone" |
    awk -F/ '{print $NF}')"
  gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || poweroff
  exit "$rc"
}
echo "run $RUN_URI: attempt $ATTEMPT: setting up"
gcloud storage cp "$RUN_URI/code.tgz" code.tgz || finish 20
tar -xzf code.tgz || finish 21
export DEBIAN_FRONTEND=noninteractive
# The image's own boot-time apt (or an organisation's VM-create preamble) may
# hold the package-list lock, which DPkg::Lock::Timeout does not wait for:
# retry for up to 15 minutes.
apt_retry() {
  local i
  for i in $(seq 1 90); do
    "$@" && return 0
    echo "apt busy (attempt $i/90): $*"; sleep 10
  done
  return 1
}
apt_retry apt-get -o DPkg::Lock::Timeout=60 update -qq || finish 22
apt_retry apt-get -o DPkg::Lock::Timeout=60 install -yqq python3-venv \
  || finish 22
python3 -m venv venv || finish 23
venv/bin/pip install -q --upgrade pip || finish 24
venv/bin/pip install -q -r requirements.txt || finish 24
SITE="$(venv/bin/python -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# Overlay the shipped packages onto the installed ones (a parent package the
# wheel lacks gets an empty __init__.py).
(cd pkgs && find . -type f) | while read -r f; do
  mkdir -p "$SITE/$(dirname "$f")"; cp "pkgs/$f" "$SITE/$f"
done
(cd pkgs && find . -mindepth 1 -type d) | while read -r d; do
  p="$(dirname "$d")"
  while [ "$p" != "." ]; do
    [ -f "$SITE/$p/__init__.py" ] || touch "$SITE/$p/__init__.py"
    p="$(dirname "$p")"
  done
done
ENV_EXPORTS="$(attr train-env | base64 -d | venv/bin/python -c '
import json, shlex, sys
for k, v in json.load(sys.stdin).items():
  print("export %s=%s" % (k, shlex.quote(v)))
')" || finish 25
eval "$ENV_EXPORTS"
restore || finish 26
sync_loop &
SYNC_PID=$!
PUBLISH=1
mapfile -t ARGS < <(attr train-args | base64 -d |
  venv/bin/python -c 'import json,sys; print("\n".join(json.load(sys.stdin)))')
echo "run $RUN_URI: training: python -m $MODULE ${ARGS[*]}"
venv/bin/python -u -m "$MODULE" "${ARGS[@]}" --checkpoint_dir="$W/ckpt" \
  > "$W/train.log" 2>&1
finish $?
""".replace("STORAGE_ERROR_STATUS", STORAGE_ERROR).replace(
    "STORAGE_ERROR_RC", str(STORAGE_ERROR_RC)
)


def train_env(plugin_modules: Sequence[str]) -> dict[str, str]:
  """The trainer's extra environment on the VM: plugins.ENV when set.

  Args:
    plugin_modules: the trainer's plugin modules.

  Returns:
    the environment.
  """
  return {plugins.ENV: ",".join(plugin_modules)} if plugin_modules else {}


def _b64json(value: Any) -> str:
  """`value` as base64 JSON.

  Args:
    value: A JSON-serialisable value.

  Returns:
    The encoded string.
  """
  return base64.b64encode(json.dumps(value).encode()).decode()


def create_args(
    s: Settings,
    zone: str,
    train_args: Sequence[str],
    script_path: str,
    attempt_id: str,
) -> list[str]:
  """`gcloud compute instances create ...` for the run's VM (one attempt).

  Args:
    s: the settings.
    zone: the zone.
    train_args: the trainer's flags.
    script_path: the startup script's local path.
    attempt_id: the launch's attempt id.

  Returns:
    the arguments after `gcloud`.

  Raises:
    LaunchError: the attempt id is not lower-case letters, digits and '-'.
  """
  if not re.fullmatch(r"[-a-z0-9]+", attempt_id):
    raise LaunchError(
        f"attempt id {attempt_id!r}: want lower-case letters, digits and '-'"
    )
  args = [
      "compute",
      "instances",
      "create",
      s.run_id,
      f"--project={s.project}",
      f"--zone={zone}",
      f"--machine-type={s.machine_type}",
      f"--image-family={s.image_family}",
      f"--image-project={s.image_project}",
      f"--boot-disk-size={s.boot_disk_gb}GB",
      f"--network={s.network}",
      f"--provisioning-model={'SPOT' if s.spot else 'STANDARD'}",
      f"--max-run-duration={int(round(s.max_run_hours * 60))}m",
      "--instance-termination-action=DELETE",
      f"--labels={LABEL_KEY}={s.run_id},created-by={CREATED_BY}",
      "--scopes=cloud-platform",
      "--metadata="
      + ",".join([
          f"run-uri={s.run_uri}",
          f"attempt-id={attempt_id}",
          f"train-module={TRAIN_MODULE}",
          "train-args=" + _b64json(list(train_args)),
          "train-env=" + _b64json(train_env(s.plugins)),
      ]),
      f"--metadata-from-file=startup-script={script_path}",
  ]
  if s.service_account != "default":
    args.append(f"--service-account={s.service_account}")
  if not s.external_ip:
    args.append("--no-address")
  acc = _accelerator(s.accelerator)
  if acc:
    args += [
        f"--accelerator=type={acc[0]},count={acc[1]}",
        "--maintenance-policy=TERMINATE",
    ]
  return args


def create_command(
    s: Settings,
    zone: str,
    train_args: Sequence[str],
    script_path: str,
    attempt_id: str,
) -> Optional[list[str]]:
  """The --create_command argv for the run's VM, or None for plain gcloud.

  Args:
    s: the settings.
    zone: the zone.
    train_args: the trainer's flags.
    script_path: the startup script's local path.
    attempt_id: the launch's attempt id.

  Returns:
    the argv, or None.
  """
  if s.create_command == "gcloud":
    return None
  return [
      *shlex.split(s.create_command),
      *create_args(s, zone, train_args, script_path, attempt_id)[3:],
  ]


def labelled_vms(
    project: str, run_id: Optional[str], gcloud: Runner = run_gcloud
) -> list[tuple[str, str]]:
  """[(name, zone)] of the VMs this launcher labelled (for run_id, or all).

  Args:
    project: the GCP project.
    run_id: the run, or None for all runs.
    gcloud: runs gcloud.

  Returns:
    the VMs.

  Raises:
    LaunchError: gcloud cannot list the VMs.
  """
  flt = f"labels.{LABEL_KEY}={run_id}" if run_id else f"labels.{LABEL_KEY}:*"
  r = gcloud([
      "compute",
      "instances",
      "list",
      f"--project={project}",
      f"--filter={flt} AND labels.created-by={CREATED_BY}",
      "--format=value(name,zone.basename(),labels)",
  ])
  if r.returncode != 0:
    raise LaunchError(f"listing VMs failed: {r.stderr.strip()}")
  out = []
  for line in r.stdout.splitlines():
    parts = line.split()
    if len(parts) >= 2:
      out.append((parts[0], parts[1]))
  return out


def down(project: str, run_id: str, gcloud: Runner = run_gcloud) -> int:
  """Deletes the run's labelled VMs; returns how many.

  Args:
    project: the GCP project.
    run_id: the run.
    gcloud: runs gcloud.

  Returns:
    the number of VMs deleted.

  Raises:
    LaunchError: run_id is not a run id, or a VM cannot be deleted.
  """
  if not RUN_ID_RE.fullmatch(run_id):
    raise LaunchError(f"--down {run_id!r} is not a run id")
  vms = labelled_vms(project, run_id, gcloud)
  for name, zone in vms:
    r = gcloud([
        "compute",
        "instances",
        "delete",
        name,
        f"--zone={zone}",
        f"--project={project}",
        "--quiet",
    ])
    gone = (
        "was not found",
        "is being deleted",
        "already being deleted",
        "resourceNotReady",
    )
    if r.returncode != 0 and not any(g in r.stderr for g in gone):
      raise LaunchError(f"deleting {name} failed: {r.stderr.strip()}")
    print(f"deleted {name} ({zone}), labelled {LABEL_KEY}={run_id}")
  if not vms:
    print(
        f"no VM labelled {LABEL_KEY}={run_id} in {project}; nothing to delete"
    )
  return len(vms)


def _gs_read(uri: str, gcloud: Runner) -> Optional[str]:
  """The text of a gs:// object.

  Args:
    uri: The gs:// URI.
    gcloud: The gcloud runner.

  Returns:
    The text, or None when the read fails.
  """
  r = gcloud(["storage", "cat", uri])
  return r.stdout if r.returncode == 0 else None


def read_requirements(path: str) -> list[str]:
  """A pip requirements file's lines, comments and blanks dropped.

  Args:
    path: the file.

  Returns:
    the requirement lines.

  Raises:
    LaunchError: the file lists no requirement.
  """
  with open(path, encoding="utf-8") as f:
    out = [l.split("#", 1)[0].strip() for l in f]
  out = [l for l in out if l]
  if not out:
    raise LaunchError(f"{path} lists no requirement")
  return out


def launch(
    s: Settings,
    train_args: Sequence[str],
    *,
    dry_run: bool,
    detach: bool,
    gcloud: Runner = run_gcloud,
    sleep: Callable[[float], None] = time.sleep,
    poll_s: float = POLL_S,
    reqs: Optional[Sequence[str]] = None,
    run: Runner = run_command,
) -> int:
  """Launches one attempt on a VM.

  Uploads the code, creates the VM (trying each zone), and unless
  detached waits for this attempt's exit code and makes sure the VM is gone
  (unless its final checkpoint upload failed: the VM is then kept and
  STORAGE_ERROR_RC returned).

  Args:
    s: the settings.
    train_args: the trainer's flags.
    dry_run: print what would run; create nothing.
    detach: return once the VM is created.
    gcloud: runs gcloud.
    sleep: the sleep between polls.
    poll_s: seconds between polls.
    reqs: the VM's pip requirements; default: requirements(), the versions
        installed here.
    run: runs --create_command when one is set.

  Returns:
    the attempt's exit code; 0 for a dry run or a detached launch;
    STORAGE_ERROR_RC when the final checkpoint upload failed.

  Raises:
    LaunchError: a VM of the run already exists, the code upload fails, or no
        zone creates the VM.
  """
  acc = _accelerator(s.accelerator) is not None
  reqs = list(reqs) if reqs else requirements(acc)
  attempt = new_attempt_id()
  attempt_uri = s.attempt_uri(attempt)
  with tempfile.TemporaryDirectory(prefix=f"{SHORT}-launch-") as tmp:
    script = pathlib.Path(tmp) / "startup.sh"
    script.write_text(STARTUP_SCRIPT)
    zs = (
        zones(s, gcloud)
        if not dry_run or shutil.which("gcloud")
        else [f"{s.region}-a"]
    )
    if dry_run:
      bundle = code_bundle(reqs)
      print(f"run {s.run_id}: DRY RUN, nothing uploaded or created")
      print(
          f"  code.tgz ({len(bundle) // 1024} KiB: "
          f"{', '.join(shipped_packages())})"
          f" -> {s.run_uri}/code.tgz"
      )
      print(f"  requirements: {' '.join(reqs)}")
      print(f"  zones, in order: {' '.join(zs)}")
      print(f"  trainer environment: {train_env(s.plugins) or 'nothing added'}")
      wrapped = create_command(s, zs[0], train_args, "<startup.sh>", attempt)
      if wrapped:
        command = shlex.join(wrapped)
      else:
        command = "gcloud " + shlex.join(
            create_args(s, zs[0], train_args, "<startup.sh>", attempt)
        )
      print("  " + command)
      print(
          f"  results: {s.run_uri}/ckpt/, and {attempt_uri}/ (train.log,"
          " startup.log, exit_code)"
      )
      print(
          f"  take it down: python -m {FRAMEWORK}.launch.cloud "
          f"--project={s.project}"
          f" --down={s.run_id}"
      )
      return 0
    if labelled_vms(s.project, s.run_id, gcloud):
      raise LaunchError(
          f"a VM labelled {LABEL_KEY}={s.run_id} already exists;"
          f" pick another --run_id or --down={s.run_id}"
      )
    bundle = pathlib.Path(tmp) / "code.tgz"
    bundle.write_bytes(code_bundle(reqs))
    r = gcloud(["storage", "cp", str(bundle), f"{s.run_uri}/code.tgz"])
    if r.returncode != 0:
      raise LaunchError(f"uploading the code failed: {r.stderr.strip()}")
    zone = None
    for z in zs:
      wrapped = create_command(s, z, train_args, str(script), attempt)
      r = (
          run(wrapped)
          if wrapped
          else gcloud(create_args(s, z, train_args, str(script), attempt))
      )
      if r.returncode == 0:
        zone = z
        break
      if not any(e in r.stderr for e in CAPACITY_ERRORS):
        raise LaunchError(f"creating the VM in {z} failed: {r.stderr.strip()}")
      print(f"no capacity in {z}; trying the next zone")
    if zone is None:
      raise LaunchError(
          f"no capacity for {s.machine_type} in {' '.join(zs)};"
          " try again later, another --region or --nospot"
      )
  print(
      f"run {s.run_id}: VM {s.run_id} created in {zone} for attempt {attempt};"
      f" checkpoints under {s.run_uri}/ckpt/, logs under {attempt_uri}/"
  )
  print(
      f"  take it down: python -m {FRAMEWORK}.launch.cloud "
      f"--project={s.project}"
      f" --down={s.run_id}"
  )
  if detach:
    return 0
  deadline = time.monotonic() + s.max_run_hours * 3600 + 900
  try:
    while time.monotonic() < deadline:
      sleep(poll_s)
      # Only this attempt's marker: one left by an earlier launch of the run
      # must never end this one or take its VM down.
      code = _gs_read(f"{attempt_uri}/exit_code", gcloud)
      if code is not None and code.strip() == STORAGE_ERROR:
        print(
            f"run {s.run_id}: the final checkpoint upload failed; the VM is"
            " kept with the latest checkpoint in /opt/run/ckpt (see"
            f" {attempt_uri}/startup.log). Copy it, e.g. gcloud compute ssh"
            f" {s.run_id} --project={s.project} -- sudo gcloud storage rsync -r"
            f" /opt/run/ckpt {s.run_uri}/ckpt, then take the VM down with"
            f" --down={s.run_id}"
        )
        return STORAGE_ERROR_RC
      if code is not None:
        # The VM deletes itself after writing its exit code; give it a few
        # polls, then delete whatever carries the run's label.
        for _ in range(10):
          if not labelled_vms(s.project, s.run_id, gcloud):
            break
          sleep(poll_s)
        down(s.project, s.run_id, gcloud)
        rc = int(code.strip() or 1)
        print(
            f"run {s.run_id}: finished with exit code {rc}; checkpoints under"
            f" {s.run_uri}/ckpt/, train.log under {attempt_uri}/"
        )
        return rc
      if not labelled_vms(s.project, s.run_id, gcloud):
        print(
            f"run {s.run_id}: the VM is gone with no exit code (a SPOT"
            " preemption or the maximum run duration); launch the same"
            f" --run_id={s.run_id} again to resume from its checkpoint"
        )
        return 3
  except KeyboardInterrupt:
    print(
        "\nstopped waiting; the VM keeps running. Take it down with "
        f"--down={s.run_id}"
    )
    return 130
  down(s.project, s.run_id, gcloud)
  return 4


def check_shipped(
    train_args: Sequence[str], plugin_modules: Sequence[str]
) -> None:
  """Refuses a run whose game needs a game package code.tgz does not ship.

  Refuses, before anything is created (and in a dry run), a run whose game
  needs a game package code.tgz does not ship (launch.preflight).

  Args:
    train_args: the trainer's flags.
    plugin_modules: the trainer's plugin modules.

  Raises:
    LaunchError: the game needs a game package that is not shipped.
  """
  try:
    preflight_lib.check_game_packages(
        train_args,
        shipped_packages(),
        plugin_modules,
        fix=(
            f"The launcher ships the JAX games package ({GAMES}) when it is "
            "importable here: install it (it comes with OpenSpiel) or put its "
            "checkout on PYTHONPATH."
        ),
    )
  except preflight_lib.GamePackageNotShippedError as e:
    raise LaunchError(f"{e}; nothing was created") from e


def parse(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
  """The launcher's flags and the trainer's flags (after `--`).

  Args:
    argv: the command line, without the program name.

  Returns:
    (the launcher's options, the trainer's flags).
  """
  argv = list(argv)
  train_args: list[str] = []
  if "--" in argv:
    i = argv.index("--")
    argv, train_args = argv[:i], argv[i + 1 :]
  ap = argparse.ArgumentParser(
      prog=f"python -m {FRAMEWORK}.launch.cloud",
      description=(__doc__ or "").split("\n\n", maxsplit=1)[0],
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="Defaults: " + ", ".join(f"{k}={v}" for k, v in DEFAULTS.items()),
  )
  ap.add_argument("--project")
  ap.add_argument("--bucket", help="bucket name (gs:// optional)")
  ap.add_argument("--region")
  ap.add_argument(
      "--zone", help="a zone, or auto: every zone of the region in turn"
  )
  ap.add_argument("--machine_type")
  ap.add_argument(
      "--accelerator", help="none, or TYPE[:COUNT] (e.g. nvidia-tesla-t4:1)"
  )
  ap.add_argument("--spot", dest="spot", action="store_true", default=None)
  ap.add_argument("--nospot", dest="spot", action="store_false")
  ap.add_argument("--max_run_hours", type=float)
  ap.add_argument(
      "--service_account", help="default, or a service account email"
  )
  ap.add_argument("--network")
  ap.add_argument(
      "--no_external_ip",
      dest="external_ip",
      action="store_false",
      default=None,
      help="no external IP (the network then needs Cloud NAT for pip)",
  )
  ap.add_argument("--image_family")
  ap.add_argument("--image_project")
  ap.add_argument("--boot_disk_gb", type=int)
  ap.add_argument(
      "--create_command",
      help=(
          "gcloud, or a program run instead of `gcloud"
          " compute instances create` with the same arguments"
      ),
  )
  ap.add_argument(
      "--plugins",
      help=(
          "the trainer's plugin modules, comma-separated"
          f" (default: ${plugins.ENV}); passed on in {plugins.ENV}"
      ),
  )
  ap.add_argument(
      "--run_id", help="auto, or a name (relaunching one resumes it)"
  )
  ap.add_argument(
      "--config_file", help="YAML settings (see cloud_example.yaml)"
  )
  ap.add_argument(
      "--requirements",
      help=(
          "a pip requirements file for the VM (default:"
          " the versions of VM_DISTRIBUTIONS installed here)"
      ),
  )
  ap.add_argument(
      "--dry_run", action="store_true", help="check and print; create nothing"
  )
  ap.add_argument("--skip_preflight", action="store_true")
  ap.add_argument(
      "--detach", action="store_true", help="return once the VM is created"
  )
  ap.add_argument(
      "--down", metavar="RUN_ID", help="delete that run's labelled VM"
  )
  ap.add_argument(
      "--list", action="store_true", help="list this launcher's VMs"
  )
  return ap.parse_args(argv), train_args


def main(
    argv: Optional[Sequence[str]] = None, gcloud: Runner = run_gcloud
) -> int:
  """The launcher's entry point.

  Args:
    argv: the command line, without the program name (default sys.argv).
    gcloud: runs gcloud.

  Returns:
    the exit code.
  """
  opt, train_args = parse(sys.argv[1:] if argv is None else argv)
  try:
    file = load_config_file(opt.config_file) if opt.config_file else {}
    train_args = list(file.get("train_args") or []) + train_args
    cli = {k: getattr(opt, k, None) for k in SETTINGS}
    if opt.down or opt.list:
      s = resolve_settings(cli, file, need_bucket=False)
      if opt.down:
        down(s.project, opt.down, gcloud)
      else:
        for name, zone in labelled_vms(s.project, None, gcloud):
          print(f"{name}\t{zone}")
      return 0
    s = resolve_settings(cli, file)
    if not any(a.startswith("--config") for a in train_args):
      raise LaunchError(
          "pass the trainer's flags after `--`, e.g. -- "
          "--config=ppo_tic_tac_toe"
      )
    check_shipped(train_args, s.plugins)
    if not opt.skip_preflight:
      problems = preflight(s, gcloud)
      if problems:
        print(f"preflight: {len(problems)} problem(s); nothing was created:")
        for p in problems:
          print(p)
        return 2
      print(
          "preflight: gcloud, credentials, Compute Engine API, bucket and "
          "quota OK"
      )
    reqs = read_requirements(opt.requirements) if opt.requirements else None
    return launch(
        s,
        train_args,
        dry_run=opt.dry_run,
        detach=opt.detach,
        gcloud=gcloud,
        reqs=reqs,
    )
  except LaunchError as e:
    print(f"launch: {e}", file=sys.stderr)
    return 2


if __name__ == "__main__":
  sys.exit(main())

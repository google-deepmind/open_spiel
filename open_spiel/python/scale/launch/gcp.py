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

r"""The reference GCP backend: one Compute Engine VM per attempt, via gcloud.

It runs preemption-safe training on spot, flex-start or standard VMs with
nothing but the gcloud CLI, and serves as the template for other platforms.

  backend: gcp          # in backend.yaml
  python -m open_spiel.python.scale.launch.run --backend_config=backend.yaml \
      --run_id=kuhn-a -- --config=mmd_kuhn_poker_jax

Configured by one file (launch.config; every field required, none
defaults to any project or resource; ${VAR} reads the environment). A
complete example (GPU machines: SPOT, FLEX_START and STANDARD) is
launch/examples/gcp_gpu.yaml; launch/examples/gcp_cpu.yaml is the same for
CPU machines. launch/tests/gcp_examples_test.py parses both and checks a dry
run; no test creates a VM. Fields:

  project, storage (gs://<bucket>[/prefix]: runs live under <storage>/<run_id>),
  image {family | name, project}, network {network, subnetwork, external_ip},
  service_account {email, scopes}, labels (added to every VM; may be {}),
  code {root, paths} (shipped to the VM; paths relative to root, e.g. the
  package folder and the JAX games package folder: ${PACKAGE_ROOT},
  ${PACKAGE_PATH} and ${GAMES_PACKAGE_PATH}, launch.config; launch.run
  refuses a run whose game needs a game package not shipped), setup
  (shell lines run on the VM before the trainer, in the unpacked code dir:
  install Python and the package's dependencies, plus gcsfs
  for the gs:// checkpoint dir and a CUDA jax on GPU machines), python (the
  interpreter setup installed into), create_lock (null, or a lock file:
  launchers on one host that share it issue one `gcloud compute instances
  create` at a time, so several supervisors starting together do not burst
  the project's API rate or CPU quota at once), machines (the ordered
  fallback menu):
    {name, machine_type, accelerator ({type, count} or null), platform
     (cpu | gpu | tpu: what JAX must see on it; the trainer refuses to run on
     anything else), zones, provisioning (SPOT | STANDARD | FLEX_START),
     boot_disk_gb, max_run_duration (null, or <N>s|m|h|d: required on
     FLEX_START, at most 7d; optional on STANDARD and SPOT, 30s to 120d)}

The attempt's machine: provision walks machines in order and each machine's
zones in order, `gcloud compute instances create` (provisioning model SPOT,
STANDARD or FLEX_START, with --max-run-duration where the machine sets
one; GCE deletes a SPOT VM when it is preempted, and any VM with a max run
duration when it reaches it, so on an on-demand machine the max run duration
is the backstop that deletes the VM if the supervisor dies; the supervisor
treats a VM deleted at its max run like a preemption, and the next attempt
resumes from the checkpoint), and moves on when GCE reports no capacity or
no quota (CAPACITY_ERRORS); any other error raises. None left: NoCapacityError,
whose message is one line per kind of machine ("no L4 capacity in
us-central1-a, us-central1-b"); the raw errors stay in the state file and the
debug log. The supervisor waits and tries again.

On the VM the startup script (startup_script, one per machine) installs
and runs train: it writes attempts/<n>/started, copies the code bundle
prepare() staged in the run's storage, runs the setup lines, starts a
preemption-notice watcher (the metadata server's instance/preempted, long
polled with its ETag: on TRUE it writes attempts/<n>/preempted and SIGTERMs
the trainer, which checkpoints and exits PREEMPTED_EXIT), runs
`<EXPECT_PLATFORM_ENV>=<platform> python -u -m <TRAINER_MODULE> ...
--checkpoint_dir=<storage>/<run_id>/ckpt`, then copies the log back to
attempts/<n>/train.log, writes attempts/<n>/exit_code and powers the VM off.
A failed step (the started marker included) writes its exit code the same
way.

poll: attempts/<n>/exit_code decides (backend.classify_exit). Without one: a
VM that runs, or is STOPPING / SUSPENDING (GCE's ~30 s preemption grace,
while the trainer writes its checkpoint and the script copies back), is
RUNNING; a VM that is gone or stopped was preempted when the notice marker
or GCE's preempted operation says so; one that never wrote its started
marker FAILED (it never ran: a broken image, setup or storage permission is
not retried as a preemption); anything else is lost (PREEMPTED: the next
attempt resumes). teardown deletes the VM. adopt re-attaches to a VM that
still exists (or whose exit code is written). Storage is launch.storage's
GcsStorage over `gcloud storage`, every call retried with backoff before it
raises.

Inside the trainer, two GCE helpers are available to a deployment:
watch_preemption (the "gce" notice source of launch.preemption, off by
default) and machine_info (a hardware.register_machine_info provider that
reads the machine type, zone and preemptible flag from the metadata server).
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence
import urllib.error
import urllib.parse
import urllib.request

from absl import logging

from open_spiel.python.scale.launch import backend as backend_lib
from open_spiel.python.scale.launch import config as config_lib
from open_spiel.python.scale.launch import hardware
from open_spiel.python.scale.launch import storage as storage_lib

PROVISIONING = ("SPOT", "STANDARD", "FLEX_START")
# GCE's bounds on --max-run-duration: 30 s to 120 days, and at most 7 days on
# a FLEX_START VM (which requires one).
MAX_RUN_MIN_S, MAX_RUN_MAX_S, FLEX_START_MAX_RUN_S = 30, 120 * 86400, 7 * 86400
_DURATION_UNIT_S = {"s": 1, "m": 60, "h": 3600, "d": 86400}
# Substrings of a failed create's stderr that mean "no capacity here now"
# (try the next zone / machine), not a configuration error.
CAPACITY_ERRORS = (
    "ZONE_RESOURCE_POOL_EXHAUSTED",
    "does not have enough resources",
    "QUOTA_EXCEEDED",
    "Quota '",
    "quota exceeded",
    "STOCKOUT",
    "resource pool exhausted",
    "currently unavailable in the",
    "is not available in zone",
)
# The CAPACITY_ERRORS that mean a quota, not a stockout.
QUOTA_ERRORS = ("QUOTA_EXCEEDED", "Quota '", "quota exceeded")
# What a user calls the GPU of an accelerator type, or of a machine family
# with GPUs built in (matched by prefix), for the no-capacity message.
GPU_NAMES = {
    "nvidia-tesla-t4": "T4",
    "nvidia-l4": "L4",
    "nvidia-tesla-v100": "V100",
    "nvidia-tesla-p100": "P100",
    "nvidia-tesla-p4": "P4",
    "nvidia-tesla-a100": "A100",
    "nvidia-a100-80gb": "A100 80GB",
    "nvidia-h100-80gb": "H100",
    "nvidia-h100-mega-80gb": "H100 Mega",
    "nvidia-h200-141gb": "H200",
    "nvidia-b200": "B200",
}
GPU_FAMILIES = (
    ("g2-", "L4"),
    ("a2-highgpu-", "A100"),
    ("a2-megagpu-", "A100"),
    ("a2-ultragpu-", "A100 80GB"),
    ("a3-highgpu-", "H100"),
    ("a3-edgegpu-", "H100"),
    ("a3-megagpu-", "H100 Mega"),
    ("a3-ultragpu-", "H200"),
    ("a4-highgpu-", "B200"),
)
PLATFORMS = ("cpu", "gpu", "tpu")
EXPECT_PLATFORM_ENV = hardware.EXPECT_PLATFORM_ENV  # the trainer checks it
#: GCE labels every attempt's VM carries: its run id and attempt number.
RUN_LABEL, ATTEMPT_LABEL = "train-run", "train-attempt"
LIVE_STATES = (
    "PROVISIONING",
    "STAGING",
    "RUNNING",
    "REPAIRING",
    "STOPPING",
    "SUSPENDING",
)
METADATA = "http://metadata.google.internal/computeMetadata/v1/instance/"
METADATA_KEYS = (
    ("machine_type", "machine-type"),
    ("zone", "zone"),
    ("preemptible", "scheduling/preemptible"),
)
PREEMPTED_URL = METADATA + "preempted"
POLL_TIMEOUT_S = 300  # the metadata server's timeout_sec for one long poll
POLL_RETRY_S = (
    5  # seconds before retrying a failed poll (both preemption watchers)
)

__all__ = (
    "GcpBackend",
    "GcpConfig",
    "Machine",
    "PreemptionWatch",
    "code_bundle",
    "gce",
    "machine_info",
    "startup_script",
    "vm_name",
    "watch_preemption",
)


# -- the trainer's side: GCE metadata ------------------------------------------
def gce(timeout: float = 2.0, opener=urllib.request.urlopen) -> dict[str, str]:
  """The instance's machine type, zone and preemptible flag.

  The instance's machine type, zone and preemptible flag from the GCE
  metadata server, or {"error": ...} when it does not answer (not on GCE).

  Args:
    timeout: seconds per metadata request.
    opener: urlopen (tests inject a fake).

  Returns:
    {machine_type, zone, preemptible}, or {"error": ...}.
  """
  out = {}
  for key, path in METADATA_KEYS:
    req = urllib.request.Request(
        METADATA + path, headers={"Metadata-Flavor": "Google"}
    )
    try:
      with opener(req, timeout=timeout) as resp:
        out[key] = resp.read().decode().strip().rsplit("/", 1)[-1]
    except (urllib.error.URLError, OSError) as e:
      return {"error": f"{path}: {type(e).__name__}: {e}"}
  return out


def machine_info() -> dict[str, dict[str, str]]:
  """{"gce": gce()}: a launch.hardware machine-info provider.

  Returns:
    {"gce": gce()}.
  """
  return {"gce": gce()}


def _get(
    opener: Callable[..., Any], url: str, timeout: float
) -> tuple[str, Optional[str]]:
  """(body, ETag header or None) of a metadata server read.

  Args:
    opener: urlopen or a stand-in.
    url: The URL.
    timeout: Seconds.

  Returns:
    (body, ETag or None).
  """
  req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
  with opener(req, timeout=timeout) as resp:
    headers = getattr(resp, "headers", None)
    etag = headers.get("ETag") if headers is not None else None
    return resp.read().decode().strip(), etag or None


class PreemptionWatch:
  """A running watch_preemption: its daemon thread and what it is doing."""

  def __init__(
      self,
      thread: threading.Thread,
      stop: threading.Event,
      status: dict[str, str],
  ):
    self._thread, self._stop, self._status = thread, stop, status

  @property
  def status(self) -> str:
    """What the watch is doing, as text."""
    return self._status["text"]

  def stop(self) -> None:
    """Asks the watch to stop after its current poll."""
    self._stop.set()

  def join(self, timeout: Optional[float] = None) -> None:
    """Waits for the watch's thread to end.

    Args:
      timeout: the most seconds to wait (None: no limit).
    """
    self._thread.join(timeout)

  def is_alive(self) -> bool:
    """Whether the watch's thread is running.

    Returns:
      True while it runs.
    """
    return self._thread.is_alive()


def watch_preemption(
    notice,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    retry_s: float = POLL_RETRY_S,
) -> PreemptionWatch:
  """Starts a daemon thread that watches for GCE preemption.

  Starts a daemon thread that sets `notice` (reason "gce") when the
  metadata server reports this VM preempted, and returns its PreemptionWatch.

  The thread reads instance/preempted once, then long-polls it
  (wait_for_change) until it reads TRUE; a failed poll is retried after
  retry_s. Each poll passes the ETag of the last value read (last_etag), so
  a change made between polls, during a retry wait included, is returned at
  once rather than at the poll's timeout. Off GCE the first read fails and
  the thread ends without setting the notice. The watch's `status` says what
  it is doing; `stop()` ends it.

  Args:
    notice: the preemption.Notice to set.
    opener: urlopen (tests inject a fake).
    retry_s: seconds before a failed poll is retried.

  Returns:
    the PreemptionWatch.
  """
  stop = threading.Event()
  status = {"text": "starting"}

  def loop():
    try:
      first, etag = _get(opener, PREEMPTED_URL, 2.0)
    except (urllib.error.URLError, OSError) as e:
      status["text"] = (
          f"off GCE (no metadata server: {type(e).__name__}: {e}); not watching"
      )
      logging.info("preemption: %s", status["text"])
      return
    if first.upper() == "TRUE":
      notice.set("gce")
      status["text"] = "preempted (first read)"
      logging.warning("preemption: GCE reports this VM preempted")
      return
    status["text"] = "watching instance/preempted"
    logging.info("preemption: on GCE: %s", status["text"])
    base = f"{PREEMPTED_URL}?wait_for_change=true&timeout_sec={POLL_TIMEOUT_S}"
    while not stop.is_set():
      url = (
          base
          if etag is None
          else f"{base}&last_etag={urllib.parse.quote(etag)}"
      )
      try:
        got, new_etag = _get(opener, url, POLL_TIMEOUT_S + 30)
      except (urllib.error.URLError, OSError) as e:
        logging.warning(
            "preemption: metadata poll failed (%s: %s); retrying in %.0fs",
            type(e).__name__,
            e,
            retry_s,
        )
        stop.wait(retry_s)
        continue
      etag = new_etag or etag
      if got.upper() == "TRUE":
        notice.set("gce")
        status["text"] = "preempted"
        logging.warning(
            "preemption: GCE preemption notice: checkpointing after the "
            "iteration in flight"
        )
        return
    status["text"] = "stopped"

  th = threading.Thread(target=loop, name="preemption-watch", daemon=True)
  th.start()
  return PreemptionWatch(th, stop, status)


# -- the launcher's side -------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Machine:
  """One entry of the ordered machine menu (module doc).

  Attributes:
    name: the entry's name.
    machine_type: the GCE machine type.
    accelerator: {type, count}, or None.
    platform: PLATFORMS: what JAX must see on this machine.
    zones: the zones to try, in order.
    provisioning: one of PROVISIONING.
    boot_disk_gb: the boot disk size.
    max_run_duration: GCE's --max-run-duration, or None.
  """

  name: str
  machine_type: str
  accelerator: Optional[Mapping[str, Any]]  # {type, count} or None
  platform: str  # PLATFORMS: what JAX must see on this machine
  zones: tuple[str, ...]
  provisioning: str
  boot_disk_gb: int
  max_run_duration: Optional[str]


@dataclasses.dataclass(frozen=True)
class GcpConfig:
  """The parsed gcp backend config (module doc); build it with parse().

  Attributes:
    project: the GCP project id.
    storage: the gs:// root runs live under.
    image: the VM image (family or name, and its project).
    network: the VM network settings.
    service_account: the VM service account settings.
    labels: labels on every VM.
    code_root: the local dir code paths are relative to.
    code_paths: the code paths shipped to the VM.
    setup: shell lines the VM runs before the trainer.
    python: the VM's interpreter.
    create_lock: a file locked around each create, or None.
    machines: the ordered machine menu.
  """

  project: str
  storage: str
  image: Mapping[str, Any]
  network: Mapping[str, Any]
  service_account: Mapping[str, Any]
  labels: Mapping[str, str]
  code_root: str
  code_paths: tuple[str, ...]
  setup: tuple[str, ...]
  python: str
  create_lock: Optional[str]
  machines: tuple[Machine, ...]

  @classmethod
  def parse(cls, d: Mapping[str, Any], base_dir: str = ".") -> "GcpConfig":
    """Validates the config body (every field required).

    Args:
      d: the config body.
      base_dir: the dir relative paths are read from.

    Returns:
      the GcpConfig.

    Raises:
      BackendFileError: a field is missing, unknown or invalid.
    """
    d = config_lib.require(
        d,
        "gcp",
        {
            "project": str,
            "storage": str,
            "image": dict,
            "network": dict,
            "service_account": dict,
            "labels": dict,
            "code": dict,
            "setup": list,
            "python": str,
            "create_lock": (str, type(None)),
            "machines": list,
        },
    )
    if not re.match(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$", d["project"]):
      raise config_lib.BackendFileError(
          f"gcp.project {d['project']!r} is not a project id"
      )
    if not re.match(r"^gs://[a-z0-9][a-z0-9._-]+(/.*)?$", d["storage"]):
      raise config_lib.BackendFileError(
          f"gcp.storage {d['storage']!r}: want gs://<bucket>[/prefix]"
      )
    img = config_lib.require(
        d["image"],
        "gcp.image",
        {
            "family": (str, type(None)),
            "name": (str, type(None)),
            "project": str,
        },
    )
    if (img["family"] is None) == (img["name"] is None):
      raise config_lib.BackendFileError(
          "gcp.image: give exactly one of family and name"
      )
    net = config_lib.require(
        d["network"],
        "gcp.network",
        {"network": str, "subnetwork": (str, type(None)), "external_ip": bool},
    )
    sa = config_lib.require(
        d["service_account"],
        "gcp.service_account",
        {"email": str, "scopes": list},
    )
    if not sa["scopes"] or not all(
        isinstance(s, str) and s for s in sa["scopes"]
    ):
      raise config_lib.BackendFileError(
          "gcp.service_account.scopes: a non-empty list of scope URLs"
      )
    for k, v in d["labels"].items():
      if not re.match(r"^[a-z][a-z0-9_-]{0,62}$", str(k)) or not re.match(
          r"^[a-z0-9_-]{0,63}$", str(v)
      ):
        raise config_lib.BackendFileError(
            f"gcp.labels {k}={v!r}: GCE label keys/values are lower-case"
        )
    code = config_lib.require(
        d["code"], "gcp.code", {"root": str, "paths": list}
    )
    root = (
        code["root"]
        if os.path.isabs(code["root"])
        else os.path.join(base_dir, code["root"])
    )
    if not code["paths"]:
      raise config_lib.BackendFileError(
          "gcp.code.paths: name what the VM needs (e.g. the package folder)"
      )
    if not all(isinstance(x, str) and x for x in d["setup"]):
      raise config_lib.BackendFileError("gcp.setup: a list of shell lines")
    machines = []
    if not d["machines"]:
      raise config_lib.BackendFileError("gcp.machines: at least one machine")
    for i, m in enumerate(d["machines"]):
      m = config_lib.require(
          m,
          f"gcp.machines[{i}]",
          {
              "name": str,
              "machine_type": str,
              "accelerator": (dict, type(None)),
              "platform": str,
              "zones": list,
              "provisioning": str,
              "boot_disk_gb": int,
              "max_run_duration": (str, type(None)),
          },
      )
      if m["provisioning"] not in PROVISIONING:
        raise config_lib.BackendFileError(
            f"gcp.machines[{i}].provisioning {m['provisioning']!r}: want one "
            "of "
            f"{PROVISIONING}"
        )
      if m["provisioning"] == "FLEX_START" and m["max_run_duration"] is None:
        raise config_lib.BackendFileError(
            f"gcp.machines[{i}]: a FLEX_START machine needs max_run_duration"
        )
      if m["max_run_duration"] is not None:
        _check_max_run(
            m["max_run_duration"], m["provisioning"], f"gcp.machines[{i}]"
        )
      if not m["zones"] or not all(
          isinstance(z, str) and z for z in m["zones"]
      ):
        raise config_lib.BackendFileError(
            f"gcp.machines[{i}].zones: a non-empty list of zones"
        )
      if m["accelerator"] is not None:
        config_lib.require(
            m["accelerator"],
            f"gcp.machines[{i}].accelerator",
            {"type": str, "count": int},
        )
      if m["platform"] not in PLATFORMS:
        raise config_lib.BackendFileError(
            f"gcp.machines[{i}].platform {m['platform']!r}: want one of "
            f"{PLATFORMS}"
        )
      if m["accelerator"] is not None and m["platform"] == "cpu":
        raise config_lib.BackendFileError(
            f"gcp.machines[{i}]: an accelerator on a platform: cpu machine"
        )
      machines.append(
          Machine(
              name=m["name"],
              machine_type=m["machine_type"],
              accelerator=m["accelerator"],
              platform=m["platform"],
              zones=tuple(m["zones"]),
              provisioning=m["provisioning"],
              boot_disk_gb=m["boot_disk_gb"],
              max_run_duration=m["max_run_duration"],
          )
      )
    return cls(
        project=d["project"],
        storage=d["storage"].rstrip("/"),
        image=img,
        network=net,
        service_account=sa,
        labels=dict(d["labels"]),
        code_root=os.path.abspath(root),
        code_paths=tuple(code["paths"]),
        setup=tuple(d["setup"]),
        python=d["python"],
        create_lock=d["create_lock"],
        machines=tuple(machines),
    )


def duration_s(text: str) -> int:
  """Seconds in a GCE duration "<N>s|m|h|d"; raises ValueError.

  Args:
    text: the duration.

  Returns:
    its seconds.

  Raises:
    ValueError: text is not "<N>s|m|h|d".
  """
  mt = re.fullmatch(r"(\d+)([smhd])", text)
  if not mt:
    raise ValueError(f"{text!r}: want <N>s|m|h|d")
  return int(mt.group(1)) * _DURATION_UNIT_S[mt.group(2)]


def _check_max_run(text: str, provisioning: str, where: str) -> None:
  try:
    s = duration_s(text)
  except ValueError:
    raise config_lib.BackendFileError(
        f"{where}.max_run_duration {text!r}: want <N>s|m|h|d"
    ) from None
  hi = FLEX_START_MAX_RUN_S if provisioning == "FLEX_START" else MAX_RUN_MAX_S
  if not MAX_RUN_MIN_S <= s <= hi:
    raise config_lib.BackendFileError(
        f"{where}.max_run_duration {text!r}: GCE allows {MAX_RUN_MIN_S}s to "
        f"{hi // 86400}d on a {provisioning} VM"
    )


def machine_label(m: Machine) -> str:
  """What a user calls the machine: its GPU ("L4"), else its machine type.

  Args:
    m: the machine.

  Returns:
    the label.
  """
  if m.accelerator is not None:
    t = m.accelerator["type"]
    return (
        GPU_NAMES.get(t)
        or t.replace("nvidia-", "").replace("tesla-", "").upper()
    )
  for prefix, gpu in GPU_FAMILIES:
    if m.machine_type.startswith(prefix):
      return gpu
  return m.machine_type


def vm_name(run_id: str, attempt: int) -> str:
  """The VM name of an attempt (at most 63 characters).

  Args:
    run_id: the run.
    attempt: the attempt number.

  Returns:
    the name.
  """
  return f"train-{run_id}-a{int(attempt)}"[:63].rstrip("-")


def code_bundle(root: str, paths: Sequence[str]) -> bytes:
  """A deterministic tar.gz of paths under root (sorted, no caches, mtime 0).

  Args:
    root: the code root.
    paths: paths under root.

  Returns:
    the tar.gz bytes.

  Raises:
    FileNotFoundError: a path does not exist.
  """
  buf = io.BytesIO()
  files = []
  for p in paths:
    full = os.path.join(root, p)
    if not os.path.exists(full):
      raise FileNotFoundError(f"code path {full} does not exist")
    if os.path.isfile(full):
      files.append(p)
      continue
    for d, dirs, names in os.walk(full):
      dirs[:] = sorted(
          x for x in dirs if x not in ("__pycache__", ".pytest_cache", ".git")
      )
      for n in sorted(names):
        if not n.endswith((".pyc", ".pyo")):
          files.append(os.path.relpath(os.path.join(d, n), root))
  with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6) as tf:
    for rel in sorted(set(files)):
      info = tf.gettarinfo(os.path.join(root, rel), arcname=rel)
      info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
      with open(os.path.join(root, rel), "rb") as f:
        tf.addfile(info, f)
  return buf.getvalue()


def startup_script(
    cfg: GcpConfig,
    run: backend_lib.RunSpec,
    attempt: int,
    run_uri: str,
    code_uri: str,
    machine: Machine,
) -> str:
  """The VM's startup script for one attempt on `machine` (module doc).

  Args:
    cfg: the GCP config.
    run: the run.
    attempt: the attempt number.
    run_uri: the run's storage root.
    code_uri: the code bundle's URI.
    machine: the machine.

  Returns:
    the script.
  """
  a = backend_lib.join(run_uri, "attempts", str(int(attempt)))
  setup = "\n".join(f"step bash -c {shlex.quote(line)}" for line in cfg.setup)
  trainer = run.trainer_shell(
      run_uri, cfg.python, extra_env={EXPECT_PLATFORM_ENV: machine.platform}
  )
  return f"""#!/bin/bash
# Attempt {int(attempt)} of run {run.run_id} (launch.gcp): install, train,
# copy back, power off.
set -u -o pipefail
# gcloud on GCE images (snap or package)
export PATH="/snap/bin:/usr/local/bin:$PATH"
DONE=/var/lib/train-attempt-done
if [ -e "$DONE" ]; then exit 0; fi
A={shlex.quote(a)}
W=/var/lib/train-attempt
mkdir -p "$W"
cd "$W"
LOG="$W/train.log"
put() {{ printf '%s\\n' "$2" | gcloud storage cp - "$A/$1" --quiet; }}
finish() {{
  rc="$1"
  if ! gcloud storage cp "$LOG" "$A/train.log" --quiet; then
    echo "train attempt: copying the log back failed" >&2
  fi
  put exit_code "$rc"
  touch "$DONE"
  shutdown -h now
  exit "$rc"
}}
step() {{ "$@" >>"$LOG" 2>&1 || finish "$?"; }}
watch_preemption() {{
  # last_etag: a change between two long polls returns at once (etag 0: the
  # first poll returns the current value).
  etag=0
  while true; do
    url="{METADATA}preempted?wait_for_change=true&timeout_sec={POLL_TIMEOUT_S}"
    v=$(curl -sf -D "$W/preempted.h" -H 'Metadata-Flavor: Google' \\
        "$url&last_etag=$etag") || {{
      sleep {POLL_RETRY_S}; continue; }}
    etag=$(tr -d '\r' <"$W/preempted.h" |
      awk 'tolower($1) == "etag:" {{print $2}}')
    etag=${{etag:-0}}
    if [ "$v" = "TRUE" ]; then
      put preempted "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      pkill -TERM -f {shlex.quote(backend_lib.TRAINER_MODULE)}
      return
    fi
  done
}}
step put started "$(date -u +%Y-%m-%dT%H:%M:%SZ) $(hostname)"
step gcloud storage cp {shlex.quote(code_uri)} code.tar.gz --quiet
step tar -xzf code.tar.gz
{setup}
watch_preemption &
export PYTHONPATH="$W"
{trainer} >>"$LOG" 2>&1
finish "$?"
"""


def _instance_missing(
    stderr: str, handle: dict[str, Any], project: str
) -> bool:
  """Whether a gcloud compute error says this attempt's VM does not exist.

  Whether a gcloud compute error says this attempt's VM does not exist:
  its own resource path, not found. Any other error (a missing zone or
  project, a permission) is not read as a missing VM.

  Args:
    stderr: the gcloud command's stderr.
    handle: the attempt's handle.
    project: the GCP project id.

  Returns:
    True when the error is this VM, not found.
  """
  path = f"projects/{project}/zones/{handle['zone']}/instances/{handle['name']}"
  return path in stderr and "was not found" in stderr


class GcpBackend(backend_lib.Backend):
  """The GCP backend (module doc).

  Attributes:
    cfg: the parsed config.
    dry_run: log the commands that create or change anything instead of
      running them; reads still run.
    io_tries: tries per storage call (the storage retry policy).
    commands: every gcloud argv this backend ran (or would run, dry_run).
    storage: the run's storage (gs://, through gcloud).
  """

  name = "gcp"

  def __init__(
      self,
      cfg: GcpConfig,
      *,
      runner: storage_lib.Runner = storage_lib.run_command,
      dry_run: bool = False,
      io_tries: int = storage_lib.IO_TRIES,
      sleep: Callable[[float], Any] = time.sleep,
  ):
    """Initializes the backend.

    Args:
      cfg: the parsed config.
      runner: runs a gcloud argv (tests inject a fake).
      dry_run: log the commands that create or change anything instead of
        running them; reads still run.
      io_tries: tries per storage call (the storage retry policy).
      sleep: the sleep between storage tries.
    """
    assert isinstance(cfg, GcpConfig), type(cfg)
    self.cfg = cfg
    self._runner = runner
    self.dry_run = dry_run
    self.io_tries, self._sleep = int(io_tries), sleep
    # every gcloud argv this backend ran (or would run, dry_run)
    self.commands: list[list[str]] = []
    self._code_uri: dict[str, str] = {}
    self.storage = storage_lib.GcsStorage(
        self._storage_call, tries=self.io_tries, sleep=self._sleep
    )

  # -- gcloud ------------------------------------------------------------
  def _gcloud(
      self,
      argv: Sequence[str],
      *,
      # `input` is the keyword the callers pass.
      input: Optional[str] = None,  # pylint: disable=redefined-builtin
      read: bool = False,
  ) -> subprocess.CompletedProcess[str]:
    """Runs gcloud, or logs it in a dry run unless it only reads.

    Args:
      argv: The arguments after gcloud.
      input: Standard input.
      read: Whether the call only reads (it runs even in a dry run).

    Returns:
      The completed process.
    """
    argv = ["gcloud", *argv]
    self.commands.append(argv)
    if self.dry_run and not read:
      logging.info(
          "gcp backend (dry run): %s", " ".join(shlex.quote(a) for a in argv)
      )
      return subprocess.CompletedProcess(argv, 0, "", "")
    return self._runner(argv, input=input)

  def _check(self, p: subprocess.CompletedProcess[str], what: str) -> str:
    """The stdout of a successful gcloud call.

    Args:
      p: The completed process.
      what: The call, for errors.

    Returns:
      The stdout.

    Raises:
      RuntimeError: The call failed.
    """
    if p.returncode != 0:
      raise RuntimeError(
          f"{what} failed (exit {p.returncode}): {p.stderr.strip()[-1500:]}"
      )
    return p.stdout

  # -- storage -----------------------------------------------------------
  def run_uri(self, run_id: str) -> str:
    """The run's storage root, <storage>/<run_id>.

    Args:
      run_id: the run.

    Returns:
      the URI.
    """
    return backend_lib.join(self.cfg.storage, run_id)

  def _storage_call(
      self,
      argv: Sequence[str],
      *,
      # `input` is the keyword the callers pass.
      input: Optional[str] = None,  # pylint: disable=redefined-builtin
  ) -> subprocess.CompletedProcess[str]:
    """Runs a storage gcloud call through _gcloud.

    Args:
      argv: The gcloud command, gcloud first.
      input: Standard input; a call without it only reads.

    Returns:
      The completed process.
    """
    assert argv[0] == "gcloud", argv
    return self._gcloud(list(argv[1:]), input=input, read=input is None)

  def read_text(self, uri: str) -> Optional[str]:
    """The object's text, or None when it does not exist (or in a dry run).

    Args:
      uri: the object.

    Returns:
      its text, or None.
    """
    if self.dry_run:
      return None
    return self.storage.read_text(uri)

  def exists(self, uri: str) -> bool:
    """Whether the object exists (False in a dry run).

    Args:
      uri: the object.

    Returns:
      True when it exists.
    """
    return False if self.dry_run else self.storage.exists(uri)

  # -- the Backend -------------------------------------------------------------
  def prepare(self, run: backend_lib.RunSpec) -> None:
    """Stages the code bundle in the run's storage (idempotent).

    Args:
      run: the run.
    """
    blob = code_bundle(self.cfg.code_root, self.cfg.code_paths)
    uri = backend_lib.join(
        self.run_uri(run.run_id),
        "code",
        hashlib.sha256(blob).hexdigest()[:16] + ".tar.gz",
    )
    self._code_uri[run.run_id] = uri
    if self.read_text(uri) is not None:
      return
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
      f.write(blob)
      path = f.name
    try:
      storage_lib.retried(
          lambda: self._gcloud(["storage", "cp", path, uri, "--quiet"]),
          f"gcloud storage cp {uri}",
          tries=self.io_tries,
          sleep=self._sleep,
      )
    finally:
      os.unlink(path)

  def shipped_packages(self) -> tuple[str, ...]:
    """code.paths as import names (the VM imports them from the code root).

    Returns:
      the import names.
    """
    return tuple(p.strip("/").replace("/", ".") for p in self.cfg.code_paths)

  def plan(self, run: backend_lib.RunSpec, attempt: int) -> str:
    """The dry-run plan of attempt `attempt`.

    The shipped code paths, then each machine's startup script and its
    gcloud create commands, zone by zone, for attempt `attempt`; stages the
    code bundle (nothing in dry run).

    Args:
      run: the run.
      attempt: the attempt number.

    Returns:
      the plan.
    """
    self.prepare(run)
    code_uri = self._code_uri[run.run_id]
    lines = [
        f"# code shipped from {self.cfg.code_root}: "
        f"{' '.join(self.cfg.code_paths)}"
        f" -> {code_uri}"
    ]
    for m in self.cfg.machines:
      lines.append(f"# startup script on {m.name} ({m.platform})")
      lines.append(
          startup_script(
              self.cfg, run, attempt, self.run_uri(run.run_id), code_uri, m
          )
      )
      for zone in m.zones:
        lines.append(
            "gcloud "
            + " ".join(
                self.create_argv(
                    m,
                    zone,
                    vm_name(run.run_id, attempt),
                    run,
                    attempt,
                    "<startup-script>",
                )
            )
        )
    return "\n".join(lines)

  def create_argv(
      self,
      m: Machine,
      zone: str,
      name: str,
      run: backend_lib.RunSpec,
      attempt: int,
      script_path: str,
  ) -> list[str]:
    """The `gcloud compute instances create` argv of one machine in one zone.

    Args:
      m: the machine.
      zone: the zone.
      name: the VM name.
      run: the run.
      attempt: the attempt number.
      script_path: the startup script's local path.

    Returns:
      the argv.
    """
    cfg = self.cfg
    labels = {
        **cfg.labels,
        RUN_LABEL: run.run_id,
        ATTEMPT_LABEL: str(int(attempt)),
    }
    argv = [
        "compute",
        "instances",
        "create",
        name,
        f"--project={cfg.project}",
        f"--zone={zone}",
        f"--machine-type={m.machine_type}",
        f"--provisioning-model={m.provisioning}",
        f"--boot-disk-size={int(m.boot_disk_gb)}GB",
        f"--image-project={cfg.image['project']}",
        (
            f"--image-family={cfg.image['family']}"
            if cfg.image["family"]
            else f"--image={cfg.image['name']}"
        ),
        f"--network={cfg.network['network']}",
    ]
    if cfg.network["subnetwork"]:
      argv.append(f"--subnet={cfg.network['subnetwork']}")
    if not cfg.network["external_ip"]:
      argv.append("--no-address")
    argv += [
        f"--service-account={cfg.service_account['email']}",
        "--scopes=" + ",".join(cfg.service_account["scopes"]),
    ]
    # DELETE, not STOP: a SPOT VM's preemption, and the max run duration
    # (the backstop on any provisioning model if the supervisor dies), take
    # the VM down for good; the next attempt resumes from the checkpoint.
    if m.provisioning == "SPOT" or m.max_run_duration is not None:
      argv.append("--instance-termination-action=DELETE")
    if m.max_run_duration is not None:
      argv.append(f"--max-run-duration={m.max_run_duration}")
    if m.provisioning == "FLEX_START":
      argv.append("--reservation-affinity=none")
    if m.accelerator is not None or m.provisioning == "FLEX_START":
      argv.append("--maintenance-policy=TERMINATE")
    if m.accelerator is not None:
      argv.append(
          f"--accelerator=type={m.accelerator['type']},"
          f"count={m.accelerator['count']}"
      )
    argv += [
        "--labels=" + ",".join(f"{k}={v}" for k, v in sorted(labels.items())),
        f"--metadata-from-file=startup-script={script_path}",
        "--format=json",
    ]
    return argv

  def provision(self, run: backend_lib.RunSpec, attempt: int) -> dict[str, Any]:
    """Creates the attempt's VM: the first machine and zone with capacity.

    Args:
      run: the run (prepared).
      attempt: the attempt number.

    Returns:
      the handle.

    Raises:
      RuntimeError: the run was not prepared, or a create failed for a reason
          other than capacity.
      NoCapacityError: no machine and zone had capacity.
    """
    code_uri = self._code_uri.get(run.run_id)
    if code_uri is None:
      raise RuntimeError(
          "provision before prepare: the code bundle is not staged"
      )
    name = vm_name(run.run_id, attempt)
    tried = []
    lock = _Lock(self.cfg.create_lock)
    for m in self.cfg.machines:
      script = startup_script(
          self.cfg, run, attempt, self.run_uri(run.run_id), code_uri, m
      )
      with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        script_path = f.name
      try:
        for zone in m.zones:
          with lock:
            p = self._gcloud(
                self.create_argv(m, zone, name, run, attempt, script_path)
            )
          if p.returncode == 0:
            logging.info(
                "gcp backend: %s created in %s (%s, %s)",
                name,
                zone,
                m.name,
                m.provisioning,
            )
            return {
                "backend": self.name,
                "attempt": int(attempt),
                "name": name,
                "zone": zone,
                "candidate": m.name,
                "provisioning": m.provisioning,
                "run_id": run.run_id,
                "project": self.cfg.project,
            }
          err = p.stderr.strip()
          if not any(s in err for s in CAPACITY_ERRORS):
            raise RuntimeError(
                f"gcloud compute instances create {name} in {zone} ({m.name}) "
                f"failed, not for capacity (exit {p.returncode}): {err[-1500:]}"
            )
          what = machine_label(m)
          reason = (
              "quota" if any(s in err for s in QUOTA_ERRORS) else "capacity"
          )
          logging.info(
              "gcp backend: no %s %s in %s (%s)", what, reason, zone, m.name
          )
          logging.debug(
              "gcp backend: gcloud compute instances create %s in %s: %s",
              name,
              zone,
              err,
          )
          tried.append({
              "candidate": m.name,
              "zone": zone,
              "what": what,
              "reason": reason,
              "error": err[-400:],
          })
      finally:
        os.unlink(script_path)
    raise backend_lib.NoCapacityError(tried)

  def launch(
      self, handle: dict[str, Any], run: backend_lib.RunSpec, attempt: int
  ) -> dict[str, Any]:
    """GCE starts the trainer from the VM's startup script at boot.

    Args:
      handle: the attempt's handle.
      run: the run.
      attempt: the attempt number.

    Returns:
      the handle, marked launched.
    """
    del run, attempt
    return dict(handle, launched=True)

  def _attempt_uri(self, handle: dict[str, Any], name: str) -> str:
    """The URI of file `name` of the handle's attempt.

    Args:
      handle: The attempt's handle.
      name: The file name.

    Returns:
      The URI.
    """
    return backend_lib.join(
        self.run_uri(handle["run_id"]),
        "attempts",
        str(int(handle["attempt"])),
        name,
    )

  def _describe(self, handle: dict[str, Any]) -> Optional[str]:
    """The VM's status, or None when it no longer exists.

    Args:
      handle: The attempt's handle.

    Returns:
      The status, or None.
    """
    p = self._gcloud(
        [
            "compute",
            "instances",
            "describe",
            handle["name"],
            f"--project={self.cfg.project}",
            f"--zone={handle['zone']}",
            "--format=value(status)",
        ],
        read=True,
    )
    if p.returncode != 0:
      if _instance_missing(p.stderr, handle, self.cfg.project):
        return None
      self._check(p, f"gcloud compute instances describe {handle['name']}")
    return p.stdout.strip()

  def _exit_code(self, handle: dict[str, Any]) -> Optional[int]:
    """The exit code the attempt recorded, if any.

    Args:
      handle: The attempt's handle.

    Returns:
      The exit code, or None before it is written.

    Raises:
      ValueError: The exit_code file is not an integer.
    """
    text = self.read_text(self._attempt_uri(handle, "exit_code"))
    if text is None:
      return None
    if not text.strip().lstrip("-").isdigit():
      raise ValueError(
          f"{self._attempt_uri(handle, 'exit_code')}: not an exit code: "
          f"{text!r}"
      )
    return int(text.strip())

  def poll(self, handle: dict[str, Any]) -> backend_lib.Status:
    """The attempt's Status, from its exit code, the VM and GCE's records.

    Args:
      handle: the attempt's handle.

    Returns:
      the Status.
    """
    if self.dry_run:
      return backend_lib.Status(backend_lib.SUCCEEDED, 0, "dry run")
    rc = self._exit_code(handle)
    if rc is not None:
      return backend_lib.Status(
          backend_lib.classify_exit(rc),
          rc,
          f"exit {rc} (attempts/{handle['attempt']})",
      )
    status = self._describe(handle)
    if status in LIVE_STATES:
      # STOPPING / SUSPENDING: GCE's preemption grace, while the trainer
      # checkpoints and the script copies back; deleting now would cut it short.
      return backend_lib.Status(backend_lib.RUNNING)
    rc = self._exit_code(handle)  # written just before the power-off
    if rc is not None:
      return backend_lib.Status(
          backend_lib.classify_exit(rc),
          rc,
          f"exit {rc} (attempts/{handle['attempt']})",
      )
    where = f"{handle['name']} {status or 'gone'}"
    if self.read_text(self._attempt_uri(handle, "preempted")) is not None:
      return backend_lib.Status(
          backend_lib.PREEMPTED, None, f"{where}: the preemption notice came"
      )
    if self._preempted_op(handle):
      return backend_lib.Status(
          backend_lib.PREEMPTED, None, f"{where}: GCE recorded a preemption"
      )
    if self.read_text(self._attempt_uri(handle, "started")) is None:
      return backend_lib.Status(
          backend_lib.FAILED,
          None,
          f"{where}: the attempt never wrote its started marker "
          "(image, startup script or storage permission; not retried)",
      )
    return backend_lib.Status(
        backend_lib.PREEMPTED,
        None,
        f"{where}: the VM is gone or stopped without an exit code (lost)",
    )

  def _preempted_op(self, handle: dict[str, Any]) -> bool:
    """Whether GCE logged a preemption of the handle's VM.

    Args:
      handle: The attempt's handle.

    Returns:
      True when a preempted operation names the VM.
    """
    flt = (
        'operationType="compute.instances.preempted" AND '
        f'targetLink:"/instances/{handle["name"]}"'
    )
    p = self._gcloud(
        [
            "compute",
            "operations",
            "list",
            f"--project={self.cfg.project}",
            f"--zones={handle['zone']}",
            f"--filter={flt}",
            "--format=json",
        ],
        read=True,
    )
    ops = json.loads(self._check(p, "gcloud compute operations list") or "[]")
    return any(
        str(o.get("targetLink", "")).endswith(f"/instances/{handle['name']}")
        for o in ops
    )

  def teardown(self, handle: dict[str, Any]) -> None:
    """Deletes the attempt's VM; idempotent.

    Args:
      handle: the attempt's handle.
    """
    if "name" not in handle:
      return
    p = self._gcloud([
        "compute",
        "instances",
        "delete",
        handle["name"],
        f"--project={self.cfg.project}",
        f"--zone={handle['zone']}",
        "--quiet",
    ])
    if p.returncode != 0 and not _instance_missing(
        p.stderr, handle, self.cfg.project
    ):
      self._check(p, f"gcloud compute instances delete {handle['name']}")

  def adopt(self, handle: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The handle when its VM or exit code still exists, else None.

    Args:
      handle: the previous attempt's handle.

    Returns:
      the handle, or None.
    """
    if "name" not in handle:
      return None
    if (
        self._describe(handle) is not None
        or self._exit_code(handle) is not None
    ):
      return dict(handle)
    return None


class _Lock:
  """An exclusive flock on a file around each create (None: no lock).

  Attributes:
    path: the lock file, or None for no lock.
  """

  def __init__(self, path: Optional[str]):
    self.path = path
    self._f = None

  def __enter__(self):
    if self.path:
      # Lazy: an optional dependency (fcntl is POSIX-only; no lock, no need).
      import fcntl  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

      os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
      # The lock file stays open until __exit__ releases it.
      self._f = open(self.path, "a+", encoding="utf-8")  # pylint: disable=consider-using-with
      fcntl.flock(self._f, fcntl.LOCK_EX)
    return self

  def __exit__(self, *exc):
    if self._f is not None:
      # Lazy: an optional dependency (fcntl is POSIX-only; no lock, no need).
      import fcntl  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

      fcntl.flock(self._f, fcntl.LOCK_UN)
      self._f.close()
      self._f = None
    return False

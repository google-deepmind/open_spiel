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

"""Tests of the GCP example configs as a stranger runs them.

The GCP example configs (launch/examples/*.yaml) as a stranger runs them:
they load with only their three environment variables, they ship the package
folder under its import name wherever it is installed, every file a setup
line installs is in the shipped code, every package it installs is pinned (to
the requirements.txt pins where that file ships), a gs:// run installs gcsfs,
a GPU machine installs a CUDA jax on an image with the NVIDIA driver, and the
--dry_run of launch.run plans every machine. Nothing touches GCP.
"""

import glob
import os
import re
import subprocess
import sys

import pytest

from open_spiel.python.scale.launch import backend as backend_lib
from open_spiel.python.scale.launch import config as config_lib
from open_spiel.python.scale.launch import hardware
from open_spiel.python.scale.launch import run as run_cli
from open_spiel.python.scale.launch import subprocess_env

EXAMPLES = sorted(
    glob.glob(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "examples",
            "*.yaml",
        )
    )
)
ENV = {
    "PROJECT": "my-project-123",
    "BUCKET": "my-bucket",
    "SERVICE_ACCOUNT": "vm@my-project-123.iam.gserviceaccount.com",
}


def test_both_examples_exist():
  """Both GCP examples exist."""
  assert [os.path.basename(p) for p in EXAMPLES] == [
      "gcp_cpu.yaml",
      "gcp_gpu.yaml",
  ]


@pytest.fixture(name="env")
def _env_fixture(monkeypatch):
  """Sets the examples' environment variables.

  Args:
    monkeypatch: pytest's monkeypatch.
  """
  for k, v in ENV.items():
    monkeypatch.setenv(k, v)


def _requirements(root, rel):
  """The requirement lines of a file, comments dropped.

  Args:
    root: The root folder.
    rel: The file's relative path.

  Returns:
    The requirements.
  """
  out = []
  for line in open(os.path.join(root, rel), encoding="utf-8"):
    line = line.split("#", 1)[0].strip()
    if line:
      out.append(line)
  return out


@pytest.mark.parametrize("path", EXAMPLES, ids=os.path.basename)
def test_an_example_ships_what_its_setup_installs(path, env):
  """An example ships every file its setup installs."""
  del env
  be = config_lib.make_backend(config_lib.load(path), dry_run=True)
  cfg = be.cfg
  shipped = [p.rstrip("/") for p in cfg.code_paths]
  for p in shipped:
    assert os.path.exists(os.path.join(cfg.code_root, p)), p
  pkg_rel = os.path.relpath(
      subprocess_env.PACKAGE_DIR, subprocess_env.IMPORT_ROOT
  )
  assert os.path.realpath(cfg.code_root) == os.path.realpath(
      subprocess_env.IMPORT_ROOT
  )
  assert pkg_rel.replace(os.sep, "/") in shipped, (pkg_rel, shipped)
  installed = []
  for line in cfg.setup:
    for rel in re.findall(r"-r\s+(\S+)", line):
      assert any(rel == p or rel.startswith(p + "/") for p in shipped), (
          rel,
          shipped,
      )
      assert os.path.isfile(os.path.join(cfg.code_root, rel)), rel
      installed += _requirements(cfg.code_root, rel)
    # a quoted argument of pip install is a requirement
    if "pip install" in line:
      installed += re.findall(r'"([^"]+)"', line)
  assert cfg.python.startswith("/opt/venv/") and any(
      "venv /opt/venv" in s for s in cfg.setup
  )
  for r in installed:
    assert re.fullmatch(r"[\w.\[\]-]+==[\w.]+", r), r
  req = os.path.join(subprocess_env.PACKAGE_DIR, "requirements.txt")
  # where it ships, the pins are the package's tested ones
  if os.path.isfile(req):
    pins = dict(r.split("==") for r in installed if not r.startswith("jax["))
    for r in _requirements(subprocess_env.PACKAGE_DIR, "requirements.txt"):
      name, _, version = r.partition("==")
      if name in pins:
        assert pins[name] == version, (name, pins[name], version)
  if cfg.storage.startswith("gs://"):
    assert any(r.startswith("gcsfs") for r in installed), installed
  platforms = {m.platform for m in cfg.machines}
  assert len(platforms) == 1, platforms
  if platforms == {"gpu"}:
    assert any(re.match(r"jax\[cuda\d+\]==", r) for r in installed), installed
    assert "nvidia" in (cfg.image["family"] or cfg.image["name"])
  else:
    assert not any("cuda" in r for r in installed), installed


@pytest.mark.parametrize("path", EXAMPLES, ids=os.path.basename)
def test_an_examples_dry_run_plans_every_machine(path, env, capsys):
  """An example's dry run plans every machine."""
  del env
  rc = run_cli.main([
      "prog",
      f"--backend_config={path}",
      "--run_id=kuhn-a",
      "--dry_run",
      "--",
      "--config=mmd_kuhn_poker_jax",
  ])
  out = capsys.readouterr().out
  cfg = config_lib.make_backend(config_lib.load(path), dry_run=True).cfg
  assert rc == 0 and "nothing created" in out
  creates = sum(len(m.zones) for m in cfg.machines)
  assert out.count("gcloud compute instances create train-kuhn-a-a0") == creates
  for m in cfg.machines:
    assert (
        f"{hardware.EXPECT_PLATFORM_ENV}={m.platform} /opt/venv/bin/python -u "
        "-m "
        f"{backend_lib.TRAINER_MODULE}"
    ) in out
  assert "--checkpoint_dir=gs://my-bucket/runs/kuhn-a/ckpt" in out


def test_the_gpu_example_leads_with_what_a_new_project_can_get(env):
  """The GPU example leads with what a new project can get."""
  del env
  path = next(p for p in EXAMPLES if p.endswith("gcp_gpu.yaml"))
  cfg = config_lib.make_backend(config_lib.load(path), dry_run=True).cfg
  types = [
      (m.machine_type, (m.accelerator or {}).get("type")) for m in cfg.machines
  ]
  assert types[0] == ("g2-standard-8", None)  # L4 (built in)
  assert types[1] == ("n1-standard-8", "nvidia-tesla-t4")
  # H100 last
  assert all(t.startswith("a3-") for t, _ in types[2:]) and types[2:]
  # an on-demand GPU has the GCE backstop if the supervisor dies
  for m in cfg.machines:
    if m.provisioning in ("STANDARD", "FLEX_START"):
      assert m.max_run_duration is not None, m.name
  text = open(path, encoding="utf-8").read()
  assert "GPUS_ALL_REGIONS" in text and "quota" in text


@pytest.mark.parametrize("path", EXAMPLES, ids=os.path.basename)
def test_an_examples_apt_lines_wait_out_a_held_package_list_lock(
    path, env, tmp_path
):
  """An example's apt lines wait out a held package list lock.

  The image's boot-time apt may hold /var/lib/apt/lists/lock, which
  DPkg::Lock::Timeout does not wait for (a real run of the CPU example exited
  100 on it): each apt line retries, and still fails when apt never frees.

  Args:
    path: the example config (parametrized).
    env: the example's environment (fixture).
    tmp_path: pytest's per-test directory.
  """
  del env
  cfg = config_lib.make_backend(config_lib.load(path), dry_run=True).cfg
  apt_lines = [line for line in cfg.setup if "apt-get" in line]
  assert apt_lines, cfg.setup
  bin_dir = tmp_path / "bin"
  bin_dir.mkdir()
  (bin_dir / "sleep").write_text("#!/bin/bash\nexit 0\n")
  (bin_dir / "apt-get").write_text(
      '#!/bin/bash\nn=$(cat "$APT_COUNT" 2>/dev/null || echo 0)\n'
      'echo $((n + 1)) > "$APT_COUNT"\n[ "$n" -ge "$APT_FAILS" ]\n'
  )
  for f in ("sleep", "apt-get"):
    os.chmod(bin_dir / f, 0o755)
  for i, line in enumerate(apt_lines):
    for fails, want_rc, want_calls in ((2, 0, 3), (1000, 1, 90)):
      count = tmp_path / f"count-{i}-{fails}"
      run_env = {
          **os.environ,
          "PATH": f"{bin_dir}:{os.environ['PATH']}",
          "APT_COUNT": str(count),
          "APT_FAILS": str(fails),
      }
      rc = subprocess.run(
          ["bash", "-c", line],
          env=run_env,
          capture_output=True,
          text=True,
          check=False,
      ).returncode
      assert (rc, int(count.read_text())) == (want_rc, want_calls), (
          line,
          fails,
      )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

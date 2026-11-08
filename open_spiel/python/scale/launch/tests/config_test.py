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

"""Tests of the backend registry and storage.

The backend registry and storage: a registered kind builds its Backend from
the config file, an unknown kind names the registered ones, LocalStorage
round-trips atomically, and a Backend's plan describes an attempt.
"""

import os
import sys

import pytest
import yaml

from open_spiel.python.scale.launch import backend as backend_lib
from open_spiel.python.scale.launch import config as config_lib
from open_spiel.python.scale.launch import local
from open_spiel.python.scale.launch import storage
from open_spiel.python.scale.launch import subprocess_env

RUN = backend_lib.RunSpec("kuhn-a", train_args=("--config=mmd_kuhn_poker_jax",))


class _Recorded(backend_lib.Backend):
  """A backend that records the arguments its factory got.

  Attributes:
    body: the config body.
    base_dir: the config file's directory.
    dry_run: whether it was built for a dry run.
    storage: local storage.
  """

  name = "recorded"

  def __init__(self, body, base_dir, dry_run):
    self.body, self.base_dir, self.dry_run = dict(body), base_dir, dry_run
    self.storage = storage.LocalStorage()

  def run_uri(self, run_id):
    """The run's dir under base_dir.

    Args:
      run_id: the run.

    Returns:
      the dir.
    """
    return os.path.join(self.base_dir, run_id)


def test_a_registered_backend_kind_builds_from_the_config_file(
    tmp_path, monkeypatch
):
  """A registered backend kind builds its Backend from the config file."""
  # Reaches private _BACKENDS of the code under test.
  monkeypatch.setattr(config_lib, "_BACKENDS", dict(config_lib._BACKENDS))  # pylint: disable=protected-access
  # plan() only formats the trainer's command: name an interpreter, so the
  # test runs where there is none on disk (no process is started).
  monkeypatch.setattr(subprocess_env, "PYTHON", ("python3",))
  config_lib.register_backend("recorded", _Recorded)
  assert {"local", "gcp", "recorded"} <= set(config_lib.backends())
  path = tmp_path / "backend.yaml"
  path.write_text(yaml.safe_dump({"backend": "recorded", "x": 1}))
  be = config_lib.make_backend(config_lib.load(str(path)), dry_run=True)
  assert isinstance(be, _Recorded) and be.body == {"x": 1} and be.dry_run
  assert be.base_dir == str(tmp_path)
  assert be.plan(RUN, 0).endswith(f"--checkpoint_dir={tmp_path}/kuhn-a/ckpt")
  with pytest.raises(ValueError, match="already registered"):
    config_lib.register_backend("recorded", _Recorded)


def test_an_unknown_kind_names_the_registered_ones():
  """An unknown kind names the registered ones."""
  with pytest.raises(
      config_lib.BackendFileError, match="want one of .*'gcp'.*'local'"
  ):
    config_lib.make_backend({"backend": "nope", "_dir": "."})


def test_local_storage_round_trip(tmp_path):
  """LocalStorage round-trips text atomically."""
  st = storage.LocalStorage()
  uri = st.join(str(tmp_path), "runs/", "/kuhn-a", "supervisor.json")
  assert uri == f"{tmp_path}/runs/kuhn-a/supervisor.json"
  assert st.read_text(uri) is None and not st.exists(uri)
  st.write_text(uri, "{}")
  assert st.read_text(uri) == "{}" and st.exists(uri)
  assert not os.path.exists(uri + ".tmp")
  assert storage.join("gs://b/x/", "y") == "gs://b/x/y"


def test_local_backend_reads_through_its_storage_and_plans_its_trainer(
    tmp_path,
):
  """LocalBackend reads through its storage and plans its trainer."""
  be = local.LocalBackend(str(tmp_path), python="/usr/bin/python3-test")
  uri = os.path.join(be.run_uri("kuhn-a"), "state.json")
  be.write_text(uri, "s")
  assert be.read_text(uri) == "s" and be.exists(uri)
  assert be.plan(RUN, 0).startswith(
      f"/usr/bin/python3-test -u -m {backend_lib.TRAINER_MODULE} --config="
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

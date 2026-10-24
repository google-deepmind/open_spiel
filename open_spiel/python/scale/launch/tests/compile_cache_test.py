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

"""Tests of launch.compile_cache.

launch.compile_cache: the persistent XLA compilation cache a run
keeps beside its checkpoints, so a resumed process (after a preemption) loads
its programs instead of compiling them.

  * where the cache lives (known answers): a run dir (local or gs://) ->
    <dir>/xla_cache unless a deployment registered a shared-dir rule; an
    explicit dir wins; "" turns it off; no run dir -> off;
  * end to end: a first train process fills the cache and
    records misses, a second process RESUMING the same run dir records
    cache hits for the same programs (<run dir>/compile_cache.json).
"""

import json
import os
import sys

import jax
import pytest

from open_spiel.python.scale.launch import backend as backend_lib
from open_spiel.python.scale.launch import compile_cache
from open_spiel.python.scale.launch import subprocess_env


def test_cache_dir_known_answers(monkeypatch):
  """cache_dir_for has its known answers."""
  monkeypatch.setattr(compile_cache, "_SHARED_RULE", [])
  assert (
      compile_cache.cache_dir_for(compile_cache.AUTO, "gs://bkt/runs/r1")
      == "gs://bkt/runs/r1/xla_cache"
  )
  assert (
      compile_cache.cache_dir_for(compile_cache.AUTO, "gs://bkt/runs/r1/")
      == "gs://bkt/runs/r1/xla_cache"
  )
  assert (
      compile_cache.cache_dir_for(compile_cache.AUTO, "/tmp/run7")
      == "/tmp/run7/xla_cache"
  )
  assert compile_cache.cache_dir_for(compile_cache.AUTO, None) is None
  assert compile_cache.cache_dir_for("", "gs://bkt/runs/1") is None
  assert (
      compile_cache.cache_dir_for("/data/cc", "gs://bkt/runs/1") == "/data/cc"
  )
  assert compile_cache.cache_dir_for("gs://other/cc", None) == "gs://other/cc"


def test_a_shared_dir_rule_overrides_auto(monkeypatch):
  """A registered shared-dir rule overrides auto."""
  monkeypatch.setattr(compile_cache, "_SHARED_RULE", [])
  compile_cache.register_shared_dir(
      lambda ck: "gs://bkt/shared" if ck.startswith("gs://") else None
  )
  assert (
      compile_cache.cache_dir_for(compile_cache.AUTO, "gs://bkt/runs/a")
      == "gs://bkt/shared"
  )
  assert (
      compile_cache.cache_dir_for(compile_cache.AUTO, "/tmp/run7")
      == "/tmp/run7/xla_cache"
  )


def test_a_gcs_cache_turns_xlas_own_file_caches_off():
  """A GCS cache turns XLA's own file caches off.

  XLA's C++ opens its GPU autotune / kernel cache files under the cache
  dir and cannot open gs:// paths: off for a GCS dir, as JAX had them for a
  local one; disable() restores every setting.
  """
  before = (
      jax.config.jax_compilation_cache_dir,
      jax.config.jax_persistent_cache_enable_xla_caches,
  )
  off = compile_cache.enable("gs://bkt/shared/xla_cache")
  try:
    assert jax.config.jax_compilation_cache_dir == "gs://bkt/shared/xla_cache"
    assert jax.config.jax_persistent_cache_enable_xla_caches == "none"
    assert jax.config.jax_compilation_cache_max_size == -1
  finally:
    off()
  assert (
      jax.config.jax_compilation_cache_dir,
      jax.config.jax_persistent_cache_enable_xla_caches,
  ) == before


def _train(ck, stop_after):
  """The trainer argv for a short Kuhn run.

  Args:
    ck: The checkpoint folder.
    stop_after: Updates before stopping.

  Returns:
    The argv.
  """
  return [
      "-m",
      backend_lib.TRAINER_MODULE,
      "--config=mmd_kuhn_poker_jax",
      f"--checkpoint_dir={ck}",
      "--num_envs=8",
      "--num_steps=8",
      "--num_updates=6",
      f"--stop_after={stop_after}",
      "--preemption_sources=sigterm",
      "--resume_every_min=0",
      "--compilation_cache_min_compile_secs=0",
  ]


def test_a_resumed_process_reuses_the_cache(tmp_path):
  """A resumed process loads its programs from the cache."""
  ck = tmp_path / "ck"
  p1 = subprocess_env.run_python(_train(ck, 2), timeout=900)
  assert p1.returncode == 0, (p1.stdout + p1.stderr)[-3000:]
  cache = ck / compile_cache.SUBDIR
  entries = [f for f in os.listdir(cache) if not f.startswith(".")]
  assert entries, "the first process wrote no cache entry"
  rec = json.loads((ck / compile_cache.RECORD).read_text())["attempts"]
  assert len(rec) == 1 and rec[0]["dir"] == str(cache) and rec[0]["misses"] > 0
  p2 = subprocess_env.run_python(_train(ck, 4), timeout=900)
  assert p2.returncode == 0, (p2.stdout + p2.stderr)[-3000:]
  rec = json.loads((ck / compile_cache.RECORD).read_text())["attempts"]
  assert len(rec) == 2 and rec[1]["dir"] == str(cache)
  # the resumed process loaded programs from the cache
  assert rec[1]["hits"] > 0, rec
  assert rec[1]["hits"] >= rec[1]["misses"], rec  # most of them


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

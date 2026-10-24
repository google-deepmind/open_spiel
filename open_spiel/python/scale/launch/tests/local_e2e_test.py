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

"""End-to-end test of the local backend.

End to end on the local backend: a tiny run of the built-in game launched,
killed (a preemption), resumed and finished by the supervisor, from the
package folder alone (copied out as a release would be, under its import name,
with the optional packages and every module beside it in its source tree
unimportable in the trainer).
"""

import importlib.util
import json
import os
import shutil
import sys
import threading
import time

import pytest

from open_spiel.python.scale import deps
from open_spiel.python.scale import metrics
from open_spiel.python.scale import plugins
from open_spiel.python.scale.ckpt import run_state
from open_spiel.python.scale.launch import backend as backend_lib
from open_spiel.python.scale.launch import local
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.launch import supervisor as supervisor_lib

PKG_DIR = subprocess_env.PACKAGE_DIR
PKG_NAME = os.path.basename(PKG_DIR)
# The packages that hold this one (none when it is top-level): its siblings
# are named under them.
PARENT = subprocess_env.PACKAGE.rpartition(".")[0]
OPTIONAL = deps.OPTIONAL
UPDATES = 24


def _within(name: str, mods) -> bool:
  """Whether module `name` is one of `mods` or inside one.

  Args:
    name: The module.
    mods: The module names.

  Returns:
    True when it is.
  """
  return any(name == m or name.startswith(m + ".") for m in mods)


def _optional(name: str) -> bool:
  """Whether module `name` is an optional package, inside one or holds one.

  Args:
    name: The module.

  Returns:
    True when it is.
  """
  return _within(name, OPTIONAL) or any(
      o.startswith(name + ".") for o in OPTIONAL
  )


def _siblings() -> frozenset[str]:
  """The source-tree modules beside the package folder.

  The importable modules beside the package folder in its source tree that
  resolve to that tree (not the standard library, not an installed package,
  not an optional dependency): a release of the folder alone has none of them.

  Returns:
    the module names.
  """
  root = os.path.dirname(PKG_DIR)
  out = set()
  for name in os.listdir(root):
    mod = name[:-3] if name.endswith(".py") else name
    if (
        not mod.isidentifier()
        or mod in sys.stdlib_module_names
        or mod == PKG_NAME
    ):
      continue
    if not (name.endswith(".py") or os.path.isdir(os.path.join(root, name))):
      continue
    mod = f"{PARENT}.{mod}" if PARENT else mod
    try:
      spec = importlib.util.find_spec(mod)
    except (ImportError, ValueError):
      spec = None
    origin = (
        (spec.origin or (list(spec.submodule_search_locations or [""])[0]))
        if spec
        else ""
    )
    if spec is None or os.path.abspath(str(origin)).startswith(root + os.sep):
      out.add(mod)
  return frozenset(m for m in out if not _within(m, OPTIONAL))


SIBLINGS = _siblings()
#: Every module a release of the package folder alone may not need.
BLOCKED = OPTIONAL + tuple(sorted(SIBLINGS))

_BLOCKER = """import sys
BLOCK = set(%r)
LOG = %r


class _Block:
  def find_spec(self, name, path=None, target=None):
    if any(name == b or name.startswith(b + ".") for b in BLOCK):
      with open(LOG, "a") as f:
        f.write(name + "\\n")
      raise ModuleNotFoundError(
          "blocked by the standalone test: " + name, name=name)
    return None


sys.meta_path.insert(0, _Block())
"""


def _export(tmp_path):
  """Exports the package folder alone under its import name.

  The package folder alone under its import name (with the __init__.py of
  each package holding it), plus a sitecustomize that blocks BLOCKED.

  Args:
    tmp_path: the directory to export under.

  Returns:
    (the PYTHONPATH that holds the blocker, the export and the inherited
    entries other than the package's own import root, the path of the
    blocked-imports log).
  """
  out = tmp_path / "export"
  parts = subprocess_env.PACKAGE.split(".")
  for i in range(1, len(parts)):
    init = os.path.join(subprocess_env.IMPORT_ROOT, *parts[:i], "__init__.py")
    assert os.path.isfile(init), init
    os.makedirs(out.joinpath(*parts[:i]), exist_ok=True)
    shutil.copyfile(init, out.joinpath(*parts[:i], "__init__.py"))
  shutil.copytree(
      PKG_DIR,
      out.joinpath(*parts),
      ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
  )
  block = tmp_path / "block"
  block.mkdir()
  log = tmp_path / "blocked.txt"
  (block / "sitecustomize.py").write_text(_BLOCKER % (BLOCKED, str(log)))
  # Packages found through PYTHONPATH rather than a site directory (a Bazel
  # test's runfiles) stay importable; the package's own import root does not,
  # so the child still imports the package from the export alone.
  root = os.path.realpath(subprocess_env.IMPORT_ROOT)
  inherited = [
      p
      for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
      if p and os.path.realpath(p) != root
  ]
  return os.pathsep.join([str(block), str(out), *inherited]), log


def _wait(pred, timeout_s, what):
  """Polls `pred` until it returns something truthy.

  Args:
    pred: The predicate.
    timeout_s: Seconds to wait.
    what: What is awaited, for errors.

  Returns:
    The predicate's value.

  Raises:
    AssertionError: The wait timed out.
  """
  t0 = time.time()
  while time.time() - t0 < timeout_s:
    got = pred()
    if got:
      return got
    time.sleep(0.5)
  raise AssertionError(f"timed out after {timeout_s}s waiting for {what}")


@pytest.mark.slow
def test_launched_killed_resumed_finished_standalone(tmp_path):
  """A run is launched, killed, resumed and finished from the folder alone."""
  pythonpath, blocked_log = _export(tmp_path)
  # The interpreter is resolved here, in the test's thread (no interpreter
  # skips the test); the supervisor launches from its own thread.
  be = local.LocalBackend(
      str(tmp_path / "runs"),
      python=subprocess_env.python_command(),
      env={"PYTHONPATH": pythonpath, "JAX_PLATFORMS": "cpu", plugins.ENV: ""},
  )
  run = backend_lib.RunSpec(
      "kuhn-e2e",
      train_args=(
          "--config=mmd_kuhn_poker_jax",
          f"--num_updates={UPDATES}",
          "--num_envs=16",
          "--num_steps=8",
          "--set=keep_every=1",
          "--resume_every_min=0",
          "--preemption_sources=sigterm",
          "--compilation_cache_dir=",
      ),
  )
  sup = supervisor_lib.Supervisor(be, poll_s=0.5, max_preemptions=2)
  result = {}
  th = threading.Thread(
      target=lambda: result.update(rc=sup.run(run)), daemon=True
  )
  th.start()
  ck = os.path.join(be.run_uri("kuhn-e2e"), backend_lib.CKPT_DIR)

  def step():
    """Whether attempt 0 has checkpointed 3 iterations.

    Returns:
      True once it has.

    Raises:
      AssertionError: the supervisor ended early.
    """
    if not th.is_alive():
      log = os.path.join(be.run_uri("kuhn-e2e"), "attempts", "0", "train.log")
      tail = ""
      if os.path.exists(log):
        with open(log, encoding="utf-8") as f:
          tail = f.read()[-6000:]
      raise AssertionError(
          f"the supervisor ended early (rc {result.get('rc')}): {tail}"
      )
    ts = run_state.read_train_state(ck) if os.path.exists(ck) else None
    return ts is not None and int(ts["step"]) >= 3

  _wait(step, 600, "attempt 0 to checkpoint 3 iterations")
  st = sup.read_state("kuhn-e2e")
  assert len(st["attempts"]) == 1
  # SIGTERM: what a preemption notice does
  be.preempt(st["attempts"][0]["handle"])
  th.join(timeout=900)
  assert not th.is_alive(), "the supervisor did not finish"
  logs = [
      open(
          os.path.join(be.run_uri("kuhn-e2e"), "attempts", str(n), "train.log"),
          encoding="utf-8",
      ).read()
      for n in (0, 1)
  ]
  assert result.get("rc") == 0, logs[-1][-4000:]
  st = sup.read_state("kuhn-e2e")
  assert [a["outcome"] for a in st["attempts"]] == ["preempted", "done"], st
  assert st["attempts"][0]["rc"] == 143 and st["final"] == "done"
  ts = run_state.read_train_state(ck)
  assert int(ts["step"]) == UPDATES
  rows = [
      r
      for r in metrics.read_jsonl(
          os.path.join(ck, run_state.METRICS_DIR, metrics.JSONL_NAME)
      )
      if "train/main/frames" in r
  ]
  steps = [r["step"] for r in rows]
  # resumed: no gap, no duplicate
  assert steps == list(range(1, UPDATES + 1)), steps
  assert "resuming" in logs[1]
  # Standalone: the run finished with every blocked module unimportable (an
  # optional game package may be tried and skipped, for example the encoder
  # table's games); no package beside the package folder was even tried.
  tried = (
      set(open(blocked_log, encoding="utf-8").read().split())
      if blocked_log.exists()
      else set()
  )
  assert not [
      t for t in tried if _within(t, SIBLINGS) and not _optional(t)
  ], tried
  # Idempotent identity: the same run again launches nothing.
  assert sup.run(run) == 0
  assert len(sup.read_state("kuhn-e2e")["attempts"]) == 2
  json.dumps(st)  # the state is plain JSON


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )

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

"""The persistent XLA compilation cache of a training run.

A resumed process (a new attempt after a preemption) loads the programs the
run already compiled instead of compiling them again; a large league update
can take minutes to compile.

  disable = compile_cache.enable(compile_cache.cache_dir_for("auto",
  checkpoint_dir))

  cache_dir_for(flag, checkpoint_dir)   where it lives: flag "auto" (the
        train default) -> the registered shared-dir rule's answer
        (register_shared_dir: e.g. one cache per bucket that every run shares;
        JAX's cache key covers the program, the jax / XLA version, the
        backend and the device kind), else <checkpoint_dir>/xla_cache; no run
        dir: off; "" off; anything else is the dir itself.
  enable(dir, min_compile_secs)         jax_compilation_cache_dir (gs:// goes
        through etils.epath; XLA's own GPU autotune / kernel caches off for
        a gs:// dir, its C++ cannot open one), no size limit for any dir,
        local or remote (GCS has no eviction lock; a local cache also grows
        until it is cleared), and counters on JAX's cache_hits /
        cache_misses events; call before the first compile. Returns the function
        that turns it off again.
  record(checkpoint_dir, dir)           appends this process's counts to
        <run dir>/compile_cache.json ({"attempts": [...]}).
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Optional

from absl import logging
from etils import epath
import jax

from open_spiel.python.scale.ckpt import artifact

AUTO = "auto"
SUBDIR = "xla_cache"
RECORD = "compile_cache.json"
HIT_EVENT = "/jax/compilation_cache/cache_hits"
MISS_EVENT = "/jax/compilation_cache/cache_misses"

_counts = {"hits": 0, "misses": 0}
_lock = threading.Lock()
_listening = [False]
_SHARED_RULE: list[Callable[[str], Optional[str]]] = []


def register_shared_dir(fn: Callable[[str], Optional[str]]) -> None:
  """Registers the rule "auto" uses to pick a run's cache dir.

  fn(checkpoint_dir) -> the cache dir "auto" picks for that run dir, or
  None for the default (<checkpoint_dir>/xla_cache).

  Args:
    fn: fn(checkpoint_dir) -> the cache dir, or None for the default.
  """
  assert callable(fn), fn
  _SHARED_RULE[:] = [fn]


def cache_dir_for(
    flag: Optional[str], checkpoint_dir: Optional[str]
) -> Optional[str]:
  """The cache dir for a flag value and a run dir (module doc), or None.

  Args:
    flag: the flag's value (None, "auto" or a dir).
    checkpoint_dir: the run dir, or None.

  Returns:
    the cache dir, or None for no cache.
  """
  if not flag:
    return None
  if flag != AUTO:
    return str(flag)
  if not checkpoint_dir:
    return None
  ck = str(checkpoint_dir).rstrip("/")
  if _SHARED_RULE:
    got = _SHARED_RULE[0](ck)
    if got:
      return str(got)
  return f"{ck}/{SUBDIR}"


def _listen(event: str, **kwargs) -> None:
  del kwargs
  if event == HIT_EVENT:
    key = "hits"
  elif event == MISS_EVENT:
    key = "misses"
  else:
    key = None
  if key is not None:
    with _lock:
      _counts[key] += 1


def _reset_jax_cache() -> None:
  """Resets JAX's in-process compilation cache object.

  The public jax.experimental.compilation_cache reset_cache; a JAX release
  without it fails here with a message naming the call.
  """
  try:
    # Lazy: an optional dependency (not every JAX release has this module).
    from jax.experimental.compilation_cache import compilation_cache as jax_cc  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    reset = jax_cc.reset_cache
  except (ImportError, AttributeError) as e:
    raise RuntimeError(
        f"jax {jax.__version__} has no jax.experimental.compilation_cache"
        ".compilation_cache.reset_cache; launch.compile_cache needs a "
        "replacement for it"
    ) from e
  reset()


def enable(
    cache_dir: str, min_compile_secs: Optional[float] = None
) -> Callable[[], None]:
  """Turns the persistent cache on at cache_dir.

  Returns the function that turns it back off (the previous settings, JAX's
  cache object reset): a process that runs train.main more than once does not
  carry one run's cache into the next.

  Args:
    cache_dir: the cache dir.
    min_compile_secs: the least compile time cached (None: JAX's default).

  Returns:
    the function that turns the cache back off.
  """
  assert cache_dir, cache_dir
  names = (
      "jax_compilation_cache_dir",
      "jax_compilation_cache_max_size",
      "jax_persistent_cache_min_compile_time_secs",
      "jax_persistent_cache_enable_xla_caches",
  )
  prev = {n: getattr(jax.config, n) for n in names}
  _reset_jax_cache()  # a cache an earlier enable (or JAX itself) initialized
  jax.config.update("jax_compilation_cache_dir", str(cache_dir))
  jax.config.update("jax_compilation_cache_max_size", -1)
  if "://" in str(cache_dir):
    # XLA's own GPU caches (autotune results, kernels) are files XLA's C++
    # opens under the cache dir; it cannot open gs:// paths, so they stay off
    # for a GCS cache (JAX's executable cache goes through etils.epath).
    jax.config.update("jax_persistent_cache_enable_xla_caches", "none")
  if min_compile_secs is not None:
    assert float(min_compile_secs) >= 0.0, min_compile_secs
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs", float(min_compile_secs)
    )
  if not _listening[0]:
    jax.monitoring.register_event_listener(_listen)
    _listening[0] = True
  with _lock:
    _counts.update(hits=0, misses=0)
  logging.info(
      "compile_cache: persistent XLA cache at %s (min compile %s s)",
      cache_dir,
      "default" if min_compile_secs is None else min_compile_secs,
  )

  def disable() -> None:
    for n, v in prev.items():
      jax.config.update(n, v)
    _reset_jax_cache()

  return disable


def stats() -> dict[str, int]:
  """{hits, misses} of this process since enable().

  Returns:
    {hits, misses}.
  """
  with _lock:
    return dict(_counts)


def record(checkpoint_dir: str, cache_dir: Optional[str]) -> dict[str, Any]:
  """Appends this process's cache counts to compile_cache.json.

  The file is <checkpoint_dir>/compile_cache.json.

  Args:
    checkpoint_dir: the run dir.
    cache_dir: the cache dir, or None.

  Returns:
    the record appended.
  """
  rec = dict(
      stats(),
      dir=cache_dir,
      at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  )
  p = epath.Path(checkpoint_dir) / RECORD
  doc = json.loads(p.read_text()) if p.exists() else {"attempts": []}
  doc["attempts"].append(rec)
  artifact.atomic_write(p, json.dumps(doc, indent=1, sort_keys=True).encode())
  logging.info(
      "compile_cache: %d hits, %d misses (%s)",
      rec["hits"],
      rec["misses"],
      cache_dir,
  )
  return rec

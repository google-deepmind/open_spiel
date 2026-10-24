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

"""Run provenance: the code commit and repository, the run ids, the environment.

An artifact is only reproducible if it records where it came from. This
module supplies the values `write_artifact` and the learner kinds record, and
the hooks a launcher uses to declare what the library cannot discover itself.

  ids = provenance.resolve_run_ids(flag_run_id)     # RunIds(run_id, attempt_id)
  env = provenance.environment_provenance(
      params_sha256=digest, git_commit_sha=provenance.git_commit())

The commit is $OPEN_SPIEL_SCALE_GIT_COMMIT (ENV_VAR) when set (a copy of the
source tree has no .git), else `git rev-parse HEAD` of the git checkout that
tracks this package's own files, with "-dirty" appended when the package's
tracked files have uncommitted changes, else None (logged; an artifact never
claims a commit it does not know). A package installed inside some other
checkout (a virtualenv in a user's repository) is not tracked by it and
records None. The code repository is $OPEN_SPIEL_SCALE_CODE_REPO
(CODE_REPO_ENV), else what `set_code_repo` declared, else None.

A run has two ids. The run id names the run and stays the same across
restarts; the attempt id names one process of it. A launcher passes the run
id as a flag and names the environment variable holding the attempt id with
`set_attempt_id_env`, optionally with an `id_parser` that both ids go
through. Both are recorded and never compared: a restarted process is a new
attempt of the same run.
"""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
from typing import Any, Callable, NamedTuple, Optional

from absl import logging

from open_spiel.python.scale import plugins

ENV_VAR = plugins.ENV_PREFIX + "GIT_COMMIT"
CODE_REPO_ENV = plugins.ENV_PREFIX + "CODE_REPO"


def git_commit() -> Optional[str]:
  """The commit of the running code, or None when it cannot be determined.

  Returns:
    The commit sha, or None.
  """
  sha = os.environ.get(ENV_VAR, "").strip()
  if sha:
    return sha
  this = os.path.abspath(__file__)
  here = os.path.dirname(this)
  package = os.path.dirname(here)

  def git(*args):
    """Runs git in this package's directory.

    Args:
      *args: the git arguments.

    Returns:
      The completed process.
    """
    return subprocess.run(
        ["git", "-C", here, *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

  try:
    tracked = git("ls-files", "--error-unmatch", "--", this)
  except FileNotFoundError:
    logging.warning(
        "provenance: no git binary; artifacts record git_commit=None"
    )
    return None
  if tracked.returncode != 0:
    logging.warning(
        "provenance: %s is not tracked by a git checkout and $%s is unset; "
        "artifacts record git_commit=None",
        here,
        ENV_VAR,
    )
    return None
  head = git("rev-parse", "HEAD")
  if head.returncode != 0:
    logging.warning(
        "provenance: git rev-parse HEAD failed in %s (%s); artifacts record "
        "git_commit=None",
        here,
        head.stderr.strip(),
    )
    return None
  status = git("status", "--porcelain", "--untracked-files=no", "--", package)
  if status.returncode != 0:
    logging.warning(
        "provenance: git status failed in %s (%s); artifacts record "
        "git_commit=None",
        here,
        status.stderr.strip(),
    )
    return None
  sha = head.stdout.strip()
  return f"{sha}-dirty" if status.stdout.strip() else sha


def _as_string(v: Any) -> str:
  """A non-empty run id string.

  Args:
    v: The value.

  Returns:
    The stripped string.

  Raises:
    ValueError: The id is empty.
  """
  s = str(v).strip()
  if not s:
    raise ValueError("a run id is a non-empty string")
  return s


_STATE: dict[str, Any] = {
    "attempt_id_env": None,
    "parse": _as_string,
    "code_repo": None,
}


def set_attempt_id_env(
    name: Optional[str], *, id_parser: Optional[Callable[[Any], Any]] = None
) -> None:
  """Names the environment variable holding this process's attempt id.

  Args:
    name: the variable name, or None for none.
    id_parser: converts a raw id into the value recorded, raising ValueError
      on a malformed one; it parses both ids, the run id from the flag and the
      attempt id from the variable. The default keeps each as a non-empty
      string.
  """
  assert name is None or (isinstance(name, str) and name), name
  assert id_parser is None or callable(id_parser), id_parser
  _STATE["attempt_id_env"] = name
  _STATE["parse"] = id_parser or _as_string


def attempt_id_env() -> Optional[str]:
  """The environment variable set_attempt_id_env named, or None.

  Returns:
    The variable name, or None.
  """
  return _STATE["attempt_id_env"]


def set_code_repo(repo: Optional[str]) -> None:
  """Sets the code repository artifacts record.

  Args:
    repo: the repository, or None when unknown.
  """
  assert repo is None or (isinstance(repo, str) and repo), repo
  _STATE["code_repo"] = repo


def code_repo() -> Optional[str]:
  """The code repository artifacts record, or None.

  The environment variable CODE_REPO_ENV wins over set_code_repo.

  Returns:
    The repository, or None.
  """
  return os.environ.get(CODE_REPO_ENV, "").strip() or _STATE["code_repo"]


class RunIds(NamedTuple):
  """The two ids of a training process.

  run_id: the run's id, from the launcher's flag; None when the launcher
    gives none (the league runner then takes it from the run directory on a
    restart, else uses the attempt id).
  attempt_id: this process's id, from the environment variable named by
    `set_attempt_id_env`, or None.

  Attributes:
    run_id: the run's id.
    attempt_id: this process's attempt id.
  """

  run_id: Any
  attempt_id: Any


def resolve_run_ids(flag_run_id: Any) -> RunIds:
  """RunIds(the flag's run id, this attempt's id), each parsed or None.

  Args:
    flag_run_id: the run id flag's value.

  Returns:
    The RunIds.

  Raises:
    ValueError: when either id is malformed for the registered parser.
  """
  parse, name = _STATE["parse"], attempt_id_env()
  raw = os.environ.get(name, "").strip() if name else ""
  try:
    attempt = parse(raw) if raw else None
  except ValueError as e:
    raise ValueError(f"${name}={raw!r}: {e}") from e
  try:
    run = None if flag_run_id is None else parse(flag_run_id)
  except ValueError as e:
    raise ValueError(f"run id {flag_run_id!r}: {e}") from e
  return RunIds(run_id=run, attempt_id=attempt)


def package_version(name: str) -> Optional[str]:
  """The installed version of distribution `name`, or None when it is absent.

  Args:
    name: the distribution name.

  Returns:
    The version, or None.
  """
  try:
    return importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:
    return None


_EXTRAS: dict[str, Callable[[], Any]] = {}
_CORE_KEYS = (
    "jax",
    "jaxlib",
    "git_commit",
    "params_sha256",
    "device_kind",
    "num_devices",
    "platform",
    "xla_flags",
    "matmul_precision",
)


def register_provenance_extra(key: str, fn: Callable[[], Any]) -> None:
  """`environment_provenance` also records key: fn() (a JSON value).

  Args:
    key: the provenance key.
    fn: returns the value.

  Raises:
    ValueError: for a key the library records itself, or one already registered
      with another function.
  """
  assert isinstance(key, str) and key and callable(fn), (key, fn)
  if key in _CORE_KEYS:
    raise ValueError(
        f"provenance key {key!r} is recorded by the library itself"
    )
  if _EXTRAS.get(key, fn) is not fn:
    raise ValueError(f"provenance extra {key!r} is already registered")
  _EXTRAS[key] = fn


def _mctx_version() -> Optional[str]:
  """The installed mctx version.

  Returns:
    The version, or None.
  """
  return package_version("mctx")


register_provenance_extra("mctx", _mctx_version)


def environment_provenance(
    *, params_sha256: Optional[str], git_commit_sha: Optional[str]
) -> dict[str, Any]:
  """The software and hardware a computation ran on.

  Records the jax and jaxlib versions, the commit and params digest given,
  the device kind and count, the platform, $XLA_FLAGS, the default matmul
  precision, and every registered extra. Search targets reproduce from an
  artifact on the same devices with the same flags, so a record of these is
  what makes such a claim checkable. A package that is not installed is
  recorded as None.

  Args:
    params_sha256: the params digest, or None.
    git_commit_sha: the commit, or None.

  Returns:
    The provenance record.
  """
  # Lazy: ckpt imports without jax (jax_free_control_test).
  import jax  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  devs = jax.devices()
  return dict(
      jax=package_version("jax"),
      jaxlib=package_version("jaxlib"),
      git_commit=git_commit_sha,
      params_sha256=params_sha256,
      device_kind=devs[0].device_kind if devs else None,
      num_devices=len(devs),
      platform=devs[0].platform if devs else None,
      xla_flags=os.environ.get("XLA_FLAGS", ""),
      matmul_precision=str(jax.config.jax_default_matmul_precision),
      **{k: fn() for k, fn in sorted(_EXTRAS.items())},
  )

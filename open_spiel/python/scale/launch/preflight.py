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

"""Before any machine exists: does the run's game have its game package there?

A remote backend ships code packages to the machine (the gcp backend's
code.paths, the cloud launcher's code.tgz), and a game adapter imports its
game package (envs.registry.GAME_PACKAGES). A run whose game needs a code
package that is not shipped fails at startup on the machine, after the
machine was paid for. check_game_packages finds that here instead:

  check_game_packages(train_args, shipped=("pkg", "games.pkg"))

When every code game package (CODE_GAME_PACKAGES) is shipped it returns at
once. Otherwise it resolves the run config the trainer flags name (--config,
--set, --plugins) and imports its game's adapter in a child process in which
each unshipped game package does not import, exactly as on the machine; an
adapter that needs one raises GamePackageNotShippedError naming it and the fix.
pyspiel is not a code package (it is installed by the machine's setup), so it
is not checked here.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

from absl import logging

from open_spiel.python.scale import deps
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.launch import subprocess_env

#: Game packages that ship as code (a folder under its import name).
CODE_GAME_PACKAGES = (deps.GAMES_PACKAGE,)
#: The child's exit code when the game needs an unshipped game package.
_NOT_SHIPPED_RC = 3
_FRAMEWORK = __name__.rsplit(".", 2)[0]  # <package>.launch.preflight

# The child: makes each blocked package fail to import (as on a machine that
# does not have it), resolves the run config and imports its game's adapter.
_PROBE = r"""
import importlib.abc, json, sys
args = json.loads(sys.argv[1])
blocked = tuple(args["blocked"])

class _Block(importlib.abc.MetaPathFinder):
  def find_spec(self, name, path=None, target=None):
    for pkg in blocked:
      if name == pkg or name.startswith(pkg + "."):
        raise ModuleNotFoundError(
            f"No module named {name!r} (not shipped)", name=name)
    return None

sys.meta_path.insert(0, _Block())
import importlib
fw = args["framework"]
fw_mod = importlib.import_module(fw)
fw_mod.enable_x64()
plugins = importlib.import_module(fw + ".plugins")
common = importlib.import_module(fw + ".league.config_common")
env_registry = importlib.import_module(fw + ".envs.registry")
try:
  plugins.load_env()
  plugins.load(args["plugins"])
  cfg = common.load_config(args["config"])
  cfg = common.apply_overrides(cfg, args["sets"])
  env_registry.env_class(cfg.env)
except ModuleNotFoundError as e:
  root = env_registry.absent_game_package(e.name)
  if root is None:
    raise
  print(json.dumps({"package": root, "blocked": root in blocked,
                    "env": locals().get("cfg") and cfg.env, "message": str(e)}))
  sys.exit(%d)
print(json.dumps({"env": cfg.env}))
""" % _NOT_SHIPPED_RC


class GamePackageNotShippedError(RuntimeError):
  """The run's game needs a code game package the backend does not ship."""


def _covered(pkg: str, shipped: Sequence[str]) -> bool:
  """Whether shipping the import names `shipped` ships package `pkg`.

  Args:
    pkg: The package.
    shipped: The shipped import names.

  Returns:
    True when `pkg` ships.
  """
  return any(pkg == s or pkg.startswith(s + ".") for s in shipped)


def unshipped(shipped: Sequence[str]) -> tuple[str, ...]:
  """The CODE_GAME_PACKAGES that shipping `shipped` leaves out.

  `shipped` holds import names.

  Args:
    shipped: the shipped import names.

  Returns:
    the packages left out.
  """
  return tuple(p for p in CODE_GAME_PACKAGES if not _covered(p, shipped))


def trainer_flags(train_args: Sequence[str]) -> argparse.Namespace:
  """--config, --set (repeatable) and --plugins of the trainer's flags.

  Args:
    train_args: the trainer's flags.

  Returns:
    the parsed flags.

  Raises:
    GamePackageNotShippedError: no --config (there is no game to check).
  """
  ap = argparse.ArgumentParser(add_help=False)
  ap.add_argument("--config")
  ap.add_argument("--set", action="append", default=[])
  ap.add_argument("--plugins", default="")
  a, _ = ap.parse_known_args(list(train_args))
  if not a.config:
    raise GamePackageNotShippedError(
        "the trainer's flags name no --config, so its game cannot be "
        "checked; pass -- --config=NAME"
    )
  return a


def check_game_packages(
    train_args: Sequence[str],
    shipped: Sequence[str],
    plugins: Sequence[str] = (),
    *,
    fix: str = "",
    python: Optional[str] = None,
) -> Optional[str]:
  """Raises unless the run config's game imports with only `shipped`.

  See the module doc.

  Args:
    train_args: the trainer's flags.
    shipped: the import names of the code packages the machine gets.
    plugins: the launcher's plugin modules (the trainer imports them too).
    fix: how to ship a package, appended to the error.
    python: the interpreter of the child (default this one).

  Returns:
    The game the run config names, or None when every code game package is
    shipped (nothing to check).

  Raises:
    GamePackageNotShippedError: the game needs an unshipped code game package.
    RuntimeError: the run config does not resolve here (its error).
  """
  blocked = unshipped(shipped)
  if not blocked:
    return None
  a = trainer_flags(train_args)
  plugin_list = [*plugins, *(p for p in a.plugins.split(",") if p)]
  payload = json.dumps({
      "framework": _FRAMEWORK,
      "blocked": list(blocked),
      "config": a.config,
      "sets": list(a.set),
      "plugins": plugin_list,
  })
  # PACKAGE_ROOT first on PYTHONPATH (child_env), in the launcher's own working
  # directory: a plugin module may be importable only from there.
  p = subprocess_env.run_python(
      ["-c", _PROBE, payload], python=python, cwd=os.getcwd(), timeout=None
  )
  if p.returncode == _NOT_SHIPPED_RC:
    got = json.loads(p.stdout.strip().splitlines()[-1])
    if got["blocked"]:
      raise GamePackageNotShippedError(
          f"config {a.config!r} (game {got['env'] or '?'}) needs "
          f"{_describe(got['package'])}, which this launch does not ship to "
          "the machine, so "
          f"the trainer would stop at startup there. {fix}".strip()
      )
    # A game package the launcher's machine lacks but the remote one may have
    # (pyspiel, installed by setup): not ours to decide.
    logging.warning(
        "preflight: config %r needs %s, which is not installed here; the game "
        "package check stopped there",
        a.config,
        _describe(got["package"]),
    )
    return got["env"]
  if p.returncode != 0:
    raise RuntimeError(
        f"preflight: config {a.config!r} does not resolve "
        f"(exit {p.returncode}): {p.stderr.strip()[-1500:]}"
    )
  return json.loads(p.stdout.strip().splitlines()[-1])["env"]


def _describe(pkg: str) -> str:
  """The display name of a games package.

  Args:
    pkg: The package.

  Returns:
    The name.
  """
  return env_registry.GAME_PACKAGES.get(pkg, pkg)

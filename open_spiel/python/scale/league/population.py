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

"""league.population: a run's final population and its exploitability.

A trained population and its meta-strategy, read from a run directory, and
the exact exploitability of the meta-mixture.

PSRO (Lanctot et al. 2017) ends with a population of policies and a
meta-strategy over them; the strategy the population represents is the
mixture that plays member k for a whole game with probability w_k. This
module loads such a population and scores the mixture exactly on games small
enough to enumerate.

    pop = population.load_population(run_dir, population.final_step(run_dir))
    population.mixture_exploitability(env, pop)  # {exploitability, ...}

The built-in reader takes a league run (`league.runner`): the
controller's last meta-strategy at the step (`PSRO.last_meta`: the
population's names and the column strategy, solved over the full population
after the last admission) and the members from the run's ArtifactStore.
`register_reader` adds readers of other run formats. The match of two
meta-mixtures is `eval.match.Evaluator.mixture_head_to_head`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Protocol

from absl import logging
from etils import epath
import numpy as np

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.ckpt import layout
from open_spiel.python.scale.ckpt import run_state
from open_spiel.python.scale.core import types
from open_spiel.python.scale.eval import exploitability as expl
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.league import artifact_store
from open_spiel.python.scale.league import commit
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import mixture

STORE_DIR = layout.STORE_DIR


@dataclasses.dataclass(frozen=True, eq=False)
class Population:
  """A population and its meta-strategy.

  Attributes:
    name: "<run dir>@<step>".
    game: the members' game.
    members: one `types.Artifact` per member, in meta-strategy order.
    member_names: their names.
    weights: f64[K], the meta-strategy (normalised to sum to 1).
    source: the reader's name.
  """

  name: str
  game: str
  members: tuple[Any, ...]
  member_names: tuple[str, ...]
  weights: np.ndarray
  source: str

  def __post_init__(self):
    members, names = tuple(self.members), tuple(self.member_names)
    w = np.asarray(self.weights, np.float64)
    if not members or len(members) != len(names):
      raise ValueError(
          f"population {self.name}: {len(members)} members, {len(names)} names"
      )
    if (
        w.shape != (len(members),)
        or not np.all(np.isfinite(w))
        or np.any(w < 0)
    ):
      raise ValueError(
          f"population {self.name}: weights {w.tolist()} do not fit "
          f"{len(members)} members (finite, >= 0)"
      )
    if abs(float(w.sum()) - 1.0) > 1e-6:
      raise ValueError(
          f"population {self.name}: weights sum to {float(w.sum())}, not 1"
      )
    for a in members:
      if a.manifest.game != self.game:
        raise ValueError(
            f"population {self.name}: member game {a.manifest.game!r} != "
            f"{self.game!r}"
        )
    object.__setattr__(self, "members", members)
    object.__setattr__(self, "member_names", names)
    object.__setattr__(self, "weights", w / w.sum())

  def entrants(self, rule: Optional[types.DecisionRule] = None) -> list[Any]:
    """The members as eval.match.Entrants named "<pop>/<member>".

    Args:
      rule: the members' decision rule; their default when None.

    Returns:
      The entrants.
    """
    return [
        match.Entrant.from_artifact(a, rule=rule, name=f"{self.name}/{n}")
        for a, n in zip(self.members, self.member_names)
    ]


class PopulationReader(Protocol):
  """Reads the population of one run-directory format.

  Attributes:
    name: the reader's name.
  """

  name: str

  def detect(self, run_dir: str) -> bool:
    """Whether run_dir is this reader's format.

    Args:
      run_dir: the run dir.

    Returns:
      True when the reader reads run_dir.
    """

  def final_step(self, run_dir: str) -> int:
    """The last step with a complete population.

    Args:
      run_dir: the run dir.

    Returns:
      The step.
    """

  def load(self, run_dir: str, step: int) -> Population:
    """The population at `step`.

    Args:
      run_dir: the run dir.
      step: the step.

    Returns:
      The Population.
    """


class LeagueReader:
  """The population of a league run (module doc).

  The population of a league run (module doc). It detects only the layout
  this package writes (train_state.json with `league_commits`); a run stored in
  a layout read through `commit.register_layout_reader` registers its own
  reader (register_reader), which may subclass this one.
  """

  name = "league"

  def detect(self, run_dir: str) -> bool:
    """Whether run_dir is a league run in this package's layout.

    Args:
      run_dir: the run dir.

    Returns:
      True when run_dir holds a train_state.json with league commits.
    """
    ts = run_state.read_train_state(run_dir)
    root = artifact.ckpt_root(run_dir)
    return (
        ts is not None
        and "league_commits" in ts
        and (root / commit.COMMITS_DIR).exists()
    )

  def final_step(self, run_dir: str) -> int:
    """The last committed step.

    Args:
      run_dir: the run dir.

    Returns:
      The step.

    Raises:
      FileNotFoundError: nothing is committed.
    """
    ts = run_state.read_train_state(run_dir)
    if ts is None:
      raise FileNotFoundError(
          f"{run_dir}: a league run with no train_state.json (nothing "
          "committed)"
      )
    return int(ts["step"])

  def load(self, run_dir: str, step: int) -> Population:
    """The population committed at `step`.

    Args:
      run_dir: the run dir.
      step: the step.

    Returns:
      The Population.

    Raises:
      FileNotFoundError: nothing is committed at `step`.
      ValueError: the controller records no meta-strategy (not a PSRO
        population).
    """
    path = commit.locate(run_dir, step).state_path
    if not path.exists():
      raise FileNotFoundError(
          f"{run_dir}: no league state at step {step} ({path})"
      )
    blob = loop.read_state(path)
    assert int(blob["iteration"]) == int(step), (blob["iteration"], step)
    meta = getattr(blob["controller"], "last_meta", None)
    if meta is None:
      raise ValueError(
          f"{run_dir}@{step}: the controller "
          f"{type(blob['controller']).__name__} "
          "records no meta-strategy (last_meta): not a PSRO population"
      )
    names, sigma = tuple(meta[0]), np.asarray(meta[1], np.float64)
    store = artifact_store.ArtifactStore(str(epath.Path(run_dir) / STORE_DIR))
    members = tuple(store.load(n) for n in names)
    game = members[0].manifest.game
    logging.info(
        "population: %s@%d (league run, %d members)",
        run_dir,
        step,
        len(members),
    )
    return Population(
        f"{run_dir}@{int(step)}", game, members, names, sigma, self.name
    )


_READERS: list[Any] = [LeagueReader()]


def register_reader(r: PopulationReader) -> None:
  """Adds a reader, tried after the readers registered before it.

  Args:
    r: the reader.
  """
  if all(x is not r for x in _READERS):
    _READERS.append(r)


def reader_of(run_dir: str) -> PopulationReader:
  """The first reader that detects run_dir.

  Args:
    run_dir: the run dir.

  Returns:
    The PopulationReader.

  Raises:
    ValueError: none does.
  """
  for r in _READERS:
    if r.detect(run_dir):
      return r
  raise ValueError(
      f"{run_dir}: no population reader recognises it (registered: "
      f"{[r.name for r in _READERS]}; league.population.register_reader adds "
      "one)"
  )


def final_step(run_dir: str) -> int:
  """The last step with a complete population.

  Args:
    run_dir: the run dir.

  Returns:
    The step.
  """
  return int(reader_of(run_dir).final_step(run_dir))


def load_population(run_dir: str, step: int) -> Population:
  """The population of run_dir at `step` (module doc).

  Args:
    run_dir: the run dir.
    step: the step.

  Returns:
    The Population.
  """
  return reader_of(run_dir).load(run_dir, int(step))


def mixture_exploitability(
    env: types.Env,
    pop: Population,
    rule: types.DecisionRule = types.SAMPLE,
    tree: Optional[expl.GameTree] = None,
) -> dict[str, Any]:
  """Exact exploitability and NashConv of a population's meta-mixture.

  Every member's policy table under `rule` is combined realization-weighted
  (`league.mixture.mixture_table`) and scored as one policy playing every
  seat. Needs a game small enough for `GameTree` (kuhn, leduc).

  Args:
    env: the population's env.
    pop: the population.
    rule: the members' decision rule.
    tree: a prebuilt GameTree of env, or None to build one.

  Returns:
    {exploitability, nash_conv, num_infosets, num_members, support}, f64.

  Raises:
    ValueError: the population's game is not env's.
  """
  if pop.game != env.name:
    raise ValueError(f"population game {pop.game!r} != env {env.name!r}")
  tree = expl.GameTree(env) if tree is None else tree
  tables = [tree.policy_table(a.policy, a.params, rule) for a in pop.members]
  mix = mixture.mixture_table(tree, tables, pop.weights)
  r = expl.table_result(tree, mix)
  return dict(
      exploitability=float(r.exploitability),
      nash_conv=float(r.nash_conv),
      num_infosets=int(r.num_infosets),
      num_members=len(pop.members),
      support=int(np.count_nonzero(pop.weights > 0.0)),
  )

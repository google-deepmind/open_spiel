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

"""The driver of a host-driven learner: numpy in, numpy out, outside the graph.

A `Capability.HOST_DRIVEN` learner (one whose update runs as host code over
numpy batches, for example a trainer in another framework) plays slot 0 of a
`HostArena` with its own policy, which is also the slot policy: what pool
members and artifacts are. A kind registers a spec whose `build_driver`
returns a `HostDriver` (or a subclass that routes its slot policy).

Its params are a handle on host-held weights that every update re-syncs, so
a pool member is a frozen copy (`freeze`), its state stays on the host
(`commit_state`), and the PBT parameter edit is refused (`mutate`).
"""

from __future__ import annotations

from typing import Any, Optional

import jax

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.league.kinds import base


class HostDriver(base.Driver):
  """The generic HOST_DRIVEN driver (module doc).

  Attributes:
    policy: the HOST_DRIVEN policy.
    learner: the host learner.
    slot_policy: the policy the learner slot plays.
    update_offset: the updates done before this run (a resume).
    num_steps: plies per update (set by attach).
    arena: the HostArena (set by attach).
    mesh: the Arena's device mesh (set by attach).
  """

  kind = "host_driven"
  algo = "host_driven"
  in_graph_learner = False

  def __init__(
      self, policy: types.Policy, learner: Any, update_offset: int = 0
  ):
    assert (
        learner.capability == types.Capability.HOST_DRIVEN
    ), learner.capability
    self.policy, self.learner = policy, learner
    self.slot_policy = policy
    self.update_offset = int(update_offset)
    self._sources: dict[str, Any] = {}
    self._owner: Optional[str] = None

  def mutate(self, src, fresh, shrink: float, perturb: float):
    """Refuses PBT's param edit: the learner params are a host handle.

    Args:
      src: the source params.
      fresh: freshly initialised params.
      shrink: the weight of src.
      perturb: the weight of fresh.

    Raises:
      NotImplementedError: always.
    """
    raise NotImplementedError(
        f"mutate: the {self.kind} kind's learner params are a handle on "
        "host-held weights; "
        "PBT's param edit (CopyMutate) needs a learner that exposes a params "
        "entry to edit"
    )

  def attach(self, league, mesh) -> None:
    """Builds the Arena over league.opponent_slots().

    Args:
      league: the League.
      mesh: the device mesh.
    """
    self.num_steps = league.config.num_steps
    self.arena = self._arena(
        league,
        mesh,
        arena_lib.Slot(self.slot_policy, 1, league.config.learner_rule),
        host=True,
    )
    self.mesh = self.arena.mesh

  def _slot0(self, ls):
    """Arena slot 0's params: the learner's (wrapped), stacked [1].

    Args:
      ls: The learner slot.

    Returns:
      The stacked params.
    """
    return base.stack1(self.wrap(self.learner.params(ls.state)))

  def train_step(self, ls, table, opponent_params, key, fresh):
    """One update of learner slot `ls`.

    Args:
      ls: the learner slot.
      table: the drawn slot table.
      opponent_params: the opponent slots' params, in slot order.
      key: the update's key.
      fresh: whether to start a fresh trajectory source.

    Returns:
      (state, metrics, frames, decisions).
    """
    self._single(ls)
    k_roll, k_upd = jax.random.split(key)
    params = [self._slot0(ls), *opponent_params]
    src = self._sources.get(ls.name)
    if src is None or fresh:
      src = arena_lib.ArenaSource(self.arena, params, table, self.num_steps)
      self._sources[ls.name] = src
    else:
      src.set_params(params)
    f0, d0 = src.frames, src.decisions
    batch = src.next(k_roll)
    state, m = self.learner.update(ls.state, batch, k_upd)
    return state, dict(m), src.frames - f0, src.decisions - d0

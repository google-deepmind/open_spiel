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

"""StubEnv: an Env identity for a game whose trajectories come from outside.

The package.

Behaviour cloning on data produced by another engine (an external labeller,
a different simulator) needs the parts of an Env that the data path reads,
but no dynamics. A StubEnv carries the identity (name, num_players,
num_actions, max_len, zero_sum) and the InfoState template, one seat's
InfoState as ShapeDtypeStructs, which sources.shards uses to rebuild and
check every row. Every dynamics method raises, so no Arena can play it.

  env = StubEnv("my_game", num_players=2, num_actions=10, max_len=50,
                info_like={"obs": np.zeros(32, np.float32)})
  reader = shards.ShardReader(path, env, rules_sha=engine_rules_sha, ...)

The data supplier provides the template and the rules hash of its engine
(`rules_sha`, which envs.registry.rules_sha returns for the stub);
ShardReader checks both on every shard.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import jax
import numpy as np


class StubEnv:
  """An Env whose dynamics raise (module doc).

  Attributes:
    stub_rules_sha: The rules identity it was given, or None (unknown).
    name: The game name.
    num_players: The number of seats.
    num_actions: The size of the action space.
    max_len: The longest game, in decisions.
    zero_sum: Whether the game is zero-sum.
  """

  max_chance_outcomes = 0

  def __init__(
      self,
      name: str,
      num_players: int,
      num_actions: int,
      max_len: int,
      info_like: Any,
      zero_sum: bool = True,
      rules_sha: Optional[str] = None,
  ):
    """Builds the stub env.

    Args:
      name: The game name.
      num_players: The number of seats.
      num_actions: The size of the action space.
      max_len: The longest game, in decisions.
      info_like: One seat's InfoState, or anything with the same structure,
        shapes and dtypes (arrays or ShapeDtypeStructs).
      zero_sum: Whether the game is zero-sum.
      rules_sha: The rules identity of the engine that produces the rows
        (registry.rules_sha of this env); None: unknown (rules_sha raises).

    Raises:
      ValueError: info_like has no leaves, the name or rules_sha is empty, or
        a count is below 1.
    """
    if not isinstance(name, str) or not name:
      raise ValueError(f"a stub env needs a non-empty name, got {name!r}")
    if rules_sha is not None and (
        not isinstance(rules_sha, str) or not rules_sha
    ):
      raise ValueError(
          f"rules_sha is a non-empty string or None, got {rules_sha!r}"
      )
    self.stub_rules_sha = rules_sha
    if not (num_players >= 1 and num_actions >= 1 and max_len >= 1):
      raise ValueError(
          "num_players, num_actions and max_len must be >= 1, got "
          f"{num_players}, {num_actions}, {max_len}"
      )
    self.name, self.num_players, self.num_actions, self.max_len = (
        name,
        int(num_players),
        int(num_actions),
        int(max_len),
    )
    self.zero_sum = bool(zero_sum)

    def sds(x):
      """The ShapeDtypeStruct of a template leaf.

      Args:
        x: An array-like leaf.

      Returns:
        The ShapeDtypeStruct.
      """
      if hasattr(x, "shape") and hasattr(x, "dtype"):
        return jax.ShapeDtypeStruct(tuple(x.shape), x.dtype)
      a = np.asarray(x)
      return jax.ShapeDtypeStruct(a.shape, a.dtype)

    self._template = jax.tree_util.tree_map(sds, info_like)
    if not jax.tree_util.tree_leaves(self._template):
      raise ValueError("a stub env's InfoState template has no leaves")

  def info_template(self) -> Any:
    """The InfoState template.

    Returns:
      One seat's InfoState as ShapeDtypeStructs.
    """
    return self._template

  def _no(self, what: str):
    raise NotImplementedError(
        f"{self.name} is a stub env (envs.stub): {what} has no dynamics "
        "here; its trajectories come from shards"
    )

  def root(self):
    """The game's initial state, chance unresolved."""
    self._no("root")

  def apply(self, s, a):
    """Applies player action `a`; chance stays unresolved.

    Args:
      s: The state.
      a: The player action id.
    """
    del s, a  # Unused.
    self._no("apply")

  def is_chance(self, s):
    """Whether `s` is a chance node.

    Args:
      s: The state.
    """
    del s  # Unused.
    self._no("is_chance")

  def apply_chance(self, s, outcome):
    """Applies a known chance outcome at a chance node.

    Args:
      s: The state.
      outcome: The chance outcome id.
    """
    del s, outcome  # Unused.
    self._no("apply_chance")

  def init(self, deal_key):
    """The root with every leading chance node resolved.

    Args:
      deal_key: PRNG key that resolves the leading chance nodes.
    """
    del deal_key  # Unused.
    self._no("init")

  def step(self, s, a, chance_key):
    """Applies `a`, then resolves every chance node that follows.

    Args:
      s: The state.
      a: The player action id.
      chance_key: PRNG key that resolves the chance nodes after the action.
    """
    del s, a, chance_key  # Unused.
    self._no("step")

  def chance_probs(self, s):
    """The outcome probabilities at a chance node.

    Args:
      s: The state.
    """
    del s  # Unused.
    self._no("chance_probs")

  def actor(self, s):
    """The seat to move.

    Args:
      s: The state.
    """
    del s  # Unused.
    self._no("actor")

  def legal_mask(self, s):
    """The actor's legal actions.

    Args:
      s: The state.
    """
    del s  # Unused.
    self._no("legal_mask")

  def terminal(self, s):
    """Whether the game is over at `s`.

    Args:
      s: The state.
    """
    del s  # Unused.
    self._no("terminal")

  def returns(self, s):
    """Per-seat returns.

    Args:
      s: The state.
    """
    del s  # Unused.
    self._no("returns")

  def info_state(self, s, seat):
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).
    """
    del s, seat  # Unused.
    self._no("info_state")

  def labels(self, s, seat) -> Mapping[str, Any]:
    """Privileged targets for learners (e.g. the opponent's hand).

    Args:
      s: The state.
      seat: The seat (i32).
    """
    del s, seat  # Unused.
    self._no("labels")

  def visible_action(self, s, a, seat):
    """What `seat` learns of action `a` taken at state `s`.

    Args:
      s: The state.
      a: The player action id.
      seat: The seat (i32).
    """
    del s, a, seat  # Unused.
    self._no("visible_action")

  def world_from_info(self, info) -> Optional[Any]:
    """The one state consistent with `info`, when it determines it.

    Args:
      info: One seat's InfoState.
    """
    del info
    return None

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

"""The expert-iteration learner kind (ExIt; Anthony, Tian and Barber 2017).

A base network is trained by imitation on the decisions of an operator built
over it, typically a search: the operator acts (or labels while the base
acts), its policy and value targets go into a replay, and the base learns
them by supervised steps. AlphaZero is the special case with the default
operator (`learners.az`'s SearchPolicy) and the outcome value targets; that
case is `kinds.AZ`.

    spec = kinds.ExIt(az=az.AZConfig(...), updates_per_generation=50,
                      behavior="mixture(0.5)")

Generations. With `updates_per_generation=N` the acting slot plays a frozen
snapshot of the base for N updates (a generation), the snapshot the
`ScheduledPromotion` controller admits every N updates, and the replay is
cleared at each generation boundary. Without it the slot plays the current
params and the replay is never cleared.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Mapping, Optional

from open_spiel.python.scale.league import refs
from open_spiel.python.scale.league import siblings as siblings_lib
from open_spiel.python.scale.league import slot_factories
from open_spiel.python.scale.league.kinds import base
from open_spiel.python.scale.learners import az as az_lib
from open_spiel.python.scale.learners import bc
from open_spiel.python.scale.policies import operators


@dataclasses.dataclass(frozen=True)
class ExIt:
  """Expert iteration over the base network of `az` (module doc).

  Attributes:
    az: the AZConfig: network, MCTS, replay, BC steps, run length.
    factory: the operator over the base network
      (`league.slot_factories`). "" (the default) is the AZConfig's
      SearchPolicy; the explicit name "az_search" is normalised to "", so the
      two spellings are one config with one hash. Built once; each generation
      swaps params only.
    factory_kwargs: the factory's keyword arguments.
    updates_per_generation: None, or N (module doc).
    behavior: "search" (the operator acts), "plain" or "mixture(<beta>)" (the
      base acts with probability 1 - beta while the operator labels;
      `policies.operators`). A labelling slot is never published or pooled.
    host: play in the HostArena (a HOST_DRIVEN operator; the learner stays
      the in-graph BCLearner).
    plain_members: pool members are the plain base network, not the operator.
    publish_plain: every checkpoint writes the plain network at <ckpt>/plain
      first, then the operator at <ckpt>/search.
    policy_mask: BCConfig.policy_mask.
    value_lambda: 0 keeps the Monte Carlo outcome value targets; in (0, 1]
      mixes in the operator's root values (`sources.targets`). Refused for an
      operator whose values are a fused upper bound (PIMC).
    q_softmax_temperature: None keeps the operator's policy target; T uses
      softmax(Q / T) over the legal actions.
    value_scale: the base value head's units, written into every artifact
      from this config; it must equal the operator's and the init's.
    value_link: the base value head's link (`league.siblings.VALUE_LINKS`),
      written into every artifact from this config.
    init: generation 0 warm-started from a PolicyRef (its plain network when
      the artifact is routed or a search; fresh optimizer).
    parents: the init's artifact ids, named as parents.
    kind: the kind's registered name.
  """

  az: Any  # learners.az.AZConfig
  factory: str = ""
  factory_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  updates_per_generation: Optional[int] = None
  behavior: str = "search"
  host: bool = False
  plain_members: bool = False
  publish_plain: bool = False
  policy_mask: str = "valid"
  value_lambda: float = 0.0
  q_softmax_temperature: Optional[float] = None
  value_scale: Optional[float] = None
  value_link: Optional[str] = None
  init: Optional[refs.PolicyRef] = None
  parents: tuple[Any, ...] = ()
  kind: str = "expert_iteration"

  def __post_init__(self):
    assert self.kind == "expert_iteration", self.kind
    assert isinstance(self.az, az_lib.AZConfig), type(self.az)
    json.dumps(dict(self.factory_kwargs), sort_keys=True)

    if self.factory == slot_factories.DEFAULT:
      object.__setattr__(self, "factory", "")
    if (
        self.updates_per_generation is not None
        and int(self.updates_per_generation) < 1
    ):
      raise ValueError(
          "updates_per_generation must be >= 1, got "
          f"{self.updates_per_generation}"
      )
    operators.parse_behavior(self.behavior)
    if self.policy_mask not in bc.POLICY_MASKS:
      raise ValueError(
          f"policy_mask {self.policy_mask!r} not in {bc.POLICY_MASKS}"
      )
    if not 0.0 <= float(self.value_lambda) <= 1.0:
      raise ValueError(
          f"value_lambda must be in [0, 1], got {self.value_lambda}"
      )
    if self.value_lambda > 0.0 and self.value_scale is None:
      raise ValueError(
          "value_lambda > 0 mixes search values in value-head units: set "
          "value_scale"
      )
    if (
        self.q_softmax_temperature is not None
        # NaN-safe: `not x > 0` also refuses NaN, which `x <= 0` would accept.
        # pylint: disable-next=unnecessary-negation
        and not float(self.q_softmax_temperature) > 0.0
    ):
      raise ValueError(
          f"q_softmax_temperature must be > 0, got {self.q_softmax_temperature}"
      )
    # NaN-safe: `not x > 0` also refuses NaN, which `x <= 0` would accept.
    # pylint: disable-next=unnecessary-negation
    if self.value_scale is not None and not float(self.value_scale) > 0.0:
      raise ValueError(f"value_scale must be > 0, got {self.value_scale}")
    if self.value_link is not None:
      siblings_lib.check_value_link(self.value_link)
    assert self.init is None or isinstance(self.init, refs.PolicyRef), type(
        self.init
    )
    assert all(
        isinstance(p, str) and "@" in p for p in self.parents
    ), self.parents
    if self.parents and self.init is None:
      raise ValueError("parents name the warm start: set init")

  def constant_lr(self) -> bool:
    """Whether the step size is independent of the run length.

    Returns:
      True when the AZConfig's learning rate does not anneal.
    """
    return self.az.lr_end == self.az.lr

  def horizon(self) -> int:
    """The run length in updates.

    Returns:
      The AZConfig's num_iterations.
    """
    return int(self.az.num_iterations)

  def fixed_horizon(self) -> int:
    """The run length the schedule is fixed to.

    Returns:
      `horizon()`.
    """
    return self.horizon()

  def with_run_updates(self, n: int) -> "ExIt":
    """This spec with the run length set to n updates.

    Args:
      n: the run length in updates.

    Returns:
      The extended spec.
    """
    return dataclasses.replace(
        self, az=dataclasses.replace(self.az, num_iterations=n)
    )

  def validate_run(self, run: Any) -> None:
    """Checks this spec against the run config.

    Args:
      run: the run config.

    Raises:
      ValueError: the spec and the run disagree.
    """
    check_az_run(self.az, run)
    if self.updates_per_generation is None:
      if run.league.self_play_seats != "all":
        raise ValueError(
            "self-play stores every seat's decisions: self_play_seats='all'"
        )
    elif run.league.self_play_seats != "one":
      raise ValueError(
          "a generation (updates_per_generation) plays the round's mix (the "
          "frozen "
          "snapshot, the pool): self_play_seats='one'"
      )


def check_az_run(azc: Any, run: Any) -> None:
  """Raises when the AZConfig's run-level fields disagree with the run's.

  Args:
    azc: the AZConfig.
    run: the run config.

  Raises:
    ValueError: env_name, batch_size or seed differ from the run's.
  """
  for what, got, want in (
      ("env_name", azc.env_name, run.env),
      ("batch_size", azc.batch_size, run.league.num_envs),
      ("seed", azc.seed, run.seed),
  ):
    if got != want:
      raise ValueError(f"az.{what} {got!r} != the league run's {want!r}")


class AZFamilyKind(base.KindBase):
  """What the AZ and ExIt kinds share.

  What the AZ and ExIt kinds share: the AZConfig carries the run's batch
  rows, seed and length, and one update plays one whole game per row.
  """

  fixed_num_steps = True  # an update's plies are one whole game: max_len + 1

  def with_batch(self, cfg: Any, num_envs: int) -> Any:
    """Cfg with the AZConfig's batch size set to the league's rows.

    Args:
      cfg: the kind's spec.
      num_envs: the league's batch rows.

    Returns:
      The resized spec.
    """
    return dataclasses.replace(
        cfg, az=dataclasses.replace(cfg.az, batch_size=int(num_envs))
    )

  def with_seed(self, cfg: Any, seed: int) -> Any:
    """Cfg with the AZConfig's seed set.

    Args:
      cfg: the kind's spec.
      seed: the run's seed.

    Returns:
      The reseeded spec.
    """
    return dataclasses.replace(
        cfg, az=dataclasses.replace(cfg.az, seed=int(seed))
    )

  def run_length_paths(self) -> tuple[str, ...]:
    """The run-length field: az.num_iterations.

    Returns:
      The paths.
    """
    return ("az.num_iterations",)

  def build_driver(
      self,
      cfg: Any,
      env,
      *,
      policy_arch: str = "network",
      policy_kwargs: Optional[Mapping[str, Any]] = None,
      data_dir: Optional[str] = None,
      scratch_dir: Optional[str] = None,
  ):
    """The AZDriver of a spec over `env`.

    Args:
      cfg: the kind's spec.
      env: the league's env.
      policy_arch: the registered architecture of the learner's policy.
      policy_kwargs: the architecture's kwargs.
      data_dir: must be None (a BC option).
      scratch_dir: unused.

    Returns:
      The az.AZDriver.

    Raises:
      ValueError: data_dir is given.
    """
    # Lazy: cycle: kinds.az imports this module.
    from open_spiel.python.scale.league.kinds import az  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    del scratch_dir
    if data_dir is not None:
      raise ValueError("data_dir is a BC option")
    return az.AZDriver(cfg, env, policy_kwargs, policy_arch=policy_arch)


class ExItKind(AZFamilyKind):
  """The expert-iteration learner kind."""

  name = "expert_iteration"
  spec_cls = ExIt


KIND = ExItKind()
base.register_kind(KIND)

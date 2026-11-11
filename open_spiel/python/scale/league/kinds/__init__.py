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

"""Learner kinds: what a league trains and how one update runs.

Every run is a league, and a league trains one learner kind. The kind's spec
sits in the run config (`LeagueRunConfig.learner`); its `LearnerKind` object
builds the driver the League calls once per update (`kinds.base`).

  PG     policy gradient (PPO, MMD, registered losses); the one kind that
         trains several learners at once (PSRO, PBT).       `kinds.pg`
  BC     behaviour cloning from a teacher, online (DAgger) or from shards.
                                                            `kinds.bc`
  AZ     AlphaZero: search self-play, replay, supervised steps. `kinds.az`
  ExIt   expert iteration: AZ generalised to any operator over the base
         network, generations, mixed behaviour, search value targets.
                                                  `kinds.expert_iteration`

A new kind is a spec dataclass with a `kind` name plus a `LearnerKind` object
passed to `register_kind`; a host-driven learner's driver is usually a
`HostDriver` (`kinds.host`).
"""

from open_spiel.python.scale.league.kinds.az import AZ
from open_spiel.python.scale.league.kinds.az import AZDriver
from open_spiel.python.scale.league.kinds.az import plain_network
from open_spiel.python.scale.league.kinds.base import controllers_of
from open_spiel.python.scale.league.kinds.base import Driver
from open_spiel.python.scale.league.kinds.base import get
from open_spiel.python.scale.league.kinds.base import KindBase
from open_spiel.python.scale.league.kinds.base import LearnerKind
from open_spiel.python.scale.league.kinds.base import lineage_extra
from open_spiel.python.scale.league.kinds.base import make_driver
from open_spiel.python.scale.league.kinds.base import names
from open_spiel.python.scale.league.kinds.base import of
from open_spiel.python.scale.league.kinds.base import register_kind
from open_spiel.python.scale.league.kinds.base import spec_classes
from open_spiel.python.scale.league.kinds.base import stack1
from open_spiel.python.scale.league.kinds.bc import BC
from open_spiel.python.scale.league.kinds.bc import BCDriver
from open_spiel.python.scale.league.kinds.bc import INFO_SCHEMA_VERSION
from open_spiel.python.scale.league.kinds.expert_iteration import ExIt
from open_spiel.python.scale.league.kinds.host import HostDriver
from open_spiel.python.scale.league.kinds.pg import PG
from open_spiel.python.scale.league.kinds.pg import PGDriver

__all__ = [
    "AZ",
    "AZDriver",
    "BC",
    "BCDriver",
    "controllers_of",
    "Driver",
    "ExIt",
    "get",
    "HostDriver",
    "INFO_SCHEMA_VERSION",
    "KindBase",
    "LearnerKind",
    "lineage_extra",
    "make_driver",
    "names",
    "of",
    "PG",
    "PGDriver",
    "plain_network",
    "register_kind",
    "spec_classes",
    "stack1",
]

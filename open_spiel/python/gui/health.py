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

"""The health body: what code the server runs and what it can do."""

import dataclasses

from open_spiel.python.gui import contract


@dataclasses.dataclass(frozen=True)
class Capabilities:
  """What the server can do, independent of any one game.

  Attributes:
    controllers: Every registered controller, starting with the human one.
    chance_policies: The chance policies the server accepts.
    python_game_search: Whether search controllers can play games written in
      Python; that needs a pyspiel whose Python states clone safely.
  """

  controllers: tuple[contract.Controller, ...]
  chance_policies: tuple[contract.ChancePolicy, ...]
  python_game_search: bool

  def __post_init__(self) -> None:
    contract.require(
        bool(self.controllers) and self.controllers[0].name == contract.HUMAN,
        "The human controller is listed first.",
    )
    contract.require(
        bool(self.chance_policies), "The server accepts a chance policy."
    )


@dataclasses.dataclass(frozen=True)
class Health:
  """The server's identity and capabilities.

  Attributes:
    contract_version: The contract version the server speaks.
    version: The installed OpenSpiel package's version.
    revision: The source revision the server runs from, or null when unknown.
    capabilities: What the server can do.
  """

  contract_version: contract.ContractVersion = dataclasses.field(
      default=contract.CONTRACT_VERSION, kw_only=True
  )
  version: str
  revision: str | None
  capabilities: Capabilities

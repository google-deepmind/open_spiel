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


"""The interface of a game adapter, which draws a position for one viewer.

An adapter turns a state into a JSON view for one viewer, holding only what
that viewer may see, and may describe the legal actions so that its client
renderer can map a press to an action. `presenters` registers adapters and
builds a frame's presentation from them.
"""

import abc
from collections.abc import Mapping, Sequence
import dataclasses

import pyspiel

from open_spiel.python.gui import contract


@dataclasses.dataclass(frozen=True)
class Viewpoint:
  """Who an adapter presents a position to.

  Attributes:
    seat: The viewer's seat; None for the oracle and for a spectator.
    omniscient: Whether the viewer may see the full state.
  """

  seat: int | None
  omniscient: bool

  def sees_private(self, seat: int) -> bool:
    """Whether the viewer may see `seat`'s private information."""
    return self.omniscient or self.seat == seat


class Adapter(abc.ABC):
  """Presents one game's positions to its client renderer.

  Attributes:
    renderer: The name of the client renderer that draws the view.
  """

  renderer: str

  @abc.abstractmethod
  def present(
      self, state: pyspiel.State, viewpoint: Viewpoint
  ) -> dict[str, contract.JsonValue]:
    """Returns the view of `state` for `viewpoint`, as JSON-safe values.

    The view holds nothing the viewer may not see.
    """

  def describe_actions(
      self,
      state: pyspiel.State,
      actions: Sequence[int],
      viewpoint: Viewpoint,
  ) -> Mapping[int, contract.JsonValue]:
    """Returns a hint for each of `actions` the renderer can place.

    Called only when the viewer is given the legal actions, which are the
    acting seat's. An action left out has no hint; the client still offers
    it in its list of legal actions.
    """
    del state, actions, viewpoint
    return {}

  # The default hides every action; None is its answer, not a fall-through.
  # pylint: disable-next=useless-return
  def public_action_label(
      self, state: pyspiel.State, action: int
  ) -> str | None:
    """Returns what a viewer who may not see `action` may read of it.

    Called with the state before `action` (an action or a chance outcome) is
    applied. The label is shown in place of the withheld action to every
    viewer but the seat that acted and an omniscient viewer, so it names only
    what the rules make public: a card discarded face up, a draw from a
    face-up pile. None, the default, keeps the action hidden.
    """
    del state, action
    return None

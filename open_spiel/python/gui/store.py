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


"""An in-memory store of sessions, shared by the server's threads.

The store's lock guards only which sessions exist; each session serialises
its own changes, so requests to different sessions never wait on each other.

Every new game a client sets up is a new session, and no client says when it
is done with one, so the store keeps a bounded number: past its capacity, a
new session displaces the one used least recently.
"""

import collections
from collections.abc import Mapping, Sequence
import threading

from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers as controllers_lib
from open_spiel.python.gui import errors
from open_spiel.python.gui import session as session_lib


# The most sessions a store keeps, far more than the tabs one person opens.
DEFAULT_CAPACITY = 256


class SessionStore:
  """The sessions a server holds, keyed by id, least recently used first."""

  def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
    """Starts an empty store that keeps at most `capacity` sessions."""
    if capacity < 1:
      raise ValueError(f"A store keeps at least one session, not {capacity}.")
    self._capacity = capacity
    self._sessions: collections.OrderedDict[str, session_lib.Session] = (
        collections.OrderedDict()
    )
    self._lock = threading.Lock()

  def create(
      self,
      game_name: str,
      parameters: Mapping[str, contract.ParameterValue],
      *,
      seed: int | None = None,
      chance_policy: contract.ChancePolicy,
      controllers: Sequence[controllers_lib.Spec],
      autoplay: Sequence[bool] = (),
  ) -> session_lib.Session:
    """Starts a session and stores it; see `session.Session.create`.

    When the store is full, the session used least recently is forgotten.

    Args:
      game_name: A registered short name.
      parameters: Game parameters; omitted ones take their defaults.
      seed: Seeds every chance outcome and controller; None: a random seed.
      chance_policy: How chance nodes are resolved.
      controllers: Who plays each seat, in seat order.
      autoplay: Whether each seat's controller plays as soon as it is to move,
        in seat order; empty for none.

    Returns:
      The session started.

    Raises:
      errors.RequestError: If the session cannot be started.
    """
    session = session_lib.Session.create(
        game_name,
        parameters,
        seed=seed,
        chance_policy=chance_policy,
        controllers=controllers,
        autoplay=autoplay,
    )
    with self._lock:
      self._sessions[session.id] = session
      while len(self._sessions) > self._capacity:
        self._sessions.popitem(last=False)
    return session

  def get(self, session_id: str) -> session_lib.Session:
    """Returns the session with id `session_id`.

    Args:
      session_id: The id of the session.

    Raises:
      errors.NotFoundError: If there is no such session, or it was forgotten
        to make room for newer ones.
    """
    with self._lock:
      session = self._sessions.get(session_id)
      if session is not None:
        self._sessions.move_to_end(session_id)
    if session is None:
      raise errors.NotFoundError(f"There is no session {session_id!r}.")
    return session

  def delete(self, session_id: str) -> None:
    """Forgets the session with id `session_id`.

    Args:
      session_id: The id of the session.

    Raises:
      errors.NotFoundError: If there is no such session.
    """
    with self._lock:
      if self._sessions.pop(session_id, None) is None:
        raise errors.NotFoundError(f"There is no session {session_id!r}.")

  def __len__(self) -> int:
    with self._lock:
      return len(self._sessions)

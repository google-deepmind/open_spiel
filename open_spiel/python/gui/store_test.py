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


"""Tests for the session store."""

from concurrent import futures

from absl.testing import absltest

from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import errors
from open_spiel.python.gui import store as store_lib


def _create(store: store_lib.SessionStore) -> str:
  """Starts a tic-tac-toe session in `store` and returns its id."""
  return store.create(
      "tic_tac_toe",
      {},
      chance_policy=contract.ChancePolicy.AUTO,
      controllers=(controllers.HUMAN_SPEC, controllers.HUMAN_SPEC),
  ).id


class SessionStoreTest(absltest.TestCase):

  def test_returns_the_session_it_created(self) -> None:
    store = store_lib.SessionStore()
    session_id = _create(store)
    self.assertEqual(store.get(session_id).id, session_id)

  def test_an_unknown_id_is_not_found(self) -> None:
    with self.assertRaises(errors.NotFoundError):
      store_lib.SessionStore().get("missing")

  def test_deleted_sessions_are_gone(self) -> None:
    store = store_lib.SessionStore()
    session_id = _create(store)
    store.delete(session_id)
    self.assertEmpty(store)
    with self.assertRaises(errors.NotFoundError):
      store.delete(session_id)

  def test_a_full_store_forgets_the_session_used_least_recently(self) -> None:
    store = store_lib.SessionStore(capacity=2)
    first, second = _create(store), _create(store)
    store.get(first)
    third = _create(store)
    self.assertLen(store, 2)
    self.assertEqual(store.get(first).id, first)
    self.assertEqual(store.get(third).id, third)
    with self.assertRaises(errors.NotFoundError):
      store.get(second)

  def test_refuses_a_capacity_below_one(self) -> None:
    with self.assertRaises(ValueError):
      store_lib.SessionStore(capacity=0)

  def test_concurrent_creates_keep_every_session(self) -> None:
    store = store_lib.SessionStore()
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
      ids = list(pool.map(lambda _: _create(store), range(32)))
    self.assertLen(set(ids), 32)
    self.assertLen(store, 32)


if __name__ == "__main__":
  absltest.main()

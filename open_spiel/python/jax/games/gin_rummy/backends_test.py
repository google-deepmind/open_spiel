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

"""The backend building blocks give the reference game with the defaults."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games.gin_rummy import backends
from open_spiel.python.jax.games.gin_rummy import dense_cache
from open_spiel.python.jax.games.gin_rummy import game
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    PHASE_GAME_OVER,
)

_STEPS = 60


def _assert_states_equal(test, a, b, where):
  test.assertEqual(sorted(a), sorted(b), where)
  for k in a:
    np.testing.assert_array_equal(
        np.asarray(a[k]), np.asarray(b[k]), err_msg=f"{where}: {k}"
    )


def _first_legal(state):
  """The first chance outcome or legal action.

  Args:
    state: A state that is not terminal.

  Returns:
    i32[]: the action.
  """
  if bool(game.is_chance_node(state)):
    mask = game.chance_outcomes_mask(state)
  else:
    mask = game.legal_actions_mask(state)
  return jnp.argmax(mask).astype(jnp.int32)


class BackendsTest(absltest.TestCase):

  def test_defaults_deal_and_step_as_the_reference(self):
    """Defaults: the dealt state and a rollout equal game's, step by step."""
    init = jax.jit(backends.make_init_dealt_state())
    step = jax.jit(backends.make_step())
    for seed, oklahoma in ((0, False), (1, True)):
      key = jax.random.PRNGKey(seed)
      ours, ours_key = init(key, oklahoma=oklahoma)
      ref, ref_key = game.init_dealt_state(key, oklahoma=oklahoma)
      _assert_states_equal(self, ours, ref, f"deal seed={seed}")
      np.testing.assert_array_equal(np.asarray(ours_key), np.asarray(ref_key))
      for t in range(_STEPS):
        if int(ref["phase"]) == PHASE_GAME_OVER:
          break
        a = _first_legal(ref)
        ours, ref = step(ours, a), game.step(ref, a)
        _assert_states_equal(self, ours, ref, f"seed={seed} step {t}")

  def test_supplied_meld_functions_are_used(self):
    """The functions given are the ones the step calls."""
    calls = []

    def meld_info(hand):
      """The dense meld function, recording the call.

      Args:
        hand: The hand.

      Returns:
        What the dense meld function returns.
      """
      calls.append("meld_info")
      return dense_cache.compute_all_meld_info_dense_discard_only(hand)

    state, _ = game.init_dealt_state(jax.random.PRNGKey(3))
    backends.make_step(meld_info=meld_info)(state, _first_legal(state))
    self.assertIn("meld_info", calls)

  def test_not_callable_raises(self):
    """A meld function that is not callable raises TypeError."""
    with self.assertRaisesRegex(TypeError, "meld_info must be callable"):
      backends.make_step(meld_info=None)
    with self.assertRaisesRegex(TypeError, "knocker_melds must be callable"):
      backends.make_step(knocker_melds=3)
    with self.assertRaisesRegex(TypeError, "meld_info must be callable"):
      backends.make_init_dealt_state(meld_info="no")

  def test_arguments_are_keyword_only(self):
    """The meld functions are keyword-only, so a call names what it swaps."""
    with self.assertRaises(TypeError):
      backends.make_step(dense_cache.compute_all_meld_info_dense_discard_only)  # pylint: disable=too-many-function-args


if __name__ == "__main__":
  absltest.main()

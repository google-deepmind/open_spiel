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

"""Which auction view the installed pyspiel's bridge has.

OpenSpiel's current bridge observation seats auction calls relative to the
dealer; releases up to 2.0.2 seat them as if North always dealt. The JAX
bridge is dealer-relative, so it agrees with an earlier pyspiel only
on boards North deals. The equivalence tests compare the other dealers only
against a dealer-relative pyspiel.
"""

import functools

from open_spiel.python.jax.games import bridge

SKIP_REASON = "pyspiel predates OpenSpiel's dealer-relative bridge observations"


@functools.cache
def pyspiel_auction_view_is_dealer_relative() -> bool:
  """Whether the installed pyspiel seats bridge calls relative to the dealer.

  East deals and passes; East's own view shows that pass at slot 0 (its own
  seat) when the view is dealer-relative, and at slot 3 (where North's first
  call would sit) when it is laid out as if North dealt.

  Returns:
    True when the auction view is dealer-relative.

  Raises:
    AssertionError: the pass is at neither slot (an unknown layout).
  """
  # Lazy: pyspiel is an optional dependency of this package.
  import pyspiel  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  game = pyspiel.load_game("bridge(use_double_dummy_result=false,dealer=1)")
  state = game.new_initial_state()
  for card in range(bridge.NUM_CARDS):
    state.apply_action(card)
  state.apply_action(bridge.PASS_ACTION)
  obs = state.observation_tensor(1)
  start = bridge.NUM_OBSERVATION_TYPES + 4
  opening = [int(v) for v in obs[start : start + 4]]
  assert opening in ([1, 0, 0, 0], [0, 0, 0, 1]), opening
  return opening == [1, 0, 0, 0]

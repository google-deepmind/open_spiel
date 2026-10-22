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


"""Whether a game hides information from its players.

A game's type declares its information, but a few games that declare perfect
information keep state a player must not see. Both the loader, which decides
what each viewer is shown, and the controllers, which may read the full state,
decide from `hides_information`.
"""

import pyspiel

_PERFECT = pyspiel.GameType.Information.PERFECT_INFORMATION

# Games that declare perfect information although their state holds something
# a player must not see. They are served as hiding information, and their
# information state strings, which print the hidden part, are withheld.
HIDDEN_DESPITE_DECLARATION = frozenset({
    "python_hangman",  # The state holds the word the guesser looks for.
})


def hides_information(game_type: pyspiel.GameType) -> bool:
  """Whether a game of this type may hide part of its state from a player.

  Args:
    game_type: The type of a game as it is served. OpenSpiel's turn-based
      wrapper declares imperfect information, since it hides the moves made
      simultaneously, so a wrapped game hides information.

  Returns:
    True if the type does not declare perfect information or its game is listed
    in `HIDDEN_DESPITE_DECLARATION`.
  """
  return (
      game_type.information != _PERFECT
      or game_type.short_name in HIDDEN_DESPITE_DECLARATION
  )

# Copyright 2022 DeepMind Technologies Limited
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
"""Mean Field Crowd Modelling Game in 2d.

Please see the C++ implementation under games/mfg/crowd_modelling_2d.h for
more information.
"""

from typing import Sequence


def grid_to_forbidden_states(grid: Sequence[str]) -> str:
  """Converts a grid into string representation of forbidden states.

  Args:
    grid: Rows of the grid. '#' character denotes a forbidden state. All rows
      should have the same number of columns, i.e. cells.

  Returns:
    String representation of forbidden states in the form of x (column) and y
    (row) pairs, e.g. [1|1;0|2].
  """
  forbidden_states = []
  num_cols = len(grid[0])
  for y, row in enumerate(grid):
    assert len(row) == num_cols, f'Number of columns should be {num_cols}.'
    for x, cell in enumerate(row):
      if cell == '#':
        forbidden_states.append(f'{x}|{y}')
  return '[' + ';'.join(forbidden_states) + ']'


FOUR_ROOMS_FORBIDDEN_STATES = grid_to_forbidden_states([
    '#############',
    '#     #     #',
    '#     #     #',
    '#           #',
    '#     #     #',
    '#     #     #',
    '### ##### ###',
    '#     #     #',
    '#     #     #',
    '#           #',
    '#     #     #',
    '#     #     #',
    '#############',
])

# Four rooms with an initial state at top-left corner.
FOUR_ROOMS = {
    'forbidden_states': FOUR_ROOMS_FORBIDDEN_STATES,
    'horizon': 40,
    'initial_distribution': '[1|1]',
    'initial_distribution_value': '[1.0]',
    'size': 13,
}

MAZE_FORBIDDEN_STATES = grid_to_forbidden_states([
    '######################',
    '#      #     #     # #',
    '#      #     #     # #',
    '######    #  # ##  # #',
    '#         #  # #   # #',
    '#         #  # ### # #',
    '#  ########  #   #   #',
    '#    # # #  ##   #   #',
    '#    # # #     # # ###',
    '#    # # #     # # # #',
    '###### # ####### # # #',
    '#  #         #   # # #',
    '#  # ## ###  #   # # #',
    '## # #    #  ##### # #',
    '## # # #  #      # # #',
    '#    # ####        # #',
    '# ####  # ########   #',
    '#       #  #   # ### #',
    '#  #  # #  # # #   # #',
    '# ##### #    # #     #',
    '#            #       #',
    '######################',
])

# 22x22 maze with an initial state at top-left corner,
MAZE = {
    'forbidden_states': MAZE_FORBIDDEN_STATES,
    'horizon': 100,
    'initial_distribution': '[1|1]',
    'initial_distribution_value': '[1.0]',
    'size': 22,
}

# Copyright 2023 DeepMind Technologies Limited
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

"""Utils for constructing strings in color."""

RESET = '\033[0m'         # Reset
BLACK = '\033[30m'        # Black
RED = '\033[31m'          # Red -- Terminating Game
GREEN = '\033[32m'        # Green -- Computing Payoffs
YELLOW = '\033[33m'       # Yellow -- Generated Game Def
BLUE = '\033[34m'         # Blue
PURPLE = '\033[35m'       # Purple -- Information States
CYAN = '\033[36m'         # Cyan -- Generating Lists
WHITE = '\033[37m'        # White
BLACK2 = '\033[39m'       # Black?


class ColorText:
  """Color text class."""

  def __init__(self, reset_color=RESET):
    self.reset_color = reset_color
    self.current_color = reset_color

  def set_color(self, color: str):
    self.current_color = color

  def set_reset_color(self, color: str):
    self.reset_color = color

  def reset(self):
    self.current_color = self.reset_color

  def color(self, log_str: str, color: str = ''):
    c = color if color else self.current_color
    log_str = c + log_str + self.reset_color
    return log_str

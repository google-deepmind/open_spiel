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

"""Checks a Python library of OpenSpiel, with its pip dependencies (numpy)."""

from open_spiel.python import rl_environment


def main() -> None:
  env = rl_environment.Environment("tic_tac_toe")
  time_step = env.reset()
  while not time_step.last():
    player = time_step.observations["current_player"]
    action = time_step.observations["legal_actions"][player][0]
    time_step = env.step([action])
  assert time_step.rewards == [1.0, -1.0], time_step.rewards


if __name__ == "__main__":
  main()

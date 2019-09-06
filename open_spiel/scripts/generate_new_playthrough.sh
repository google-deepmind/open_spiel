#!/bin/sh

# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generates a playthrough for a new game with optional parameters.
# This script exists mainly as a reminder for the command to run.

GAME="$1"
shift

if [ "$GAME" = "" ]
then
  echo "Usage: generate_new_playthrough GAME"
  exit
fi

python open_spiel/python/examples/playthrough.py \
--game $GAME \
--output_file open_spiel/integration_tests/playthroughs/$GAME.txt \
--alsologtostdout

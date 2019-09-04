#!/bin/sh

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

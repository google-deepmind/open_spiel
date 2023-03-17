#!/bin/bash
game=$1
echo "Start $game experiments".
for seed in 49 48 121 207 227 84 190 77 123 73
do
  ./run_experiment.sh ${game}_pg_${seed} $seed --game $game --correction_type none
  #./run_experiment.sh ${game}_dice_1_lookahead_${seed} $seed --game $game --n_lookaheads 1 --correction_type dice
  #./run_experiment.sh ${game}_dice_2_lookahead_${seed} $seed --game $game --n_lookaheads 2 --correction_type dice
  #./run_experiment.sh ${game}_dice_3_lookahead_${seed} $seed --game $game --n_lookaheads 3 --correction_type dice
  #./run_experiment.sh ${game}_dice_1_lookahead_om_${seed} $seed --game $game --n_lookaheads 1 --correction_type dice --use_opponent_modelling
  #./run_experiment.sh ${game}_dice_2_lookahead_om_${seed} $seed --game $game --n_lookaheads 2 --correction_type dice --use_opponent_modelling
  #./run_experiment.sh ${game}_dice_3_lookahead_om_${seed} $seed --game $game --n_lookaheads 3 --correction_type dice --use_opponent_modelling
done

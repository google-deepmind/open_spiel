#!/bin/bash
name=${1}
seed=${2}
docker run --rm -itd --gpus all -u $(id -u):$(id -g) \
  --name ${name} \
  -v $(pwd):/open_spiel \
  open_spiel/lola:latest --seed $seed ${@:3}
#!/usr/bin/env bash

for file in ./slurm_scripts/*
do
  sbatch "$file"
done
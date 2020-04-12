#!/usr/bin/env bash

for file in ./scripts/*
do
  sbatch "$file"
  sleep 2
done

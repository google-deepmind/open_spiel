#!/usr/bin/env bash

for file in ./scripts/*
do
  sbatch "$file"
done
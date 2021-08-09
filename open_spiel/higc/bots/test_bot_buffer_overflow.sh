#!/usr/bin/bash

# Test for some fun overflows and killing child processes.
python -c 'for _ in range(10000000000): print("x", end="")'

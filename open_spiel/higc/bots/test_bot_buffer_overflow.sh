#!/usr/bin/bash

echo "ready"
echo "start"

# Test for some fun overflows and killing child processes.
python -c 'print("x"*100000)'

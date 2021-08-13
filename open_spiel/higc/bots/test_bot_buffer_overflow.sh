#!/usr/bin/bash

echo "ready"
echo "start"

# Test for some fun overflows and killing child processes.
for i in {1..100000}; do echo -n "x"; done
echo ""

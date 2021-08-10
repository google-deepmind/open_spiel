#!/bin/bash

BASE=$(dirname "$0")
source "$BASE/../../../venv/bin/activate"
python "$BASE/test_bot_first_action.py"

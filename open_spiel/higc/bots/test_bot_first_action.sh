#!/bin/bash

>&2 env

BASE=$(dirname "$0")
python "$BASE/test_bot_first_action.py"

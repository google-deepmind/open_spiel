#!/bin/bash

set -e
set -x

virtualenv -p python ./venv
source ./venv/bin/activate

python --version
pip3 install -r requirements.txt

./open_spiel/scripts/build_and_run_tests.sh
deactivate

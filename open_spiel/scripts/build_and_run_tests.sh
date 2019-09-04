#!/bin/bash

# The following builds open_spiel and executes the tests using the `python`
# command. The version under 'python' is automatically detected.
#
# It assumes:
#   - we are at the root of the project
#   - `install.sh` has been run
#   - the Python dependencies has been installed, e.g. using
#     `pip install --upgrade -r ../requirements.txt`.
#
# Using a virtualenv is recommended but not mandatory.
#
set -e  # exit when any command fails
set -x

CXX=g++
NPROC=nproc
if [[ "$OSTYPE" == "darwin"* ]]; then  # Mac OSX
  NPROC="sysctl -n hw.physicalcpu"
  CXX=/usr/local/bin/g++-7
fi

MAKE_NUM_PROCS=$(${NPROC})
let TEST_NUM_PROCS=4*${MAKE_NUM_PROCS}

PYVERSION=$(python3 -c 'import sys; print(sys.version.split(" ")[0])')
PY_VERSION_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')

BUILD_DIR="build_python_$PY_VERSION_MAJOR"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "Building and testing in $PWD using 'python' (version $PYVERSION)."

cmake -DPython_TARGET_VERSION=${PYVERSION} -DCMAKE_CXX_COMPILER=${CXX} ../open_spiel
make -j$MAKE_NUM_PROCS

export PYTHONPATH=$PYTHONPATH:`pwd`/../open_spiel
export PYTHONPATH=$PYTHONPATH:`pwd`/python  # For the Python bindings of Pyspiel

if ctest -j$TEST_NUM_PROCS --output-on-failure ../open_spiel; then
  echo -e "\033[32mAll tests passed. Nicely done!\e[0m"
else
  echo -e "\033[31mAt least one test failed.\e[0m"
  echo "If this is the first time you have run these tests, try:"
  echo "pip install -r requirements.txt"
  echo "Note that outside a virtualenv, you will need to install the system "
  echo "wide matplotlib: sudo apt-get install python-matplotlib"
  exit 1
fi

cd ..

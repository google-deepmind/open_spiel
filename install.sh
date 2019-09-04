#!/usr/bin/env bash


# The following should be easy to setup as a submodule:
# https://git-scm.com/docs/git-submodule

set -e  # exit when any command fails
set -x

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  sudo apt-get update
  sudo apt-get install git virtualenv cmake python3 python3-dev python3-pip python3-setuptools python3-wheel
  if [[ "$TRAVIS" ]]; then
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${OS_PYTHON_VERSION} 10
  fi
elif [[ "$OSTYPE" == "darwin"* ]]; then  # Mac OSX
  [[ -x "$(type python3)" ]] || brew install python3 || echo "** Warning: failed 'brew install python3' -- continuing"
  # [[ -x "${type g++-7)" ]] || brew install gcc@7 || echo "** Warning: failed 'brew install gcc@7' -- continuing"
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python3 get-pip.py
  pip3 install virtualenv
else
  echo "The OS '$OSTYPE' is not supported (Only Linux and MacOS is). " \
       "Feel free to contribute the install for a new OS."
  exit 1
fi

[[ -d "./pybind11" ]] || git clone -b 'v2.2.4' --single-branch --depth 1 https://github.com/pybind/pybind11.git
# The official https://github.com/dds-bridge/dds.git seems to not accept PR,
# so we have forked it.
[[ -d open_spiel/games/bridge/double_dummy_solver ]] || \
  git clone -b 'develop' --single-branch --depth 1 https://github.com/jblespiau/dds.git  \
  open_spiel/games/bridge/double_dummy_solver
[[ -d open_spiel/abseil-cpp ]] || \
  git clone -b 'master' --single-branch --depth 1 https://github.com/abseil/abseil-cpp.git \
  open_spiel/abseil-cpp

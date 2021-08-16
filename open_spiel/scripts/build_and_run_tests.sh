#!/bin/bash

# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The following scripts:
# (optionally) create a virtualenv
# (optionally) install the pip package dependencies
# builds open_spiel
# executes the C++ tests
# executes the Python tests using the `python3` command.
# (optionally) runs the Julia tests
#
# We assume "install.sh` has been run once before.

# As we encourage the use of a virtualenv, it is set to be used by default.
# Use the --virtualenv=false flag to disable this feature.

# You will need to install at some points the requirements, within the
# virtualenv or as system wide packages. By default, it will be installed the
# first time the virtualenv is setup, but you can force an install using the
# --install=true flag.

# Load argslib for parsing of command-line arguments.
source $(dirname "$0")/argslib.sh

ArgsLibAddArg virtualenv bool true "Whether to use virtualenv. We enter a virtualenv (stored in venv/) only if this flag is true and we are not already in one."
# We define a string and not a boolean, because we can to know whether this flag
# has been explicitly set or not.
ArgsLibAddArg install string "default" 'Whether to install requirements.txt packages. Doing it is slow. By default, it will be true (a) the first-time a virtualenv is being setup (if system_wide_packages is false), (b) if the user overrides it with "true".'
ArgsLibAddArg system_wide_packages bool false 'Whether to use --system-site-packages on the virtualenv.'
ArgsLibAddArg build_with_pip bool false 'Whether to use "python3 -m pip install ." or the usual cmake&make and ctest.'
ArgsLibAddArg build_only bool false 'Builds only the library, without running tests.'
ArgsLibAddArg test_only string "all" 'Build and runs the tests matching this string (use "all" to run all tests)'
ArgsLibAddArg build_dir string "build" 'Location of the build directory.'
ArgsLibAddArg num_threads int -1 'Number of threads to use when paralellizing build / tests. (Defaults to 4*<number of CPUs>)'
ArgsLibParse $@

function die() {
  echo -e "\033[31m${1}\e[0m"
  exit -1
}

set -e  # exit when any command fails
# set -x  # Prints all executed command

MYDIR="$(dirname "$(realpath "$0")")"
source "${MYDIR}/global_variables.sh"

CXX=${CXX:-`which clang++`}
if [ ! -x $CXX ]
then
  echo -n "clang++ not found (the clang C++ compiler is needed to "
  echo "compile OpenSpiel). Exiting..."
  exit 1
fi

if [ "$ARG_num_threads" -eq -1 ]; then
  NPROC="nproc"
  if [[ "$OSTYPE" == "darwin"* ]]; then  # Mac OSX
    NPROC="sysctl -n hw.physicalcpu"
  fi

  MAKE_NUM_PROCS=$(${NPROC})
  let TEST_NUM_PROCS=4*${MAKE_NUM_PROCS}
else
  MAKE_NUM_PROCS=$ARG_num_threads
  TEST_NUM_PROCS=$ARG_num_threads
fi

# if we are in a virtual_env, we will not create a new one inside.
if [[ "$VIRTUAL_ENV" != "" ]]
then
  echo -e "\e[1m\e[93mVirtualenv already detected. We do not create a new one.\e[0m"
  ArgsLibSet virtualenv false
fi

echo -e "\e[33mRunning ${0} from $PWD\e[0m"
PYBIN=${PYBIN:-"python3"}
PYBIN=`which ${PYBIN}`
if [ ! -x $PYBIN ]
then
  echo -e "\e[1m\e[93m$PYBIN not found! Skip build and test.\e[0m"
  continue
fi

PYVERSION=$($PYBIN -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')

VENV_DIR="./venv"
if [[ $ARG_virtualenv == "true" ]]; then
  if ! [ -d "$VENV_DIR" ]; then
    extra_args=''
    if [[ $ARG_system_wide_packages == "true" ]]; then
      extra_args="--system-site-packages"
    else
      # If we are in a virtual-env, and are not using the system-wide packages
      # then we need to install the dependencies the first time the virtualenv
      # is created
      ArgsLibSet install true
    fi
    echo "Installing..."
    echo -e "\e[33mInstalling a virtualenv to $VENV_DIR. The setup is long the first time, please wait.\e[0m"
    virtualenv -p $PYBIN $VENV_DIR $extra_args
  else
    echo -e "\e[33mReusing virtualenv from $VENV_DIR.\e[0m"
  fi
  source $VENV_DIR/bin/activate
fi

# We only exit the virtualenv if we were asked to create one.
function cleanup {
  if [[ $ARG_virtualenv == "true" ]]; then
    echo "Exiting virtualenv"
    deactivate
  fi
}
trap cleanup EXIT

if [[ $ARG_install == "true" ]]; then
  echo -e "\e[33mInstalling the requirements (use --noinstall to skip).\e[0m"
  ${PYBIN} -m pip install --upgrade -r ./requirements.txt
else
  echo -e "\e[33mSkipping installation of requirements.txt.\e[0m"
fi

BUILD_DIR="$ARG_build_dir"
mkdir -p $BUILD_DIR

# Configure Julia compilation if required.
if [[ ${OPEN_SPIEL_BUILD_WITH_JULIA:-"OFF"} == "ON" ]]; then
  # Check that Julia is in the path.
  if [[ ! -x `which julia` ]] || [[ "$(julia -e 'println(VERSION >= v"1.6.0-rc1")')" == "false" ]]
  then
    echo -e "\e[33mWarning: julia not in your PATH or it's too old. Trying \$HOME/.local/bin\e[0m"
    PATH=${HOME}/.local/bin:${PATH}
    [[ -x `which julia` ]] && [[ "$(julia -e 'println(VERSION >= v"1.6.0-rc1")')" == "true" ]] || die "could not find julia command. Please add it to PATH and rerun."
  fi
  LIBCXXWRAP_JULIA_DIR=`julia --project=${MYDIR}/../julia -e 'using CxxWrap; print(dirname(dirname(CxxWrap.CxxWrapCore.libcxxwrap_julia_jll.libcxxwrap_julia_path)))'`
  JULIA_VERSION_INFO=`julia --version`
  echo "Found libcxxwrap_julia at $LIBCXXWRAP_JULIA_DIR with $JULIA_VERSION_INFO"
fi

function print_tests_passed {
  echo -e "\033[32mAll tests passed. Nicely done!\e[0m"
}

function print_tests_failed {
  echo -e "\033[31mAt least one test failed.\e[0m"
  echo "If this is the first time you have run these tests, try:"
  echo "python3 -m pip install -r requirements.txt"
  echo "Note that outside a virtualenv, you will need to install the system "
  echo "wide matplotlib: sudo apt-get install python-matplotlib"
  exit 1
}

function print_skipping_tests {
  echo -e "\033[32m*** Skipping to run tests.\e[0m"
}

function execute_export_graph {
  echo "Running tf_trajectories_example preliminary Python script"
  python ../open_spiel/contrib/python/export_graph.py
}

# Build / install everything and run tests (C++, Python, optionally Julia).
if [[ $ARG_build_with_pip == "true" ]]; then
  # TODO(author2): We probably want to use `python3 -m pip install .` directly
  # and skip the usage of nox.
  ${PYBIN} -m pip install nox

  if nox -s tests; then
    echo -e "\033[32mAll tests passed. Nicely done!\e[0m"
  else
    echo -e "\033[31mAt least one test failed.\e[0m"
    exit 1
  fi
else
  cd $BUILD_DIR
  echo "Building and testing in $PWD using 'python' (version $PYVERSION)."

  pwd=`pwd`
  export PYTHONPATH=$PYTHONPATH:$pwd/..
  export PYTHONPATH=$PYTHONPATH:$pwd/../open_spiel
  export PYTHONPATH=$PYTHONPATH:$pwd/python  # For pyspiel bindings

  # Build in testing, so that we can run tests fast.
  cmake -DPython3_EXECUTABLE=${PYBIN} \
        -DCMAKE_CXX_COMPILER=${CXX}                  \
        -DCMAKE_PREFIX_PATH=${LIBCXXWRAP_JULIA_DIR}  \
        -DBUILD_TYPE=Testing                         \
        ../open_spiel

  if [ "$ARG_test_only" != "all" ]
  then
    # Check for building and running a specific test.
    # TODO(author5): generlize this; currently only covers Python and C++ tests
    echo "Build and testing only $ARG_test_only"
    if [[ $ARG_test_only == python_* ]]; then
      echo "Building pyspiel"
      make -j$MAKE_NUM_PROCS pyspiel
    elif [[ $ARG_test_only == julia_test ]]; then
      echo "Building Julia API"
      make -j$MAKE_NUM_PROCS spieljl
    elif [[ $ARG_test_only == gospiel_test ]]; then
      echo "Building Go API"
      make -j$MAKE_NUM_PROCS gospiel
    else
      echo "Building everything"
      make -j$MAKE_NUM_PROCS
    fi

    if [[ $ARG_build_only == "true" ]]; then
      echo -e "\033[32m*** Skipping runing tests as build_only is ${ARG_build_only} \e[0m"
    else
      if [[ ${OPEN_SPIEL_BUILD_WITH_TENSORFLOW_CC:-"OFF"} == "ON" && $ARG_test_only =~ "tf_trajectories_example" ]]; then
        execute_export_graph
      fi

      if ctest -j$TEST_NUM_PROCS --output-on-failure -R "$ARG_test_only" ../open_spiel; then
        print_tests_passed
      else
        print_tests_failed
      fi
    fi
  else
    # Make everything
    echo "Building project"
    make -j$MAKE_NUM_PROCS

    if [[ $ARG_build_only == "true" ]]; then
      echo -e "\033[32m*** Skipping runing tests as build_only is ${ARG_build_only} \e[0m"
    else
      # Test everything
      echo "Running all tests"

      if [[ ${OPEN_SPIEL_BUILD_WITH_TENSORFLOW_CC:-"OFF"} == "ON" ]]; then
        execute_export_graph
      fi

      if ctest -j$TEST_NUM_PROCS --output-on-failure ../open_spiel; then
        print_tests_passed
      else
        print_tests_failed
      fi
    fi
  fi

  cd ..
fi

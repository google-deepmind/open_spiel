# Installation

The instructions here are for Linux and MacOS. For installation on Windows, see
[these separate installation instructions](windows.md).

Currently there are two installation methods:

1.  building from the source code and editing `PYTHONPATH`.
2.  using `pip install` to build and testing using
    [nox](https://nox.thea.codes/en/stable/). A pip package to install directly
    does not exist yet.
3.  installing via [Docker](https://www.docker.com).

## Summary

In a nutshell:

```bash
./install.sh  # Needed to run once and when major changes are released.
./open_spiel/scripts/build_and_run_tests.sh # Run this every-time you need to rebuild.
```

1.  Install system packages (e.g. cmake) and download some dependencies. Only
    needs to be run once or if you enable some new conditional dependencies (see
    specific section below).

    ```bash
    ./install.sh
    ```

2.  Install your Python dependencies, e.g. in Python 3 using
    [`virtualenv`](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/):

    ```bash
    virtualenv -p python3 venv
    source venv/bin/activate
    ```

    Use `deactivate` to quit the virtual environment.

    `pip` should be installed once and upgraded:

    ```
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    # Install pip deps as your user. Do not use the system's pip.
    python3 get-pip.py
    pip3 install --upgrade pip
    pip3 install --upgrade setuptools testresources
    ```

3.  This sections differs depending on the installation procedure:

    **Building and testing from source**

    ```bash
    pip3 install -r requirements.txt
    ./open_spiel/scripts/build_and_run_tests.sh
    ```

    **Building and testing using PIP**

    ```bash
    python3 -m pip install .
    pip install nox
    nox -s tests
    ```

    Optionally, use `pip install -e` to install in
    [editable mode](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs),
    which will allow you to skip this `pip install` step if you edit any Python
    source files. If you edit any C++ files, you will have to rerun the install
    command.

4.  Only when building from source:

    ```bash
    # For the python modules in open_spiel.
    export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel>
    # For the Python bindings of Pyspiel
    export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel>/build/python
    ```

    to `./venv/bin/activate` or your `~/.bashrc` to be able to import OpenSpiel
    from anywhere.

To make sure OpenSpiel works on the default configurations, we do use the
`python3` command and not `python` (which still defaults to Python 2 on modern
Linux versions).

## Installing via Docker

Option 1 (Basic, 3.13GB):

```bash
docker build --target base -t openspiel . --rm
```

Option 2 (Slim, 2.26GB):

```bash
docker build --target python-slim -t openspiel . --rm
```

If you are only interested in developing in Python, use the second image. You
can navigate through the runtime of the container (after the build step) with:

```bash
docker run -it --entrypoint /bin/bash openspiel
```

Finally you can run examples using:

```bash
docker run openspiel python3 python/examples/matrix_game_example.py
docker run openspiel python3 python/examples/example.py
```

## Running the first examples

In the `build` directory, running `examples/example` will prints out a list of
registered games and the usage. Now, let’s play game of Tic-Tac-Toe with uniform
random players:

```bash
examples/example --game=tic_tac_toe
```

Once the proper Python paths are set, from the main directory (one above
`build`), try these out:

```bash
# Similar to the C++ example:
python3 open_spiel/python/examples/example.py --game=breakthrough

# Play a game against a random or MCTS bot:
python3 open_spiel/python/examples/mcts.py --game=tic_tac_toe --player1=human --player2=random
python3 open_spiel/python/examples/mcts.py --game=tic_tac_toe --player1=human --player2=mcts
```

## Detailed steps

### Configuration conditional dependencies

See [open_spiel/scripts/global_variables.sh](https://github.com/deepmind/open_spiel/blob/master/open_spiel/scripts/global_variables.sh) to configure the
conditional dependencies. See also the [Developer Guide](developer_guide.md).

### Installing system-wide dependencies

See [open_spiel/scripts/install.sh](https://github.com/deepmind/open_spiel/blob/master/open_spiel/scripts/install.sh) for the required packages and cloned
repositories.

### Installing Python dependencies

Using a `virtualenv` to install python dependencies is highly recommended. For
more information see:
[https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

Install dependencies (Python 3):

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Alternatively, although not recommended, you can install the Python dependencies
system-wide with:

```bash
pip3 install --upgrade -r requirements.txt
```

### Building and running tests

Make sure that the virtual environment is still activated.

By default, Clang C++ compiler is used (and potentially installed by
[open_spiel/scripts/install.sh](https://github.com/deepmind/open_spiel/blob/master/open_spiel/scripts/install.sh)).

Build and run tests (Python 3):

```bash
mkdir build
cd build
CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=${CXX} ../open_spiel
make -j$(nproc)
ctest -j$(nproc)
```

The CMake variable `Python3_EXECUTABLE` is used to specify the Python
interpreter. If the variable is not set, CMake's FindPython3 module will prefer
the latest version installed. Note, Python >= 3.6.0 is required.

One can run an example of a game running (in the `build/` folder):

```bash
./examples/example --game=tic_tac_toe
```

### Setting Your PYTHONPATH environment variable

To be able to import the Python code (both the C++ binding `pyspiel` and the
rest) from any location, you will need to add to your PYTHONPATH the root
directory and the `open_spiel` directory.

When using a virtualenv, the following should be added to
`<virtualenv>/bin/activate`. For a system-wide install, ddd it in your `.bashrc`
or `.profile`.

```bash
# For the python modules in open_spiel.
export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel>
# For the Python bindings of Pyspiel
export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel>/build/python
```

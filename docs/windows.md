# OpenSpiel Installation on Windows

OpenSpiel has limited support on Windows and is not being regularly tested,
which means support could break at any time. This may change in the future, but
for now please be aware that Windows support is experimental. Please report any
bugs or problems you encounter.

OpenSpiel has limited support on Windows and is not being regularly tested,
which means support could break at any time. This may change in the future
(contributions are welcome), with Github Actions supporting
[windows workers](https://docs.github.com/en/actions/using-github-hosted-runners/customizing-github-hosted-runners#installing-software-on-windows-runners!),
but for now please be aware that Windows support is experimental. Please report
any bugs or problems you encounter.

## Option 1: Windows Installation using Visual Studio Community Edition

This option will describe how to install and use OpenSpiel on Windows 10 via
[Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/).
This process has been written for Windows 10 and tested on Windos 10 Home
Version 20H2, build 19042.1415 (installed on Nov 26th, 2021).

When installing Visual Studio, enable the C++ and Python development, and also
the C++ CMake tools for Windows. C++/CLI support and C++ Clang tools may also be
useful (but not necessary).

You will need to have the following dependencies installed:

*   [CMake](https://cmake.org/download/)
*   [git](https://gitforwindows.org/)
*   [Python](https://www.python.org/downloads/windows/). Note: get the latest
    3.9 release as OpenSpiel has not been tested on 3.10 yet. Also, tick the box
    during instalation to ensure Python executable is in your path.
*   Recommended: Windows Terminal / Powershell.

The rest of the instructions will assume that OpenSpiel is cloned in
`C:\Users\MyUser\open_spiel`.

Open a Windows Terminal (Windows Powershell), clone OpenSpiel and its
dependencies (commands adapted from open_spiel/scripts/install.sh)

```
cd C:\Users\MyUser
git clone https://github.com/deepmind/open_spiel.git
cd open_spiel
git clone -b smart_holder --single-branch --depth 1 https://github.com/pybind/pybind11.git pybind11
git clone -b 20211102.0 --single-branch --depth 1 https://github.com/abseil/abseil-cpp.git open_spiel\abseil-cpp
git clone -b develop --single-branch --depth 1 https://github.com/jblespiau/dds.git open_spiel\games\bridge\double_dummy_solver
```

Open Visual Studio and continue without code. Then, click on File | Open ->
CMake, and choose `C:\Users\MyUser\open_spiel\open_spiel\CMakeLists.txt`. CMake
will then run; once you see `CMake generation finished`, choose Build -> Build
All. The files will be available in
`C:\Users\MyUser\open_spiel\open_spiel\out\build\x64-Debug`, when the build
completes with "Build All succeeded."

To be able to import the Python code (both the C++ binding `pyspiel` and the
rest) from any location, you will need to add to your PYTHONPATH the root
directory and the `open_spiel` directory. Open
[Windows environment variables and add to the PYTHONPATH](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-so-it-finds-my-modules-packages).
Add the directories `C:\Users\MyUser\open_spiel\open_spiel\out\build\x64-Debug`
and `C:\Users\MyUser\open_spiel\open_spiel\out\build\x64-Debug\python` to
PYTHONPATH. If your PYTHONPATH does not exist, then create a new environment
variable for it. To check that python is working, you can run the example in
`open_spiel\python\examples`.

OpenSpiel has various Python dependencies which may require installing. At a
minimum, you will need the ones in
[requirements.txt](https://github.com/deepmind/open_spiel/blob/master/requirements.txt).

```
pip install absl-py
pip install attrs
pip install numpy
```

For a complete list, depending on what you will use, see
[python_extra_deps.sh](https://github.com/deepmind/open_spiel/blob/master/open_spiel/scripts/python_extra_deps.sh).

## Option 2: Windows Installation using Windows Subsystem for Linux (WSL)

This section describes the installation steps to get OpenSpiel running in a
Windows 10 environment using Windows Subsystem for Linux (WSL). Note that WSL
does not include GPU support, so will run on CPU only.

## Process

This process has been written for Windows 10, and tested on Windows 10 build
1903 (March 2019).

1.  Install the Windows Subsystem for Linux:

    Run the following command in Windows Powershell:

    ```powershell
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
    ```

2.  Install Ubuntu Linux from the Windows Store. Currently this is version
    18.04::

    Open up the Windows Store. Search for Ubuntu. Open up Ubuntu and press "Get"
    to install this.

3.  First time run of Ubuntu:

    Click on the Start Button and choose the Ubuntu icon. Wait until the distro
    installs. Provide a username and password for the default user account. Note
    that this account is a member of the Linux administrators (sudo) group so
    choose a secure username and password combination.

4.  Update / Upgrade packages (optional step)

    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    ```

5.  Run through the first part of the OpenSpiel installation

    ```bash
    git clone https://github.com/deepmind/open_spiel.git
    cd open_spiel
    ./install.sh # you will be prompted for the password created at stage 3. Press Y to continue and install. During installation press Yes to restart services during package upgrades
    pip install -U pip # Upgrade pip (required for TF >= 1.15)
    pip3 install --upgrade -r requirements.txt # Install Python dependencies
    ```

6.  Now need to upgrade make version as the version of make which comes with
    Ubuntu 18.04 is not high enough to build OpenSpiel. (Note, this step won't
    be necessary if the version of Ubuntu in the Windows store gets upgraded to
    19.04)

    ```bash
    cd ..
    wget http://www.cmake.org/files/v3.12/cmake-3.12.4.tar.gz
    tar -xvzf cmake-3.12.4.tar.gz
    cd cmake-3.12.4/
    ./configure
    make
    sudo make install
    sudo update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force
    cd ../open_spiel
    ```

7.  Finally, continue with the installation and run tests.

    ```bash
    mkdir build
    cd build
    CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=clang++ ../open_spiel
    make -j12 # The 12 here is the number of parallel processes used to build
    ctest -j12 # Run the tests to verify that the installation succeeded
    ```

    The CMake variable `Python3_EXECUTABLE` is used to specify the Python
    interpreter. If the variable is not set, CMake's FindPython3 module will
    prefer the latest version installed. Note, Python >= 3.6.0 is required.

    One can run an example of a game running (in the `build/` folder):

    ```bash
    ./examples/example --game=tic_tac_toe
    ```

8.  Setting Your PYTHONPATH environment variable

    To be able to import the Python code (both the C++ binding `pyspiel` and the
    rest) from any location, you will need to add to your PYTHONPATH the root
    directory and the `open_spiel` directory.

    When using a virtualenv, the following should be added to
    `<virtualenv>/bin/activate`. For a system-wide install, ddd it in your
    `.bashrc` or `.profile`.

    ```bash
    # For the python modules in open_spiel.
    export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel>
    # For the Python bindings of Pyspiel
    export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel>/build/python
    ```

9.  Running the first example

    In the `build` directory, running `examples/example` will prints out a list
    of registered games and the usage. Now, let’s play game of Tic-Tac-Toe with
    uniform random players:

    ```bash
    examples/example --game=tic_tac_toe
    ```

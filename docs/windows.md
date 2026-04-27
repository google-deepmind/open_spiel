
<!-- disableFinding(LINE_OVER_80) -->
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

This option will describe how to install and use OpenSpiel on Windows 11 via
[Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/).
This process has been written for Windows 11 and tested on Windows 11 Version
25H2, build 26200.8037 (April 2026).

When installing Visual Studio, enable the C++ and Python development, and also
the C++ CMake tools for Windows. C++/CLI support and C++ Clang tools may also be
useful (but not necessary).

You will need to have the following dependencies installed:

*   [CMake](https://cmake.org/download/)
*   [git](https://gitforwindows.org/)
*   [Python](https://www.python.org/downloads/windows/) >= 3.11. Tick the box
    during installation to ensure Python executable is in your path.
*   Recommended: Windows Terminal / Powershell.

The rest of the instructions will assume that OpenSpiel is cloned in
`C:\Users\MyUser\open_spiel`.

Open a Windows Terminal (Windows Powershell), clone OpenSpiel and its
dependencies (commands adapted from open_spiel/scripts/install.sh)

```
cd C:\Users\MyUser
git clone https://github.com/deepmind/open_spiel.git
cd open_spiel
git clone --single-branch --depth 1 https://github.com/pybind/pybind11.git pybind11
git clone --single-branch --depth 1 https://github.com/pybind/pybind11_json.git open_spiel\pybind11_json
git clone --single-branch --depth 1 https://github.com/abseil/abseil-cpp.git open_spiel\abseil-cpp
git clone --single-branch --depth 1 https://github.com/nlohmann/json.git open_spiel\json
git clone https://github.com/pybind/pybind11_abseil.git open_spiel\pybind11_abseil
git clone -b develop --single-branch --depth 1 https://github.com/jblespiau/dds.git open_spiel\games\bridge\double_dummy_solver
```

Open Visual Studio and continue without code. Then, click on File | Open ->
CMake, and choose `C:\Users\MyUser\open_spiel\open_spiel\CMakeLists.txt`. CMake
will then run; once you see `CMake generation finished`, choose Build -> Build
All. The files will be available in
`C:\Users\MyUser\open_spiel\open_spiel\out\build\x64-Debug`, when the build
completes with "Build All succeeded." Extra compilation options may be necessary
if errors occur. \
MSVC options to deal with required C++ standard, file encoding (for chess
characters) and large object files include `/std:c++20`, `/utf-8`, `/bigobj`. To
use them together with default MSVC arguments, you can use the following CMake
command line arguments: `-DCMAKE_CXX_FLAGS="/std:c++20 /utf-8 /bigobj /DWIN32
/D_WINDOWS /GR /EHsc"`

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

## Option 2: Windows Installation using Windows Subsystem for Linux (WSL2)

This section describes the installation steps to get OpenSpiel running in a
Windows 10/11 environment using WSL2.

### Process

1.  Install WSL2 with Ubuntu:

    Run the following command in an elevated Windows Powershell:

    ```powershell
    wsl --install
    ```

    This installs WSL2 with Ubuntu by default. Restart your machine if prompted.

2.  First time run of Ubuntu:

    Open Ubuntu from the Start Menu. Provide a username and password for the
    default user account.

3.  Once the Ubuntu environment is running, follow the standard Linux
    installation instructions in [install.md](install.md) to clone, build, and
    set up OpenSpiel.

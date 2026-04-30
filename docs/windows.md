
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(HEADING_REPEAT_H1) -->
<!-- disableFinding(LIST_NO_LINE) -->
# OpenSpiel Installation on Windows

OpenSpiel now has official support for Windows with pre-built binary wheels
available on PyPI. Windows wheels are built and tested automatically via GitHub
Actions CI for Python 3.11, 3.12, and 3.13.

> **Note:** Windows support is currently experimental. If you encounter any
> issues, please [open an issue](https://github.com/deepmind/open_spiel/issues)
> on GitHub so we can improve the Windows experience.

## Option 1: Quick Installation using pip (Recommended)

The easiest way to install OpenSpiel on Windows is using pip:

```bash
pip install open-spiel
```

### Optional dependencies

For additional features like visualization and machine learning: `bash pip
install open-spiel[full]`

### Verification

Test your installation: ```python import pyspiel

# Create a simple game

game = pyspiel.load_game("tic_tac_toe") state = game.new_initial_state()
print("OpenSpiel is working!") ```

### Building from Source

If you need to build from source or contribute to the project:

#### Prerequisites

-   **Python 3.11+** (get from [python.org](https://python.org))
-   **Git** (get from [git-scm.com](https://git-scm.com))
-   **CMake 3.15+** (get from [cmake.org](https://cmake.org))
-   **Visual Studio 2019 or later** with C++ development tools

#### Build Steps

1.  Clone the repository: `bash git clone
    https://github.com/deepmind/open_spiel.git cd open_spiel`

2.  Build the wheel: `bash python -m pip wheel . --no-deps -w dist`

3.  Install the built wheel: `bash pip install dist/open_spiel-*.whl`

### Troubleshooting

**"CMake not found"** - Install CMake from [cmake.org](https://cmake.org) and
add it to your PATH

**"Git not found"** \
- Install Git from [git-scm.com](https://git-scm.com) and add it to your PATH

**"MSVC compiler not found"** - Install Visual Studio Community with C++
development tools - Or install "Microsoft C++ Build Tools"

**"Import pyspiel failed"** - Make sure you installed the package: `pip install
open-spiel` - Try reinstalling: `pip uninstall open-spiel && pip install
open-spiel`

### Development Installation

```bash
git clone https://github.com/deepmind/open_spiel.git
cd open_spiel
pip install -e .
```

### Using with Conda

```bash
conda create -n openspiel python=3.11
conda activate openspiel
pip install open-spiel
```

## Option 2: Windows Installation using Visual Studio Community Edition

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

## Option 3: Windows Installation using Windows Subsystem for Linux (WSL2)

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

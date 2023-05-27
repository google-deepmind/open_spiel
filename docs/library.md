# Using OpenSpiel as a C++ Library

OpenSpiel has been designed as a framework: a suite of games, algorithms, and
tools for research in reinforcement learning and search in games. However, there
are situations where one may only want or need a single game/algorithm or small
subset from this collection, or a research experiment does not require modifying
or otherwise interacting very closely with OpenSpiel other than strictly
calling/using it.

In cases like this, it might be nice to use OpenSpiel as a library rather than a
framework. This has the benefit of not forcing the use of certain tools like
CMake or having to continually recompile OpenSpiel when doing your research.

Luckily, this is easy to achieve with OpenSpiel: you simply need to build it as
a shared library once, and then load it dynamically at runtime. This page walks
through how to do this assuming a bash shell on Linux, but is very similar on
MacOS or for other shells.

## Install Dependencies

The dependencies of OpenSpiel need to be installed before it can be used as a
library. On MacOS and Debian/Ubuntu Linux, this is often simply just running
`./install.sh`. Please see the [installation from source instructions](https://github.com/deepmind/open_spiel/blob/master/docs/install.md#installation-from-source) for more details.

## Compiling OpenSpiel as a Shared Library

To build OpenSpiel as a shared library, simply run:

```
mkdir build
cd build
BUILD_SHARED_LIB=ON CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=${CXX} ../open_spiel
make -j$(nproc) open_spiel
```

This produces a dynamically-linked library `libopen_spiel.so` (or
`lib_openspiel.dylib` on MacOS) in `build/` that can be linked against and
loaded dynamically at run-time.

Suppose OpenSpiel was installed in `$HOME/open_spiel`. The following line adds
the necessary environment variable to let the shell know where to find
`libopen_spiel.so` at run-time:

```
export LD_LIBRARY_PATH="${HOME}/open_spiel/build"
```

You might want to add this line to your `$HOME/.bash_profile` to avoid having to
do it every time you load the library. Of course, if you are already using
`LD_LIBRARY_PATH` for something else, then you need to add
`${HOME}/open_spiel/build` to it (space-separated paths).

## Compiling and Running the Example

```
cd ../open_spiel/examples
clang++ -I${HOME}/open_spiel -I${HOME}/open_spiel/open_spiel/abseil-cpp \
        -std=c++17 -o shared_library_example shared_library_example.cc \
        -L${HOME}/open_spiel/build  -lopen_spiel
```

The first two flags are the include directory paths and the third is the link
directory path. The `-lopen_spiel` instructs the linker to link against the
OpenSpiel shared library.

That's it! Now you can run the example using:

```
./shared_library_example breakthrough
```

You should also be able to register new games externally without the
implementation being within OpenSpiel nor built into the shared library, though
we are always interested in growing the library and recommend you contact us
about contributing any new games to the suite.

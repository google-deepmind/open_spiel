# OpenSpiel Rust API

This is a basic [Rust](https://www.rust-lang.org/) API for OpenSpiel. Please
note that it is currently experimental and may not work as expected. If you use
it, please report any issues. Fixes and improvements are more than welcome!

See the `CMakeLists.txt` to see how it is setup: a dynamic shared library is
created similarly to python extension (`librust_spiel.so`). A simple rust crate
is created in this directory using `cargo build` and a simple example is run
using cargo as well. Note that currently `LD_LIBRARY_PATH` must include the
location of the dynamic library so that it gets properly loaded at run time.

Note: this API currently only supports turn-based games. To support
simultaneous-move games, several API functions would need to be added, such as
legal actions for specific players, observation and information state tensors
for specific players, and apply action for joint actions.

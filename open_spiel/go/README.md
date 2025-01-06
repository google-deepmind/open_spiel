# OpenSpiel Go API

*Note:* This API has issues and is no longer maintained. See
[issue 1301](https://github.com/google-deepmind/open_spiel/issues/1301) for
details.

This is a basic [Go](https://golang.org/) API for OpenSpiel. Please note that it
is currently experimental and may not work as expected. Please see the
[announcement thread](https://github.com/deepmind/open_spiel/issues/541) and
report any issues. Fixes and improvements are more than welcome!

See the `CMakeLists.txt` to see how it is setup: a dynamic shared library is
created similarly to python extension (`libgospiel.so`). A simple go module is
created in this directory using `go mod init` so that go tests can be run. Note
that currently `LD_LIBRARY_PATH` must include the location of the dynamic
library so that it gets properly loaded at run time.

# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Build files

The Bazel `BUILD.bazel` files are the source of truth for the C++ build. The
per-package `CMakeLists.txt` files (and `open_spiel/targets.cmake`) are
generated from them; do not edit those by hand. After changing any
`BUILD.bazel` file, regenerate them and include the result in your pull
request:

```shell
open_spiel/scripts/generate_cmake.sh
```

This needs only Python 3.9 or later (no Bazel). Continuous integration runs
`open_spiel/scripts/generate_cmake.sh --check` and fails if the generated files
are out of date.

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google.com/conduct/).

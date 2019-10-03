# Ludii Wrapper

This is an experimental work-in-progress C++ wrapper of the
[Ludii General Game System](https://ludii.games/). The Ludii library is written
in Java so this wrapper uses
[JNI](https://docs.oracle.com/javase/8/docs/technotes/guides/jni/) to interact
with the Ludii jar through C++.

For discussion on the development of this wrapper, please see
[issue #39](https://github.com/deepmind/open_spiel/issues/39).

## How to build

Tested on Ubuntu 16.04 with Java 8 openJDK and Ludii player (0.3.0).

1.  Install openjdk if you haven't already.

2.  Download Ludii player (0.3.0) jar from
    [downloads page](https://ludii.games/downloads.php).

3.  Check `games/ludii/CMakeLists`. Assuming Java 8 openJDK is installed the
    JDK_HOME is set to `/usr/lib/jvm/java-8-openjdk-amd64`. This might have to
    be changed if a different version is installed.

4.  Uncomment the `add_subdirectory (ludii)` line in `games/CMakeLists.txt`

5.  Build OpenSpiel as usual, then run `build/games/ludii/ludii_demo <path to
    Ludii jar>`

If `libjvm.so` is not found, run:

`export LD_LIBRARY_PATH=/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/`

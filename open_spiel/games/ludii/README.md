# Ludii Wrapper

This is an experimental work-in-progress C++ wrapper of the [Ludii General Game System](https://ludii.games/).  The Ludii library is written in Java so this wrapper uses [JNI](https://docs.oracle.com/javase/8/docs/technotes/guides/jni/) to interact with the Ludii jar through C++.

## How to build

Tested on Ubuntu 16.04 with Java 8 openJDK and Ludii player (0.3.0).

1.  Install openjdk if you haven't already.

2.	Download Ludii player (0.3.0) jar from [downloads page](https://ludii.games/downloads.php).

3.	Enter path to jar on line 31 of `ludii_demo.cpp`.

3.  Assuming Java 8 openJDK is installed at `/usr/lib/jvm/java-8-openjdk-amd64` build with

	```bash
 	g++ -g -I/usr/lib/jvm/java-8-openjdk-amd64/include -I/usr/lib/jvm/java-8-openjdk-amd64/include/linux -L/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/ ludii_demo.cpp jni_utils.cpp game_loader.cpp game.cpp mode.cpp trial.cpp context.cpp state.cpp container_state.cpp region.cpp chunk_set.cpp moves.cpp move.cpp -ljvm
 	```

	If `libjvm.so` is not found 

	```export LD_LIBRARY_PATH=/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/```
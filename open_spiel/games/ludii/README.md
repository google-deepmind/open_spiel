# Ludi Wrapper

## How to build

Tested on Ubuntu 16.04 with Java 8 openJDK 

Assuming Java 8 openJDK is installed at `/usr/lib/jvm/java-8-openjdk-amd64` build with

``` g++ -g -I/usr/lib/jvm/java-8-openjdk-amd64/include -I/usr/lib/jvm/java-8-openjdk-amd64/include/linux -L/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/ ludii.cpp -ljvm```

If `libjvm.so` is not found you might have to 

```export LD_LIBRARY_PATH=/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/```
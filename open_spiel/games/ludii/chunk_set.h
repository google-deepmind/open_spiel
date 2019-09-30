#ifndef CHUNKSET_H_
#define CHUNKSET_H_

#include "jni.h"
#include <string>


class ChunkSet
{

public:

    ChunkSet(JNIEnv *env, jobject chunkset);

    std::string Print();

    std::string ToChunkString();

private:

    JNIEnv *env;
    jobject chunkset;
};

#endif
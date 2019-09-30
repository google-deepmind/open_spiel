#ifndef CHUNKSET_H_
#define CHUNKSET_H_

#include "jni.h"
#include <string>


class ChunkSet
{

public:

    ChunkSet(JNIEnv *env, jobject chunkset);

    std::string Print() const;

    std::string ToChunkString() const;

private:

    JNIEnv *env;
    jobject chunkset;
};

#endif
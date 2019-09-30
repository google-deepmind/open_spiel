#ifndef CONTAINERSTATE_H_
#define CONTAINERSTATE_H_

#include "jni.h"
#include "region.h"
#include "chunk_set.h"

class ContainerState
{

public:

    ContainerState(JNIEnv *env, jobject container_state);

    Region Empty() const;

    ChunkSet CloneWho() const;

    ChunkSet CloneWhat() const;

private:

    JNIEnv *env;
    jobject container_state;

};

#endif
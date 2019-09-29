#ifndef REGION_H_
#define REGION_H_

#include "jni.h"
#include "chunk_set.h"

class Region{

public:

	Region(JNIEnv *env, jobject region);

	ChunkSet BitSet();

private:

	JNIEnv *env;
	jobject region;
};

#endif
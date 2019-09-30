#include "region.h"


Region::Region(JNIEnv *env, jobject region): env(env), region(region) {}


ChunkSet Region::BitSet() const
{

    jclass regionClass = env->FindClass("util/Region");
    jmethodID bitSet_id = env->GetMethodID(regionClass, "bitSet", "()Lutil/ChunkSet;");
    jobject chunkset_obj = env->CallObjectMethod(region, bitSet_id);

    return ChunkSet(env, chunkset_obj);
}

#include "chunk_set.h"


ChunkSet::ChunkSet(JNIEnv *env, jobject chunkset):env(env),chunkset(chunkset){}

std::string ChunkSet::Print(){

	jclass chunkSetClass = env->FindClass("util/ChunkSet");
    jmethodID tostring_id = env->GetMethodID(chunkSetClass,"toString","()Ljava/lang/String;");
    jstring string_obj = (jstring) env->CallObjectMethod(chunkset,tostring_id);
   
    const char *rawString = env->GetStringUTFChars(string_obj, 0);
    std::string cppString(rawString);
    env->ReleaseStringUTFChars(string_obj, rawString);

    return cppString;
}

std::string ChunkSet::ToChunkString(){

	jclass chunkSetClass = env->FindClass("util/ChunkSet");
    jmethodID toChunkString_id = env->GetMethodID(chunkSetClass,"toChunkString","()Ljava/lang/String;");
    jstring string_obj = (jstring) env->CallObjectMethod(chunkset,toChunkString_id);
   
    const char *rawString = env->GetStringUTFChars(string_obj, 0);
    std::string cppString(rawString);
    env->ReleaseStringUTFChars(string_obj, rawString);

    return cppString;
}

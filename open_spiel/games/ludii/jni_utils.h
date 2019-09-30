#ifndef JNIUTILS_H_
#define JNIUTILS_H_

#include "jni.h"
#include <string>
#include <cstring>


class JNIUtils
{

public:

    JNIUtils(const std::string jar_location);
    ~JNIUtils();

    JNIEnv *GetEnv() const;

    void InitJVM(std::string jar_location);
    void CloseJVM();

private:

    JavaVM *jvm;
    JNIEnv *env;
    jint res;

};

#endif
#ifndef JNIUTILS_H_
#define JNIUTILS_H_

#include <iostream>
#include "jni.h"
#include <string>
#include <cstring>


class JNIUtils{

public:

	JNIUtils(const std::string jar_location);
	~JNIUtils();

	JNIEnv * getEnv();

	void initJVM(std::string jar_location);
	void closeJVM();

private:

	JavaVM *jvm;
	JNIEnv *env;
	jint res;

};

#endif
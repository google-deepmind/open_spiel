#ifndef MOVE_H_
#define MOVE_H_

#include "jni.h"
#include <iostream>
#include <string>
#include <vector>

class Move{

public:

	Move(JNIEnv *env, jobject move);

	jobject GetObj();

private:

	JNIEnv *env;
	jobject move;
};

#endif
#ifndef TRIAL_H_
#define TRIAL_H_

#include "jni.h"
#include "game.h"

class Trial{

public:

	Trial(JNIEnv *env, Game game);

	jobject getTrialObj();

private:

	JNIEnv *env;
	jobject trial;
};

#endif
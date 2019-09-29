#ifndef TRIAL_H_
#define TRIAL_H_

#include "jni.h"
#include "game.h"
#include "state.h"


class Trial{

public:

	Trial(JNIEnv *env, Game game);

	jobject GetObj();

	State GetState();

	bool Over();


private:

	JNIEnv *env;
	jobject trial;
};

#endif
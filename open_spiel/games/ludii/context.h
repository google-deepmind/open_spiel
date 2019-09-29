#ifndef CONTEXT_H_
#define CONTEXT_H_

#include "trial.h"

class Game;

class Context{


public:

	Context(JNIEnv *env,Game game, Trial trial);

	jobject GetObj();

private:

	JNIEnv *env;
	jobject context;

};


#endif
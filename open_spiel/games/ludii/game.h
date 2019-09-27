#ifndef GAME_H_
#define GAME_H_

#include "jni.h"
#include <string>
#include "mode.h"

class Context;

class Game{

public:

	Game(JNIEnv *env, jobject game_object, std::string game_path);

	std::string getPath();

	jobject getGameObj();

	std::string getName();

	int stateFlags();

	Mode mode();

	void start(Context context);

private:	

	JNIEnv *env;
	jobject game_object;
	std::string game_path;

};

#endif
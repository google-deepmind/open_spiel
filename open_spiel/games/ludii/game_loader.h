#ifndef LUDII_H_
#define LUDII_H_

#include "jni.h"
#include <string>
#include <vector>
#include "game.h"

class GameLoader{
	
public:

	GameLoader(JNIEnv *env_const);
	std::vector<std::string> listGames();
	Game loadGame(std::string game_name);

private:

	JNIEnv *env;

};

#endif
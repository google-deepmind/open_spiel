#ifndef LUDII_H_
#define LUDII_H_

#include "jni.h"
#include <string>
#include <vector>
#include "game.h"

class GameLoader
{

public:

    GameLoader(JNIEnv *env_const);
    std::vector<std::string> ListGames() const;
    Game LoadGame(std::string game_name) const;

private:

    JNIEnv *env;

};

#endif
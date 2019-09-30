#ifndef GAME_H_
#define GAME_H_

#include "jni.h"
#include <string>
#include "mode.h"
#include "moves.h"
#include "move.h"

class Context;

class Game
{

public:

    Game(JNIEnv *env, jobject game, std::string game_path);

    std::string GetPath() const;

    jobject GetObj() const;

    void Create(int viewSize) const;

    std::string GetName() const;

    int StateFlags() const;

    Mode GetMode() const;

    void Start(Context context) const;

    Moves GetMoves(Context context) const;

    Move Apply(Context context, Move move) const;

private:

    JNIEnv *env;
    jobject game;
    std::string game_path;

};

#endif
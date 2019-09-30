#ifndef MOVES_H_
#define MOVES_H_

#include "jni.h"
#include <string>
#include <vector>
#include "move.h"

class Moves
{

public:

    Moves(JNIEnv *env, jobject moves);

    std::vector<Move> GetMoves() const;

private:

    JNIEnv *env;
    jobject moves;
};

#endif
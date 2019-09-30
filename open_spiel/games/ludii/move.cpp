#include "move.h"



Move::Move(JNIEnv *env, jobject move): env(env), move(move) {}


jobject Move::GetObj() const
{
    return move;
}
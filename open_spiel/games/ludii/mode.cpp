#include "mode.h"


Mode::Mode(JNIEnv *env, jobject mode): env(env), mode(mode) {}


int Mode::NumPlayers() const
{

    jclass gameClass = env->FindClass("game/mode/Mode");
    jmethodID stateFlags_id = env->GetMethodID(gameClass, "numPlayers", "()I");
    return (int) env->CallIntMethod(mode, stateFlags_id);
}
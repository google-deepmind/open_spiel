#include "mode.h"


Mode::Mode(JNIEnv *env, jobject mode_object):env(env),mode_object(mode_object){}


int Mode::numPlayers(){

	jclass gameClass = env->FindClass("game/mode/Mode");
	jmethodID stateFlags_id = env->GetMethodID(gameClass,"numPlayers","()I");
	return (int) env->CallIntMethod(mode_object,stateFlags_id);
}
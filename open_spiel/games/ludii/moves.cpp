#include "moves.h"
#include <iostream>

Moves::Moves(JNIEnv *env, jobject moves):env(env),moves(moves){}

std::vector<Move> Moves::GetMoves(){

	std::vector<Move> moveVector;

	jclass moves_class = env->FindClass("game/rules/play/moves/Moves");
	jmethodID moves_id = env->GetMethodID(moves_class,"moves","()Lmain/FastArrayList;");
    	jobject moveFastArray_obj = env->CallObjectMethod(moves,moves_id);

    	jclass fastArray_class = env->FindClass("main/FastArrayList");
    	jmethodID fastArraySize_id = env->GetMethodID(fastArray_class,"size","()I");
    	jmethodID fastArrayGet_id = env->GetMethodID(fastArray_class,"get","(I)Ljava/lang/Object;");

    	jint fastArraySize = env->CallIntMethod(moveFastArray_obj, fastArraySize_id);

    	for (int i=0; i<fastArraySize; i++) {

        	jobject move_obj = env->CallObjectMethod(moveFastArray_obj,fastArrayGet_id,i);
        	moveVector.push_back(Move(env,move_obj));
    	}

    	return moveVector;
}

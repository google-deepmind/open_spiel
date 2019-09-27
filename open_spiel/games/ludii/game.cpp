#include "game.h"
#include <iostream>
#include "context.h"

Game::Game(JNIEnv *env, jobject game_object, std::string game_path)
	:env(env)
	,game_object(game_object)
	,game_path(game_path)
	{}

std::string Game::getPath(){
	return game_path;
}

jobject Game::getGameObj(){
	return game_object;
}

std::string Game::getName(){
	
	jclass gameClass = env->FindClass("game/Game");
	jmethodID name_id = env->GetMethodID(gameClass,"name","()Ljava/lang/String;");
	jstring stringArray =(jstring) env->CallObjectMethod(game_object,name_id);
	
	//convert jstring game name to char array
	const char *strReturn = env->GetStringUTFChars(stringArray, 0);
	std::string string_name(strReturn);
    env->ReleaseStringUTFChars(stringArray, strReturn);

    return string_name;
}

int Game::stateFlags(){
	jclass gameClass = env->FindClass("game/Game");
	jmethodID stateFlags_id = env->GetMethodID(gameClass,"stateFlags","()I");
	return (int) env->CallIntMethod(game_object,stateFlags_id);
}

Mode Game::mode(){
	jclass gameClass = env->FindClass("game/Game");
	jmethodID mode_id = env->GetMethodID(gameClass,"mode","()Lgame/mode/Mode;");	
	jobject mode = env->CallObjectMethod(game_object,mode_id);
	return Mode(env, mode);
}

void Game::start(Context context){

	jclass gameClass = env->FindClass("game/Game");
	jmethodID start_id = env->GetMethodID(gameClass,"start","(Lutil/Context;)V");	
	env->CallVoidMethod(game_object,start_id,context.getContextObj());
}
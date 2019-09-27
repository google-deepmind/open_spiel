#include "game_loader.h"
#include <string>
#include <iostream>
#include <cstring>


GameLoader::GameLoader(JNIEnv *env):env(env){}

std::vector<std::string> GameLoader::listGames(){

	std::vector<std::string> gamesVector;

	jclass gameLoader = env->FindClass("player/GameLoader");

    //find the listGames method
    jmethodID mid = env->GetStaticMethodID(gameLoader,"listGames","()[Ljava/lang/String;");

    //execute the listGames method
    jobjectArray stringArray = (jobjectArray) env->CallStaticObjectMethod(gameLoader,mid);

    //put list of games in vector
    int stringCount = env->GetArrayLength(stringArray);

	for (int i=0; i<stringCount; i++) {
		//get array element and convert it from jstrng
        jstring string = (jstring) (env->GetObjectArrayElement(stringArray, i));
        const char *rawString = env->GetStringUTFChars(string, 0);
       
       	std::string cppString(rawString);
        gamesVector.push_back(cppString);

        env->ReleaseStringUTFChars(string, rawString);
    }

	return gamesVector;
}

Game GameLoader::loadGame(std::string game_name){

	jclass gameLoader = env->FindClass("player/GameLoader");
    jmethodID mid = env->GetStaticMethodID(gameLoader,"loadGameFromName","(Ljava/lang/String;)Lgame/Game;");

    //convert game name to java string
	char game_name_char[1024];
	strcpy(game_name_char, game_name.c_str());
	jstring j_game_name = env->NewStringUTF(game_name_char);

	jobject game_obj = env->CallStaticObjectMethod(gameLoader,mid,j_game_name);

	return Game(env, game_obj, game_name);
}
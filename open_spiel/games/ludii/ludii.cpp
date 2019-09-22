#include "ludii.h"
#include <iostream>
#include <string>
#include <vector>


Ludii::Ludii(const std::string jar_location){
	initJVM(jar_location);
}

Ludii::~Ludii(){
	closeJVM();
}

std::vector<std::string> Ludii::listGames(){

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

void Ludii::loadGameAndDoStuff(std::string game_name){

	//create game object i.e. `Game game = GameLoader.loadGameFromName`
	jclass gameLoader = env->FindClass("player/GameLoader");
    jmethodID mid = env->GetStaticMethodID(gameLoader,"loadGameFromName","(Ljava/lang/String;)Lgame/Game;");

	//convert game name to java string
	char game_name_char[1024];
	strcpy(game_name_char, game_name.c_str());
	jstring j_game_name = env->NewStringUTF(game_name_char);

	jobject game_obj = env->CallStaticObjectMethod(gameLoader,mid,j_game_name);

	//call the `name()` method on the game object we just created
	jclass gameClass = env->FindClass("game/Game");
	jmethodID name_id = env->GetMethodID(gameClass,"name","()Ljava/lang/String;");
	jstring stringArray =(jstring) env->CallObjectMethod(game_obj,name_id);
	
	//convert jstring game name to char array and print
	const char *strReturn = env->GetStringUTFChars(stringArray, 0);
	std::cout<< "name() result: "<<strReturn<<std::endl;
    
    env->ReleaseStringUTFChars(stringArray, strReturn);
}


int main(){

	std::string jar_location = "/home/alex/Downloads/Ludii-0.3.0.jar";

	Ludii test_ludii = Ludii(jar_location);

	std::vector<std::string> games = test_ludii.listGames();

	test_ludii.loadGameAndDoStuff("board/space/blocking/Amazons.lud");
}
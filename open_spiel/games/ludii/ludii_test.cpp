#include "jni_utils.h"
#include "game_loader.h"
#include "game.h"
#include "trial.h"
#include "context.h"
#include <iostream>
#include <vector>
#include <string>

int main(){

	JNIUtils test_utils = JNIUtils("/home/alex/Downloads/Ludii-0.3.0.jar");
	JNIEnv *env = test_utils.getEnv();

	GameLoader gameLoader = GameLoader(env);

	std::vector<std::string> game_names = gameLoader.listGames();

	for (std::vector<std::string>::const_iterator i = game_names.begin(); i != game_names.end(); ++i)
    	std::cout << *i << ' '<<std::endl;

    Game test_game = gameLoader.loadGame("board/space/blocking/Amazons.lud");

    std::string game_name = test_game.getName();

    std::cout<<game_name<<std::endl;

    int stateFlgs = test_game.stateFlags();

    std::cout<<stateFlgs<<std::endl;

    Mode m = test_game.mode();

    int numPlys = m.numPlayers();

    std::cout<<numPlys<<std::endl;

    Trial t = Trial(env,test_game);

    Context c = Context(env,test_game,t);

    test_game.start(c);

	return 1;
}
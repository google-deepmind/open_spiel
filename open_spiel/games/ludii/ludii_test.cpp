#include "jni_utils.h"
#include "game_loader.h"
#include "game.h"
#include "trial.h"
#include "context.h"
#include "state.h"
#include "container_state.h"
#include "moves.h"
#include "region.h"
#include "chunk_set.h"
#include <iostream>
#include <vector>
#include <string>
#include "move.h"

int main(){

	JNIUtils test_utils = JNIUtils("/home/alex/Downloads/Ludii-0.3.0.jar");
	JNIEnv *env = test_utils.GetEnv();

	GameLoader gameLoader = GameLoader(env);

	std::vector<std::string> game_names = gameLoader.ListGames();

	for (std::vector<std::string>::const_iterator i = game_names.begin(); i != game_names.end(); ++i)
    	std::cout << *i << ' '<<std::endl;

    Game test_game = gameLoader.LoadGame("board/space/blocking/Amazons.lud");

    test_game.Create(0);

    std::string game_name = test_game.GetName();

    std::cout<<game_name<<std::endl;

    int stateFlgs = test_game.StateFlags();

    std::cout<<stateFlgs<<std::endl;

    Mode m = test_game.GetMode();

    int numPlys = m.NumPlayers();

    std::cout<<numPlys<<std::endl;

    Trial t = Trial(env,test_game);

    Context c = Context(env,test_game,t);

    test_game.Start(c);

    State s = t.GetState();

    std::vector<ContainerState> c_states = s.ContainerStates();

    bool is_ov = t.Over();

    int mo = s.Mover();

    ContainerState cs = c_states[0];

    Region r = cs.Empty();

    ChunkSet chunks = r.BitSet();

    std::cout<<chunks.Print()<<std::endl;

    ChunkSet chunks2 = cs.CloneWho();

    std::cout<<chunks2.ToChunkString()<<std::endl;

    ChunkSet chunks3 = cs.CloneWhat();

    Moves ms = test_game.GetMoves(c);

    std::vector<Move> mv = ms.GetMoves();

    Move move_after_apply = test_game.Apply(c,mv[0]);


	return 1;
}
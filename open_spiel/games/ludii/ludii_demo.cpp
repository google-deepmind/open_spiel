// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

// path to downloaded Ludii jar
#define JAR_LOCATION "/home/alex/Downloads/Ludii-0.3.0.jar"

int main()
{

    //launch JVM with the Ludii jar on the classpath
    JNIUtils test_utils = JNIUtils(JAR_LOCATION);

    //Get JNI environment variable
    JNIEnv *env = test_utils.GetEnv();

    //Ludii GameLoader object
    GameLoader gameLoader = GameLoader(env);

    //List Ludii games
    std::vector<std::string> game_names = gameLoader.ListGames();

    std::cout << "listing games" << std::endl;
    for (std::vector<std::string>::const_iterator i = game_names.begin(); i != game_names.end(); ++i)
        std::cout << *i << ' ' << std::endl;

    //Load a Ludii game
    Game test_game = gameLoader.LoadGame("board/space/blocking/Amazons.lud");

    //Test some Ludii API calls
    test_game.Create(0);

    int stateFlgs = test_game.StateFlags();
    std::cout << "state flags: " << stateFlgs << std::endl;

    Mode m = test_game.GetMode();

    int numPlys = m.NumPlayers();
    std::cout << "number of players: " << numPlys << std::endl;

    Trial t = Trial(env, test_game);

    Context c = Context(env, test_game, t);

    test_game.Start(c);

    State s = t.GetState();

    std::vector<ContainerState> c_states = s.ContainerStates();

    bool is_ov = t.Over();

    int mo = s.Mover();

    ContainerState cs = c_states[0];

    Region r = cs.Empty();

    ChunkSet chunks = r.BitSet();

    std::cout << "chunk set: " << chunks.Print() << std::endl;

    ChunkSet chunks2 = cs.CloneWho();

    ChunkSet chunks3 = cs.CloneWhat();

    Moves ms = test_game.GetMoves(c);

    //get the moves for the game
    std::vector<Move> mv = ms.GetMoves();

    //apply a move to the game
    Move move_after_apply = test_game.Apply(c, mv[0]);


    return 1;
}

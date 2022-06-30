// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "open_spiel/games/ludii/chunk_set.h"
#include "open_spiel/games/ludii/container_state.h"
#include "open_spiel/games/ludii/context.h"
#include "open_spiel/games/ludii/game.h"
#include "open_spiel/games/ludii/game_loader.h"
#include "open_spiel/games/ludii/jni_utils.h"
#include "open_spiel/games/ludii/move.h"
#include "open_spiel/games/ludii/moves.h"
#include "open_spiel/games/ludii/region.h"
#include "open_spiel/games/ludii/state.h"
#include "open_spiel/games/ludii/trial.h"

namespace ludii = open_spiel::ludii;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: ludii_demo <Ludii jar file>" << std::endl;
    exit(-1);
  }

  // launch JVM with the Ludii jar on the classpath
  std::cout << "Loading jar: " << argv[1] << std::endl;
  ludii::JNIUtils test_utils = ludii::JNIUtils(argv[1]);

  // Get JNI environment variable
  JNIEnv* env = test_utils.GetEnv();

  // Ludii GameLoader object
  ludii::GameLoader gameLoader = ludii::GameLoader(env);

  // List Ludii games
  std::vector<std::string> game_names = gameLoader.ListGames();

  std::cout << "listing games" << std::endl;
  for (std::vector<std::string>::const_iterator i = game_names.begin();
       i != game_names.end(); ++i)
    std::cout << *i << ' ' << std::endl;

  // Load a Ludii game
  ludii::Game test_game =
      gameLoader.LoadGame("board/space/blocking/Amazons.lud");

  // Test some Ludii API calls
  test_game.Create(0);

  int stateFlgs = test_game.StateFlags();
  std::cout << "state flags: " << stateFlgs << std::endl;

  ludii::Mode m = test_game.GetMode();

  int numPlys = m.NumPlayers();
  std::cout << "number of players: " << numPlys << std::endl;

  ludii::Trial t = ludii::Trial(env, test_game);

  ludii::Context c = ludii::Context(env, test_game, t);

  test_game.Start(c);

  ludii::State s = t.GetState();

  std::vector<ludii::ContainerState> c_states = s.ContainerStates();

  bool is_ov = t.Over();

  int mo = s.Mover();

  ludii::ContainerState cs = c_states[0];

  ludii::Region r = cs.Empty();

  ludii::ChunkSet chunks = r.BitSet();

  std::cout << "chunk set: " << chunks.Print() << std::endl;

  ludii::ChunkSet chunks2 = cs.CloneWho();

  ludii::ChunkSet chunks3 = cs.CloneWhat();

  ludii::Moves ms = test_game.GetMoves(c);

  // get the moves for the game
  std::vector<ludii::Move> mv = ms.GetMoves();

  // apply a move to the game
  ludii::Move move_after_apply = test_game.Apply(c, mv[0]);

  return 1;
}

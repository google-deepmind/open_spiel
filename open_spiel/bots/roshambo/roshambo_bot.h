// Copyright 2021 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_BOTS_ROSHAMBO_ROSHAMBO_BOT_H_
#define OPEN_SPIEL_BOTS_ROSHAMBO_ROSHAMBO_BOT_H_

// Bots from the International Roshambo Programming Competition.
// https://webdocs.cs.ualberta.ca/~darse/rsbpc.html
// This OpenSpiel bot provides an interface to all of the bots in the 1999
// competition, as well as Greenberg, the winner of the 2000 competition,
// written by Andrzej Nagorko.
// http://www.mathpuzzle.com/older.htm
// http://www.mathpuzzle.com/greenberg.c

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace roshambo {

// The underlying C code requires that the number of throws in a game be
// specified at compile time. Changing it requires modifying the file
// rsb-ts1-modified.c. Set the constant 'trials' on line 42 to the desired
// number of throws. Then set kNumThrows below to the same number, and rebuild
// OpenSpiel by running the script build_and_run_tests.sh.

// Note that in his discussion of the results of the first competition, Darse
// Billings observed that match length was not particularly important: "The
// results were remarkably robust, and increasing the match length to 10000
// turns or decreasing it to 400 turns had a negligible effect."
// https://webdocs.cs.ualberta.ca/~darse/rsb-results1.html
inline constexpr int kNumThrows = 1000;
inline constexpr int kNumBots = 43;

class RoshamboBot : public Bot {
 public:
  explicit RoshamboBot(int player_id, std::string bot_name);
  Action Step(const State& state) override;

 private:
  Player player_id_;
  Player opponent_id_;
  std::string bot_name_;
};

std::unique_ptr<Bot> MakeRoshamboBot(int player_id, std::string bot_name);
std::vector<std::string> RoshamboBotNames();

}  // namespace roshambo
}  // namespace open_spiel

// Bots use these global arrays to inform their decisions.
// Element 0 is the number of rounds so far in the match.
// Element i is the action taken on turn i (1 <= i <= kNumThrows)
extern "C" int ROSHAMBO_BOT_my_history[open_spiel::roshambo::kNumThrows + 1];
extern "C" int ROSHAMBO_BOT_opp_history[open_spiel::roshambo::kNumThrows + 1];
namespace roshambo_tournament {
extern std::map<std::string, std::function<int()>> bot_map;
}

#endif  // OPEN_SPIEL_BOTS_ROSHAMBO_ROSHAMBO_BOT_H_

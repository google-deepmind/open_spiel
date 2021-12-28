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

#include <charconv>

#include "absl/strings/escaping.h"
#include "open_spiel/spiel.h"

// Example implementation of the random bot for HIG competition.
// The bot must strictly follow the communication protocol via stdin/stdout,
// but it can print any message to stderr for debugging.

namespace open_spiel {
namespace higc {

void RandomBotMainLoop() {
  std::mt19937 rng;

  // Read the current setup.
  std::string game_name;
  int play_as;
  std::cin >> game_name >> play_as;

  std::cerr << game_name << ' ' << play_as
            << std::endl;  // For debugging purposes.

  // Load the provided game.
  std::shared_ptr<const Game> game = LoadGame(game_name);

  // Observations will be received later from the referee.
  // The referee factors the observation into public (common knowledge across
  // all players) and private parts.
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicObsType, {});
  std::shared_ptr<Observer> private_observer =
      game->MakeObserver(kPrivateObsType, {});
  Observation public_observation(*game, public_observer);
  Observation private_observation(*game, private_observer);

  // Now there is 5 secs warm-up time that could be used for loading relevant
  // supplementary data. All data can be read/written from persistent /data
  // directory mounted from an external storage.
  std::cout << "ready" << std::endl;

  // Loop per match. This loop will end when referee instructs the player to do
  // so.
  while (true) {
    // Acknowledge the match started.
    std::cout << "start" << std::endl;

    // This is just a placeholder for other implementations -- we do not use
    // state in random agent, as it receives list of actions it can pick from.
    std::unique_ptr<State> state = game->NewInitialState();

    std::string message;
    while (true) {                      // Loop per state in match.
      std::getline(std::cin, message);  // Read message from the referee.
      if (message.empty()) continue;
      std::cerr << message << std::endl;  // For debugging purposes.

      if (message == "tournament over") {
        // The tournament is now over: there is 60 sec shutdown time
        // available for processing tournament results by the agent,
        // for example to update supplementary data.
        std::cout << "tournament over" << std::endl;
        std::exit(0);
      }

      if (message.rfind("match over", 0) == 0) {
        // The full message has format "game over 123"
        // where 123 is the final float reward received by this bot.
        //
        // Note that this message does not necessarily mean the match
        // reached a terminal state: if opponent crashed / violated
        // rules, the match will be over as well.
        std::cout << "match over" << std::endl;
        break;
      }

      // Regular message: a public and private observation followed by
      // a list of legal actions (if the bot should be acting).
      std::vector<absl::string_view> xs = absl::StrSplit(message, ' ');
      SPIEL_CHECK_GE(xs.size(), 2);
      std::vector<Action> legal_actions;
      for (int i = 0; i < xs.size(); ++i) {
        absl::string_view x = xs[i];
        if (i <= 1) {  // Observations.
          std::string decoded;
          absl::Base64Unescape(x, &decoded);
          if (i == 0)
            public_observation.Decompress(decoded);
          else if (i == 1)
            private_observation.Decompress(decoded);
        } else {  // Legal actions.
          Action a;
          auto [p, ec] = std::from_chars(x.begin(), x.end(), a);
          SPIEL_CHECK_TRUE(p == x.end());
          legal_actions.push_back(a);
        }
      }

      const bool should_act = !legal_actions.empty();
      if (should_act) {
        std::uniform_int_distribution<int> dist(0, legal_actions.size() - 1);
        std::cout << legal_actions[dist(rng)] << std::endl;
      } else {
        // Pondering phase, i.e. thinking when the bot is not acting.
        // The time limit is always at least 0.2s, but can be longer,
        // up to 5s, depending on how long the opponent thinks.
        std::cout << "ponder" << std::endl;  // This bot does not ponder.
      }
    }

    SPIEL_CHECK_EQ(message.rfind("match over", 0), 0);
    int score = 0;
    std::from_chars(message.data() + 11, message.data() + message.size(),
                    score);
    std::cerr << "score: " << score << std::endl;
  }
}

}  // namespace higc
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::higc::RandomBotMainLoop(); }

#include "open_spiel/games/universal_poker/repeated_poker.h"

#include <memory>

#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/canonical_game_strings.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/utils/init.h"

namespace open_spiel {
namespace universal_poker {
namespace repeated_poker {
namespace {

namespace testing = open_spiel::testing;

void BasicRepeatedPokerTest() {
  std::shared_ptr<const Game> game =
      LoadGame("repeated_poker",
               {{"max_num_hands", GameParameter(100)},
                {"reset_stacks", GameParameter(true)},
                {"rotate_dealer", GameParameter(true)},
                {"universal_poker_game_string",
                 GameParameterFromString(open_spiel::HunlGameString(
                     "fullgame"))}});
  std::unique_ptr<State> state = game->NewInitialState();
  testing::RandomSimTest(*game, 5);
}

}  // namespace
}  // namespace repeated_poker
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::Init("", &argc, &argv, true);
  absl::ParseCommandLine(argc, argv);
  open_spiel::universal_poker::repeated_poker::BasicRepeatedPokerTest();
}

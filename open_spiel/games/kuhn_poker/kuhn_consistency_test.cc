#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/kuhn_poker/kuhn_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace kuhn_poker {
void PrintObservations() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  std::unique_ptr<State> state = game->NewInitialState();

  // Deal
  state->ApplyAction(0); // Player 0 gets card 0
  state->ApplyAction(1); // Player 1 gets card 1

  std::cout << "State: " << state->ToString() << std::endl;

  // Imperfect Recall Public Info
  // Default is Public Info=True, Perfect Recall=False, Private Info=Single
  // Player
  std::cout << "Observation (Default): " << state->ObservationString(0)
            << std::endl;

  // Check Bet
  state->ApplyAction(1); // Bet
  std::cout << "State: " << state->ToString() << std::endl;
  std::cout << "State: " << state->ToString() << std::endl;

  // 1. Perfect Recall Observer (should use 'b'/'p')
  auto obs_perfect = game->MakeObserver(
      IIGObservationType{true, true, PrivateInfoType::kSinglePlayer}, {});
  std::cout << "Perfect Recall: " << obs_perfect->StringFrom(*state, 0)
            << std::endl;

  // 2. Public Info Only Imperfect Recall (This logic currently uses
  // "Bet"/"Pass")
  auto obs_public_imperfect = game->MakeObserver(
      IIGObservationType{true, false, PrivateInfoType::kNone}, {});
  std::cout << "Public Imperfect: "
            << obs_public_imperfect->StringFrom(*state, 0) << std::endl;
}
} // namespace kuhn_poker
} // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::kuhn_poker::PrintObservations();
  return 0;
}

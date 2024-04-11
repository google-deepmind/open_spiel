#include "open_spiel/spiel.h"
#include "open_spiel/tests/console_play_test.h"

int main() {
  // Create a game instance (e.g., Tic-Tac-Toe)
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("tic_tac_toe");

  // Call ConsolePlayTest to play the game interactively via the console
  open_spiel::testing::ConsolePlayTest(*game);

  return 0;
}

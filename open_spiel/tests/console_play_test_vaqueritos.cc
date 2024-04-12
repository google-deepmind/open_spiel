#include "open_spiel/tests/console_play_test.h"

void TestMyGame() {
  ConsolePlayTest my_game_test("my_game");
  my_game_test.Run(100);  // Run 100 random simulation tests
}
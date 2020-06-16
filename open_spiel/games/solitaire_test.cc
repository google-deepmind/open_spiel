#include "cassert"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "solitaire.h"

namespace open_spiel::solitaire {
namespace {

namespace testing = open_spiel::testing;

void BasicSolitaireTests() {
  testing::LoadGameTest("solitaire");
  testing::RandomSimTest(*LoadGame("solitaire"), 100);
}

void TestMoveActionId() {
  std::vector<Move> valid_moves = {
      // region List of valid moves in order
      // Spades

      Move(kRank2, kSuitSpades, kRank3, kSuitSpades),
      Move(kRank2, kSuitSpades, kRankA, kSuitHearts),
      Move(kRank2, kSuitSpades, kRankA, kSuitDiamonds),

      Move(kRank3, kSuitSpades, kRank4, kSuitSpades),
      Move(kRank3, kSuitSpades, kRank2, kSuitHearts),
      Move(kRank3, kSuitSpades, kRank2, kSuitDiamonds),

      Move(kRank4, kSuitSpades, kRank5, kSuitSpades),
      Move(kRank4, kSuitSpades, kRank3, kSuitHearts),
      Move(kRank4, kSuitSpades, kRank3, kSuitDiamonds),

      Move(kRank5, kSuitSpades, kRank6, kSuitSpades),
      Move(kRank5, kSuitSpades, kRank4, kSuitHearts),
      Move(kRank5, kSuitSpades, kRank4, kSuitDiamonds),

      Move(kRank6, kSuitSpades, kRank7, kSuitSpades),
      Move(kRank6, kSuitSpades, kRank5, kSuitHearts),
      Move(kRank6, kSuitSpades, kRank5, kSuitDiamonds),

      Move(kRank7, kSuitSpades, kRank8, kSuitSpades),
      Move(kRank7, kSuitSpades, kRank6, kSuitHearts),
      Move(kRank7, kSuitSpades, kRank6, kSuitDiamonds),

      Move(kRank8, kSuitSpades, kRank9, kSuitSpades),
      Move(kRank8, kSuitSpades, kRank7, kSuitHearts),
      Move(kRank8, kSuitSpades, kRank7, kSuitDiamonds),

      Move(kRank9, kSuitSpades, kRankT, kSuitSpades),
      Move(kRank9, kSuitSpades, kRank8, kSuitHearts),
      Move(kRank9, kSuitSpades, kRank8, kSuitDiamonds),

      Move(kRankT, kSuitSpades, kRankJ, kSuitSpades),
      Move(kRankT, kSuitSpades, kRank9, kSuitHearts),
      Move(kRankT, kSuitSpades, kRank9, kSuitDiamonds),

      Move(kRankJ, kSuitSpades, kRankQ, kSuitSpades),
      Move(kRankJ, kSuitSpades, kRankT, kSuitHearts),
      Move(kRankJ, kSuitSpades, kRankT, kSuitDiamonds),

      Move(kRankQ, kSuitSpades, kRankK, kSuitSpades),
      Move(kRankQ, kSuitSpades, kRankJ, kSuitHearts),
      Move(kRankQ, kSuitSpades, kRankJ, kSuitDiamonds),

      // Hearts

      Move(kRank2, kSuitHearts, kRank3, kSuitHearts),
      Move(kRank2, kSuitHearts, kRankA, kSuitSpades),
      Move(kRank2, kSuitHearts, kRankA, kSuitClubs),

      Move(kRank3, kSuitHearts, kRank4, kSuitHearts),
      Move(kRank3, kSuitHearts, kRank2, kSuitSpades),
      Move(kRank3, kSuitHearts, kRank2, kSuitClubs),

      Move(kRank4, kSuitHearts, kRank5, kSuitHearts),
      Move(kRank4, kSuitHearts, kRank3, kSuitSpades),
      Move(kRank4, kSuitHearts, kRank3, kSuitClubs),

      Move(kRank5, kSuitHearts, kRank6, kSuitHearts),
      Move(kRank5, kSuitHearts, kRank4, kSuitSpades),
      Move(kRank5, kSuitHearts, kRank4, kSuitClubs),

      Move(kRank6, kSuitHearts, kRank7, kSuitHearts),
      Move(kRank6, kSuitHearts, kRank5, kSuitSpades),
      Move(kRank6, kSuitHearts, kRank5, kSuitClubs),

      Move(kRank7, kSuitHearts, kRank8, kSuitHearts),
      Move(kRank7, kSuitHearts, kRank6, kSuitSpades),
      Move(kRank7, kSuitHearts, kRank6, kSuitClubs),

      Move(kRank8, kSuitHearts, kRank9, kSuitHearts),
      Move(kRank8, kSuitHearts, kRank7, kSuitSpades),
      Move(kRank8, kSuitHearts, kRank7, kSuitClubs),

      Move(kRank9, kSuitHearts, kRankT, kSuitHearts),
      Move(kRank9, kSuitHearts, kRank8, kSuitSpades),
      Move(kRank9, kSuitHearts, kRank8, kSuitClubs),

      Move(kRankT, kSuitHearts, kRankJ, kSuitHearts),
      Move(kRankT, kSuitHearts, kRank9, kSuitSpades),
      Move(kRankT, kSuitHearts, kRank9, kSuitClubs),

      Move(kRankJ, kSuitHearts, kRankQ, kSuitHearts),
      Move(kRankJ, kSuitHearts, kRankT, kSuitSpades),
      Move(kRankJ, kSuitHearts, kRankT, kSuitClubs),

      Move(kRankQ, kSuitHearts, kRankK, kSuitHearts),
      Move(kRankQ, kSuitHearts, kRankJ, kSuitSpades),
      Move(kRankQ, kSuitHearts, kRankJ, kSuitClubs),

      // Clubs

      Move(kRank2, kSuitClubs, kRank3, kSuitClubs),
      Move(kRank2, kSuitClubs, kRankA, kSuitHearts),
      Move(kRank2, kSuitClubs, kRankA, kSuitDiamonds),

      Move(kRank3, kSuitClubs, kRank4, kSuitClubs),
      Move(kRank3, kSuitClubs, kRank2, kSuitHearts),
      Move(kRank3, kSuitClubs, kRank2, kSuitDiamonds),

      Move(kRank4, kSuitClubs, kRank5, kSuitClubs),
      Move(kRank4, kSuitClubs, kRank3, kSuitHearts),
      Move(kRank4, kSuitClubs, kRank3, kSuitDiamonds),

      Move(kRank5, kSuitClubs, kRank6, kSuitClubs),
      Move(kRank5, kSuitClubs, kRank4, kSuitHearts),
      Move(kRank5, kSuitClubs, kRank4, kSuitDiamonds),

      Move(kRank6, kSuitClubs, kRank7, kSuitClubs),
      Move(kRank6, kSuitClubs, kRank5, kSuitHearts),
      Move(kRank6, kSuitClubs, kRank5, kSuitDiamonds),

      Move(kRank7, kSuitClubs, kRank8, kSuitClubs),
      Move(kRank7, kSuitClubs, kRank6, kSuitHearts),
      Move(kRank7, kSuitClubs, kRank6, kSuitDiamonds),

      Move(kRank8, kSuitClubs, kRank9, kSuitClubs),
      Move(kRank8, kSuitClubs, kRank7, kSuitHearts),
      Move(kRank8, kSuitClubs, kRank7, kSuitDiamonds),

      Move(kRank9, kSuitClubs, kRankT, kSuitClubs),
      Move(kRank9, kSuitClubs, kRank8, kSuitHearts),
      Move(kRank9, kSuitClubs, kRank8, kSuitDiamonds),

      Move(kRankT, kSuitClubs, kRankJ, kSuitClubs),
      Move(kRankT, kSuitClubs, kRank9, kSuitHearts),
      Move(kRankT, kSuitClubs, kRank9, kSuitDiamonds),

      Move(kRankJ, kSuitClubs, kRankQ, kSuitClubs),
      Move(kRankJ, kSuitClubs, kRankT, kSuitHearts),
      Move(kRankJ, kSuitClubs, kRankT, kSuitDiamonds),

      Move(kRankQ, kSuitClubs, kRankK, kSuitClubs),
      Move(kRankQ, kSuitClubs, kRankJ, kSuitHearts),
      Move(kRankQ, kSuitClubs, kRankJ, kSuitDiamonds),

      // Diamonds

      Move(kRank2, kSuitDiamonds, kRank3, kSuitDiamonds),
      Move(kRank2, kSuitDiamonds, kRankA, kSuitSpades),
      Move(kRank2, kSuitDiamonds, kRankA, kSuitClubs),

      Move(kRank3, kSuitDiamonds, kRank4, kSuitDiamonds),
      Move(kRank3, kSuitDiamonds, kRank2, kSuitSpades),
      Move(kRank3, kSuitDiamonds, kRank2, kSuitClubs),

      Move(kRank4, kSuitDiamonds, kRank5, kSuitDiamonds),
      Move(kRank4, kSuitDiamonds, kRank3, kSuitSpades),
      Move(kRank4, kSuitDiamonds, kRank3, kSuitClubs),

      Move(kRank5, kSuitDiamonds, kRank6, kSuitDiamonds),
      Move(kRank5, kSuitDiamonds, kRank4, kSuitSpades),
      Move(kRank5, kSuitDiamonds, kRank4, kSuitClubs),

      Move(kRank6, kSuitDiamonds, kRank7, kSuitDiamonds),
      Move(kRank6, kSuitDiamonds, kRank5, kSuitSpades),
      Move(kRank6, kSuitDiamonds, kRank5, kSuitClubs),

      Move(kRank7, kSuitDiamonds, kRank8, kSuitDiamonds),
      Move(kRank7, kSuitDiamonds, kRank6, kSuitSpades),
      Move(kRank7, kSuitDiamonds, kRank6, kSuitClubs),

      Move(kRank8, kSuitDiamonds, kRank9, kSuitDiamonds),
      Move(kRank8, kSuitDiamonds, kRank7, kSuitSpades),
      Move(kRank8, kSuitDiamonds, kRank7, kSuitClubs),

      Move(kRank9, kSuitDiamonds, kRankT, kSuitDiamonds),
      Move(kRank9, kSuitDiamonds, kRank8, kSuitSpades),
      Move(kRank9, kSuitDiamonds, kRank8, kSuitClubs),

      Move(kRankT, kSuitDiamonds, kRankJ, kSuitDiamonds),
      Move(kRankT, kSuitDiamonds, kRank9, kSuitSpades),
      Move(kRankT, kSuitDiamonds, kRank9, kSuitClubs),

      Move(kRankJ, kSuitDiamonds, kRankQ, kSuitDiamonds),
      Move(kRankJ, kSuitDiamonds, kRankT, kSuitSpades),
      Move(kRankJ, kSuitDiamonds, kRankT, kSuitClubs),

      Move(kRankQ, kSuitDiamonds, kRankK, kSuitDiamonds),
      Move(kRankQ, kSuitDiamonds, kRankJ, kSuitSpades),
      Move(kRankQ, kSuitDiamonds, kRankJ, kSuitClubs),

      // Special

      Move(kRankNone, kSuitSpades, kRankA, kSuitSpades),
      Move(kRankNone, kSuitHearts, kRankA, kSuitHearts),
      Move(kRankNone, kSuitClubs, kRankA, kSuitClubs),
      Move(kRankNone, kSuitDiamonds, kRankA, kSuitDiamonds),

      Move(kRankNone, kSuitNone, kRankK, kSuitSpades),
      Move(kRankNone, kSuitNone, kRankK, kSuitHearts),
      Move(kRankNone, kSuitNone, kRankK, kSuitClubs),
      Move(kRankNone, kSuitNone, kRankK, kSuitDiamonds),

      Move(kRankA, kSuitSpades, kRank2, kSuitSpades),
      Move(kRankA, kSuitHearts, kRank2, kSuitHearts),
      Move(kRankA, kSuitClubs, kRank2, kSuitClubs),
      Move(kRankA, kSuitDiamonds, kRank2, kSuitDiamonds),

      Move(kRankK, kSuitSpades, kRankQ, kSuitHearts),
      Move(kRankK, kSuitSpades, kRankQ, kSuitDiamonds),
      Move(kRankK, kSuitHearts, kRankQ, kSuitSpades),
      Move(kRankK, kSuitHearts, kRankQ, kSuitClubs),
      Move(kRankK, kSuitClubs, kRankQ, kSuitHearts),
      Move(kRankK, kSuitClubs, kRankQ, kSuitDiamonds),
      Move(kRankK, kSuitDiamonds, kRankQ, kSuitSpades),
      Move(kRankK, kSuitDiamonds, kRankQ, kSuitClubs),
      // endregion
  };

  for (const auto &move : valid_moves) {
    std::cout << move.ToString() << " == " << Move(move.ActionId()).ToString() << std::endl;
    assert(move.ToString() == Move(move.ActionId()).ToString());
  }
}

} // namespace
} // namespace open_spiel::solitaire

int main(int argc, char **argv) {
  open_spiel::solitaire::TestMoveActionId();
  open_spiel::solitaire::BasicSolitaireTests();
}
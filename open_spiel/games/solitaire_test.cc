#include "solitaire.h"

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel::solitaire {
namespace {

namespace testing = open_spiel::testing;

void BasicSolitaireTests() {
  testing::LoadGameTest("solitaire");
  testing::RandomSimTest(*LoadGame("solitaire"), 100);
}

void TestMoveActionId() {
  // TODO: Use a for loop here
  std::vector<Move> valid_moves = {
      // region List of valid moves in order
      // Spades

      Move(RankType::k2, SuitType::kSpades, RankType::k3, SuitType::kSpades),
      Move(RankType::k2, SuitType::kSpades, RankType::kA, SuitType::kHearts),
      Move(RankType::k2, SuitType::kSpades, RankType::kA, SuitType::kDiamonds),

      Move(RankType::k3, SuitType::kSpades, RankType::k4, SuitType::kSpades),
      Move(RankType::k3, SuitType::kSpades, RankType::k2, SuitType::kHearts),
      Move(RankType::k3, SuitType::kSpades, RankType::k2, SuitType::kDiamonds),

      Move(RankType::k4, SuitType::kSpades, RankType::k5, SuitType::kSpades),
      Move(RankType::k4, SuitType::kSpades, RankType::k3, SuitType::kHearts),
      Move(RankType::k4, SuitType::kSpades, RankType::k3, SuitType::kDiamonds),

      Move(RankType::k5, SuitType::kSpades, RankType::k6, SuitType::kSpades),
      Move(RankType::k5, SuitType::kSpades, RankType::k4, SuitType::kHearts),
      Move(RankType::k5, SuitType::kSpades, RankType::k4, SuitType::kDiamonds),

      Move(RankType::k6, SuitType::kSpades, RankType::k7, SuitType::kSpades),
      Move(RankType::k6, SuitType::kSpades, RankType::k5, SuitType::kHearts),
      Move(RankType::k6, SuitType::kSpades, RankType::k5, SuitType::kDiamonds),

      Move(RankType::k7, SuitType::kSpades, RankType::k8, SuitType::kSpades),
      Move(RankType::k7, SuitType::kSpades, RankType::k6, SuitType::kHearts),
      Move(RankType::k7, SuitType::kSpades, RankType::k6, SuitType::kDiamonds),

      Move(RankType::k8, SuitType::kSpades, RankType::k9, SuitType::kSpades),
      Move(RankType::k8, SuitType::kSpades, RankType::k7, SuitType::kHearts),
      Move(RankType::k8, SuitType::kSpades, RankType::k7, SuitType::kDiamonds),

      Move(RankType::k9, SuitType::kSpades, RankType::kT, SuitType::kSpades),
      Move(RankType::k9, SuitType::kSpades, RankType::k8, SuitType::kHearts),
      Move(RankType::k9, SuitType::kSpades, RankType::k8, SuitType::kDiamonds),

      Move(RankType::kT, SuitType::kSpades, RankType::kJ, SuitType::kSpades),
      Move(RankType::kT, SuitType::kSpades, RankType::k9, SuitType::kHearts),
      Move(RankType::kT, SuitType::kSpades, RankType::k9, SuitType::kDiamonds),

      Move(RankType::kJ, SuitType::kSpades, RankType::kQ, SuitType::kSpades),
      Move(RankType::kJ, SuitType::kSpades, RankType::kT, SuitType::kHearts),
      Move(RankType::kJ, SuitType::kSpades, RankType::kT, SuitType::kDiamonds),

      Move(RankType::kQ, SuitType::kSpades, RankType::kK, SuitType::kSpades),
      Move(RankType::kQ, SuitType::kSpades, RankType::kJ, SuitType::kHearts),
      Move(RankType::kQ, SuitType::kSpades, RankType::kJ, SuitType::kDiamonds),

      // Hearts

      Move(RankType::k2, SuitType::kHearts, RankType::k3, SuitType::kHearts),
      Move(RankType::k2, SuitType::kHearts, RankType::kA, SuitType::kSpades),
      Move(RankType::k2, SuitType::kHearts, RankType::kA, SuitType::kClubs),

      Move(RankType::k3, SuitType::kHearts, RankType::k4, SuitType::kHearts),
      Move(RankType::k3, SuitType::kHearts, RankType::k2, SuitType::kSpades),
      Move(RankType::k3, SuitType::kHearts, RankType::k2, SuitType::kClubs),

      Move(RankType::k4, SuitType::kHearts, RankType::k5, SuitType::kHearts),
      Move(RankType::k4, SuitType::kHearts, RankType::k3, SuitType::kSpades),
      Move(RankType::k4, SuitType::kHearts, RankType::k3, SuitType::kClubs),

      Move(RankType::k5, SuitType::kHearts, RankType::k6, SuitType::kHearts),
      Move(RankType::k5, SuitType::kHearts, RankType::k4, SuitType::kSpades),
      Move(RankType::k5, SuitType::kHearts, RankType::k4, SuitType::kClubs),

      Move(RankType::k6, SuitType::kHearts, RankType::k7, SuitType::kHearts),
      Move(RankType::k6, SuitType::kHearts, RankType::k5, SuitType::kSpades),
      Move(RankType::k6, SuitType::kHearts, RankType::k5, SuitType::kClubs),

      Move(RankType::k7, SuitType::kHearts, RankType::k8, SuitType::kHearts),
      Move(RankType::k7, SuitType::kHearts, RankType::k6, SuitType::kSpades),
      Move(RankType::k7, SuitType::kHearts, RankType::k6, SuitType::kClubs),

      Move(RankType::k8, SuitType::kHearts, RankType::k9, SuitType::kHearts),
      Move(RankType::k8, SuitType::kHearts, RankType::k7, SuitType::kSpades),
      Move(RankType::k8, SuitType::kHearts, RankType::k7, SuitType::kClubs),

      Move(RankType::k9, SuitType::kHearts, RankType::kT, SuitType::kHearts),
      Move(RankType::k9, SuitType::kHearts, RankType::k8, SuitType::kSpades),
      Move(RankType::k9, SuitType::kHearts, RankType::k8, SuitType::kClubs),

      Move(RankType::kT, SuitType::kHearts, RankType::kJ, SuitType::kHearts),
      Move(RankType::kT, SuitType::kHearts, RankType::k9, SuitType::kSpades),
      Move(RankType::kT, SuitType::kHearts, RankType::k9, SuitType::kClubs),

      Move(RankType::kJ, SuitType::kHearts, RankType::kQ, SuitType::kHearts),
      Move(RankType::kJ, SuitType::kHearts, RankType::kT, SuitType::kSpades),
      Move(RankType::kJ, SuitType::kHearts, RankType::kT, SuitType::kClubs),

      Move(RankType::kQ, SuitType::kHearts, RankType::kK, SuitType::kHearts),
      Move(RankType::kQ, SuitType::kHearts, RankType::kJ, SuitType::kSpades),
      Move(RankType::kQ, SuitType::kHearts, RankType::kJ, SuitType::kClubs),

      // Clubs

      Move(RankType::k2, SuitType::kClubs, RankType::k3, SuitType::kClubs),
      Move(RankType::k2, SuitType::kClubs, RankType::kA, SuitType::kHearts),
      Move(RankType::k2, SuitType::kClubs, RankType::kA, SuitType::kDiamonds),

      Move(RankType::k3, SuitType::kClubs, RankType::k4, SuitType::kClubs),
      Move(RankType::k3, SuitType::kClubs, RankType::k2, SuitType::kHearts),
      Move(RankType::k3, SuitType::kClubs, RankType::k2, SuitType::kDiamonds),

      Move(RankType::k4, SuitType::kClubs, RankType::k5, SuitType::kClubs),
      Move(RankType::k4, SuitType::kClubs, RankType::k3, SuitType::kHearts),
      Move(RankType::k4, SuitType::kClubs, RankType::k3, SuitType::kDiamonds),

      Move(RankType::k5, SuitType::kClubs, RankType::k6, SuitType::kClubs),
      Move(RankType::k5, SuitType::kClubs, RankType::k4, SuitType::kHearts),
      Move(RankType::k5, SuitType::kClubs, RankType::k4, SuitType::kDiamonds),

      Move(RankType::k6, SuitType::kClubs, RankType::k7, SuitType::kClubs),
      Move(RankType::k6, SuitType::kClubs, RankType::k5, SuitType::kHearts),
      Move(RankType::k6, SuitType::kClubs, RankType::k5, SuitType::kDiamonds),

      Move(RankType::k7, SuitType::kClubs, RankType::k8, SuitType::kClubs),
      Move(RankType::k7, SuitType::kClubs, RankType::k6, SuitType::kHearts),
      Move(RankType::k7, SuitType::kClubs, RankType::k6, SuitType::kDiamonds),

      Move(RankType::k8, SuitType::kClubs, RankType::k9, SuitType::kClubs),
      Move(RankType::k8, SuitType::kClubs, RankType::k7, SuitType::kHearts),
      Move(RankType::k8, SuitType::kClubs, RankType::k7, SuitType::kDiamonds),

      Move(RankType::k9, SuitType::kClubs, RankType::kT, SuitType::kClubs),
      Move(RankType::k9, SuitType::kClubs, RankType::k8, SuitType::kHearts),
      Move(RankType::k9, SuitType::kClubs, RankType::k8, SuitType::kDiamonds),

      Move(RankType::kT, SuitType::kClubs, RankType::kJ, SuitType::kClubs),
      Move(RankType::kT, SuitType::kClubs, RankType::k9, SuitType::kHearts),
      Move(RankType::kT, SuitType::kClubs, RankType::k9, SuitType::kDiamonds),

      Move(RankType::kJ, SuitType::kClubs, RankType::kQ, SuitType::kClubs),
      Move(RankType::kJ, SuitType::kClubs, RankType::kT, SuitType::kHearts),
      Move(RankType::kJ, SuitType::kClubs, RankType::kT, SuitType::kDiamonds),

      Move(RankType::kQ, SuitType::kClubs, RankType::kK, SuitType::kClubs),
      Move(RankType::kQ, SuitType::kClubs, RankType::kJ, SuitType::kHearts),
      Move(RankType::kQ, SuitType::kClubs, RankType::kJ, SuitType::kDiamonds),

      // Diamonds

      Move(RankType::k2, SuitType::kDiamonds, RankType::k3, SuitType::kDiamonds),
      Move(RankType::k2, SuitType::kDiamonds, RankType::kA, SuitType::kSpades),
      Move(RankType::k2, SuitType::kDiamonds, RankType::kA, SuitType::kClubs),

      Move(RankType::k3, SuitType::kDiamonds, RankType::k4, SuitType::kDiamonds),
      Move(RankType::k3, SuitType::kDiamonds, RankType::k2, SuitType::kSpades),
      Move(RankType::k3, SuitType::kDiamonds, RankType::k2, SuitType::kClubs),

      Move(RankType::k4, SuitType::kDiamonds, RankType::k5, SuitType::kDiamonds),
      Move(RankType::k4, SuitType::kDiamonds, RankType::k3, SuitType::kSpades),
      Move(RankType::k4, SuitType::kDiamonds, RankType::k3, SuitType::kClubs),

      Move(RankType::k5, SuitType::kDiamonds, RankType::k6, SuitType::kDiamonds),
      Move(RankType::k5, SuitType::kDiamonds, RankType::k4, SuitType::kSpades),
      Move(RankType::k5, SuitType::kDiamonds, RankType::k4, SuitType::kClubs),

      Move(RankType::k6, SuitType::kDiamonds, RankType::k7, SuitType::kDiamonds),
      Move(RankType::k6, SuitType::kDiamonds, RankType::k5, SuitType::kSpades),
      Move(RankType::k6, SuitType::kDiamonds, RankType::k5, SuitType::kClubs),

      Move(RankType::k7, SuitType::kDiamonds, RankType::k8, SuitType::kDiamonds),
      Move(RankType::k7, SuitType::kDiamonds, RankType::k6, SuitType::kSpades),
      Move(RankType::k7, SuitType::kDiamonds, RankType::k6, SuitType::kClubs),

      Move(RankType::k8, SuitType::kDiamonds, RankType::k9, SuitType::kDiamonds),
      Move(RankType::k8, SuitType::kDiamonds, RankType::k7, SuitType::kSpades),
      Move(RankType::k8, SuitType::kDiamonds, RankType::k7, SuitType::kClubs),

      Move(RankType::k9, SuitType::kDiamonds, RankType::kT, SuitType::kDiamonds),
      Move(RankType::k9, SuitType::kDiamonds, RankType::k8, SuitType::kSpades),
      Move(RankType::k9, SuitType::kDiamonds, RankType::k8, SuitType::kClubs),

      Move(RankType::kT, SuitType::kDiamonds, RankType::kJ, SuitType::kDiamonds),
      Move(RankType::kT, SuitType::kDiamonds, RankType::k9, SuitType::kSpades),
      Move(RankType::kT, SuitType::kDiamonds, RankType::k9, SuitType::kClubs),

      Move(RankType::kJ, SuitType::kDiamonds, RankType::kQ, SuitType::kDiamonds),
      Move(RankType::kJ, SuitType::kDiamonds, RankType::kT, SuitType::kSpades),
      Move(RankType::kJ, SuitType::kDiamonds, RankType::kT, SuitType::kClubs),

      Move(RankType::kQ, SuitType::kDiamonds, RankType::kK, SuitType::kDiamonds),
      Move(RankType::kQ, SuitType::kDiamonds, RankType::kJ, SuitType::kSpades),
      Move(RankType::kQ, SuitType::kDiamonds, RankType::kJ, SuitType::kClubs),

      // Special

      Move(RankType::kNone, SuitType::kSpades, RankType::kA, SuitType::kSpades),
      Move(RankType::kNone, SuitType::kHearts, RankType::kA, SuitType::kHearts),
      Move(RankType::kNone, SuitType::kClubs, RankType::kA, SuitType::kClubs),
      Move(RankType::kNone, SuitType::kDiamonds, RankType::kA, SuitType::kDiamonds),

      Move(RankType::kNone, SuitType::kNone, RankType::kK, SuitType::kSpades),
      Move(RankType::kNone, SuitType::kNone, RankType::kK, SuitType::kHearts),
      Move(RankType::kNone, SuitType::kNone, RankType::kK, SuitType::kClubs),
      Move(RankType::kNone, SuitType::kNone, RankType::kK, SuitType::kDiamonds),

      Move(RankType::kA, SuitType::kSpades, RankType::k2, SuitType::kSpades),
      Move(RankType::kA, SuitType::kHearts, RankType::k2, SuitType::kHearts),
      Move(RankType::kA, SuitType::kClubs, RankType::k2, SuitType::kClubs),
      Move(RankType::kA, SuitType::kDiamonds, RankType::k2, SuitType::kDiamonds),

      Move(RankType::kK, SuitType::kSpades, RankType::kQ, SuitType::kHearts),
      Move(RankType::kK, SuitType::kSpades, RankType::kQ, SuitType::kDiamonds),
      Move(RankType::kK, SuitType::kHearts, RankType::kQ, SuitType::kSpades),
      Move(RankType::kK, SuitType::kHearts, RankType::kQ, SuitType::kClubs),
      Move(RankType::kK, SuitType::kClubs, RankType::kQ, SuitType::kHearts),
      Move(RankType::kK, SuitType::kClubs, RankType::kQ, SuitType::kDiamonds),
      Move(RankType::kK, SuitType::kDiamonds, RankType::kQ, SuitType::kSpades),
      Move(RankType::kK, SuitType::kDiamonds, RankType::kQ, SuitType::kClubs),
      // endregion
  };

  for (const auto &move : valid_moves) {
    std::cout << move.ToString() << " == " << Move(move.ActionId()).ToString()
              << std::endl;
    SPIEL_CHECK_EQ(move.ToString(), Move(move.ActionId()).ToString());
  }
}

}  // namespace
}  // namespace open_spiel::solitaire

int main(int argc, char **argv) {
  open_spiel::solitaire::TestMoveActionId();
  open_spiel::solitaire::BasicSolitaireTests();
}


#include "open_spiel/games/german_whist_foregame/german_whist_foregame.h"

#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace german_whist_foregame {
namespace {

namespace testing = open_spiel::testing;


void BasicGermanWhistForegameTests() {
    testing::LoadGameTest("german_whist_foregame");
    //testing::ChanceOutcomesTest(*LoadGame("german_whist_foregame"));
    testing::RandomSimTest(*LoadGame("german_whist_foregame"),100,false,true);
}



}  // namespace
}  // namespace GermanWhistForegame_
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::german_whist_foregame::BasicGermanWhistForegameTests();
  //open_spiel::testing::ResampleInfostateTest(*open_spiel::LoadGame("german_whist_foregame"),*num_sims=*10);
}

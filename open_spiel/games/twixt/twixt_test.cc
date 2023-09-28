
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace twixt {
namespace {

namespace testing = open_spiel::testing;

void BasicTwixTTests() {
  testing::LoadGameTest("twixt");
  testing::NoChanceOutcomesTest(*LoadGame("twixt"));
  testing::RandomSimTest(*LoadGame("twixt"), 100);
}

}  // namespace
}  // namespace twixt
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::twixt::BasicTwixTTests();
}



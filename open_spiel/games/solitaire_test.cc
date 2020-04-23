#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel::solitaire {
namespace {

    namespace testing = open_spiel::testing;

    void BasicSolitaireTests() {
        // Tests that the game can be loaded (i.e. LoadGame doesn't return nullptr)
        testing::LoadGameTest("solitaire");

        // Tests that there are chance outcomes
        //testing::ChanceOutcomesTest(*LoadGame("solitaire"));

        testing::RandomSimTest(*LoadGame("solitaire"), 20);
    }

} // namespace
} // namespace open_spiel::solitaire

int main(int argc, char** argv) { open_spiel::solitaire::BasicSolitaireTests(); }
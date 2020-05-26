#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "solitaire.h"

namespace open_spiel::solitaire {
namespace {

    namespace testing = open_spiel::testing;

    void BasicSolitaireTests() {
        testing::LoadGameTest("solitaire");
        testing::RandomSimTest(*LoadGame("solitaire"), 10);
    }

    void PrintLegalActions(std::unique_ptr<open_spiel::State> & state) {
        std::cout << "LEGAL ACTIONS : " << std::endl;
        for (auto & legal_action : state->LegalActions()) {
            std::cout << state->ActionToString(legal_action) << std::endl;
        }
    }

    void TestGame() {
        std::string sep = " ================================================================================ ";
        std::shared_ptr<const Game> game = LoadGame("solitaire");
        std::unique_ptr<open_spiel::State> state = game->NewInitialState();
        std::mt19937 rng;

        while (!state->IsTerminal()) {
            if (state->IsChanceNode()) {
                std::cout << "\nChance Node" << sep << std::endl;
                std::cout << state->ToString() << std::endl;
                std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
                Action action = open_spiel::SampleAction(outcomes, rng).first;
                std::cout << "SELECTED ACTION : " << state->ActionToString(action) << std::endl;
                state->ApplyAction(action);
            } else {
                std::cout << "\nDecision Node" << sep << std::endl;
                std::cout << state->ToString() << std::endl;
                PrintLegalActions(state);
                std::vector<Action> actions = state->LegalActions();
                std::uniform_int_distribution<int> dis(0, actions.size() - 1);
                Action action = actions[dis(rng)];
                std::cout << "SELECTED ACTION : " << state->ActionToString(action) << std::endl;
                state->ApplyAction(action);
            }
        }
    }

} // namespace
} // namespace open_spiel::solitaire

int main(int argc, char** argv) {
    open_spiel::solitaire::BasicSolitaireTests();
    //open_spiel::solitaire::TestGame();
}
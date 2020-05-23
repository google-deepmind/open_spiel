#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "solitaire.h"

#define BLUE "\033[34m"
#define RED  "\033[31m"

namespace open_spiel::solitaire {
namespace {

    namespace testing = open_spiel::testing;

    void BasicSolitaireTests() {
        testing::LoadGameTest("solitaire");
        testing::RandomSimTest(*LoadGame("solitaire"), 10);
    }

    void TestTableau() {
        std::cout << "\nTestTableau()" << std::endl;
        Tableau tableau = Tableau(kPile1stTableau);
        std::vector<Card> cards_to_add = {
                Card(true),
                Card(true),
                Card(true),
                Card(false, kS, kT, kTableau),
                Card(false, kH, k9, kTableau),
                Card(false, kC, k8, kTableau),
                Card(false, kD, k7, kTableau),
        };
        tableau.Extend(cards_to_add);
        std::cout << tableau.ToString(true) << std::endl;

        std::cout << "\nTableau::Targets()" << std::endl;
        for (auto & card : tableau.Targets()) {
            std::cout << card.ToString(true) << " ";
        }
        std::cout << std::endl;

        std::cout << "\nTableau::Sources()" << std::endl;
        for (auto & card : tableau.Sources()) {
            std::cout << card.ToString(true) << " ";
        }
        std::cout << std::endl;

        std::cout << "\nTableau::Split()" << std::endl;
        auto split_cards = tableau.Split(Card(false, kC, k8, kTableau));

        std::cout << tableau.ToString(true) << std::endl;

        for (auto & card : split_cards) {
            std::cout << card.ToString(true) << " ";
        }
        std::cout << std::endl;
    }

    void TestFoundation() {
        std::cout << "\nTestFoundation()" << std::endl;
        Foundation foundation = Foundation(kPileSpades, kS);
        std::vector<Card> cards_to_add = {
                Card(false, kS, kA, kFoundation),
                Card(false, kS, k2, kFoundation),
                Card(false, kS, k3, kFoundation),
                Card(false, kS, k4, kFoundation),
                Card(false, kS, k5, kFoundation),
        };
        foundation.Extend(cards_to_add);
        std::cout << foundation.ToString(true) << std::endl;

        std::cout << "\nFoundation::Targets()" << std::endl;
        for (auto & card : foundation.Targets()) {
            std::cout << card.ToString(true) << " ";
        }
        std::cout << std::endl;

        std::cout << "\nFoundation::Sources()" << std::endl;
        for (auto & card : foundation.Sources()) {
            std::cout << card.ToString(true) << " ";
        }
        std::cout << std::endl;

        std::cout << "\nFoundation::Split()" << std::endl;
        auto split_cards = foundation.Split(Card(false, kS, k5, kFoundation));

        std::cout << foundation.ToString(true) << std::endl;

        for (auto & card : split_cards) {
            std::cout << card.ToString(true) << " ";
        }
        std::cout << std::endl;
    }

    void TestWaste() {
        std::cout << "\nTestWaste()" << std::endl;
        Waste waste = Waste();
        std::vector<Card> cards_to_add = {
                Card(false, kS, kA, kWaste),
                Card(false, kH, kA, kWaste),
                Card(false, kH, k6, kWaste),
                Card(false, kD, k7, kWaste),
                Card(true),
                Card(true),
                Card(true),
        };
        waste.Extend(cards_to_add);
        std::cout << waste.ToString(true) << std::endl;

        std::cout << "\nWaste::Targets()" << std::endl;
        for (auto & card : waste.Targets()) {
            std::cout << card.ToString(true) << " ";
        }
        std::cout << std::endl;

        std::cout << "\nWaste::Sources()" << std::endl;
        for (auto & card : waste.Sources()) {
            std::cout << card.ToString(true) << " ";
        }
        std::cout << std::endl;

        std::cout << "\nWaste::Split()" << std::endl;
        auto split_cards = waste.Split(Card(false, kH, kA, kWaste));

        std::cout << waste.ToString(true) << std::endl;

        for (auto & card : split_cards) {
            std::cout << card.ToString(true) << " ";
        }
        std::cout << std::endl;
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
    // open_spiel::solitaire::TestTableau();
    // open_spiel::solitaire::TestFoundation();
    // open_spiel::solitaire::TestWaste();
    // open_spiel::solitaire::TestGame();
}
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "solitaire.h"

#define BLUE "\033[34m"
#define RED  "\033[31m"

namespace open_spiel::solitaire {
namespace {

    namespace testing = open_spiel::testing;

    void TestGetOppositeSuits() {
        std::cout << "\nTestGetOppositeSuits()" << std::endl;

        // Initialize variable for return value
        std::vector<SuitType> returned_suits;

        // With a valid argument
        for (const auto suit_type : SUITS) {
            returned_suits = solitaire::GetOppositeSuits(suit_type);
            std::cout << returned_suits << std::endl;
        }

        // With an invalid argument
        returned_suits = GetOppositeSuits(kNoSuit);
        std::cout << returned_suits << std::endl;
    }

    void TestDefaultCardConstructor() {
        std::cout << "\nTestDefaultCardConstructor()" << std::endl;
        Card card;
        std::cout << "Rank     = " << card.rank << std::endl;
        std::cout << "Suit     = " << card.suit << std::endl;
        std::cout << "Location = " << card.location << std::endl;
        std::cout << "Hidden   = " << card.hidden << std::endl;
        std::cout << "Index    = " << card.GetIndex() << std::endl;

    }

    void TestCardGetIndex() {
        std::cout << "\nTestCardGetIndex()" << std::endl;

        // Test empty tableau card
        Card card = Card(false, kNoSuit, kNoRank);
        std::cout << "(" << card.rank << ", " << card.suit << ") -> " << card.GetIndex() << std::endl;

        // Test empty foundation cards
        for (auto & suit : SUITS) {
            Card card = Card(false, suit, kNoRank);
            std::cout << "(" << card.rank << ", " << card.suit << ") -> " << card.GetIndex() << std::endl;
        }

        // Test ordinary cards
        for (auto & suit : SUITS) {
            for (auto & rank : RANKS) {
                Card card = Card(false, suit, rank);
                std::cout << "(" << card.rank << ", " << card.suit << ") -> " << card.GetIndex() << std::endl;
            }
        }

    }

    void TestCardToString(bool colored = true) {
        std::cout << "\nTestCardToString()" << std::endl;

        // Test hidden card
        Card hidden_card = Card(true);
        std::cout << hidden_card.ToString(colored) << std::endl;

        // Test empty tableau card
        Card card = Card(false, kNoSuit, kNoRank);
        std::cout << card.ToString(colored) << std::endl;

        // Test empty foundation cards
        for (auto & suit : SUITS) {
            Card card = Card(false, suit, kNoRank);
            std::cout << card.ToString(colored) << std::endl;
        }

        // Test ordinary cards
        for (auto & suit : SUITS) {
            for (auto & rank : RANKS) {
                Card card = Card(false, suit, rank);
                std::cout << card.ToString(colored) << std::endl;
            }
        }
    }

    void TestLegalChildren(bool colored = true) {
        std::cout << "\nTestLegalChildren()" << std::endl;

        std::vector<LocationType> locations = {kDeck, kWaste, kFoundation, kTableau, kMissing};
        std::vector<std::string> location_strs = {"kDeck", "kWaste", "kFoundation", "kTableau", "kMissing"};

        for (const auto & loc : locations) {
            std::cout << "Location = " << location_strs.at(loc) << std::endl;

            // Test hidden card
            Card hidden_card = Card(true, kHiddenSuit, kHiddenRank, loc);
            std::cout << hidden_card.ToString(colored) << std::endl;
            for (auto & child : hidden_card.LegalChildren()) {
                std::cout << "\t" << child.ToString(colored) << std::endl;
            }

            // Test empty tableau card
            Card card = Card(false, kNoSuit, kNoRank, loc);
            std::cout << card.ToString(colored) << std::endl;
            for (auto & child : card.LegalChildren()) {
                std::cout << "\t" << child.ToString(colored) << std::endl;
            }

            // Test empty foundation cards
            for (auto & suit : SUITS) {
                Card card = Card(false, suit, kNoRank, loc);
                std::cout << card.ToString(colored) << std::endl;
                for (auto & child : card.LegalChildren()) {
                    std::cout << "\t" << child.ToString(colored) << std::endl;
                }
            }

            // Test ordinary cards
            for (auto & suit : SUITS) {
                for (auto & rank : RANKS) {
                    Card card = Card(false, suit, rank, loc);
                    std::cout << card.ToString(colored) << std::endl;
                    for (auto & child : card.LegalChildren()) {
                        std::cout << "\t" << child.ToString(colored) << std::endl;
                    }
                }
            }
        }

    }

    void TestCardConstructorFromIndex(bool colored = true) {
        std::cout << "\nTestCardConstructorFromIndex()" << std::endl;

        std::vector<Card> cards_to_test;
        cards_to_test.reserve(60);

        cards_to_test.emplace_back(true);
        cards_to_test.emplace_back(false, kNoSuit, kNoRank);

        std::cout << "Hidden Card Index = " << cards_to_test.front().GetIndex() << std::endl;

        for (auto & suit : SUITS) {
            cards_to_test.emplace_back(false, suit, kNoRank);
        }
        for (auto & suit : SUITS) {
            for (auto & rank : RANKS) {
                cards_to_test.emplace_back(false, suit, rank);
            }
        }

        for (auto & card : cards_to_test) {
            auto inverse_card = Card(card.GetIndex());
            std::string card_str = card.ToString(colored);
            std::string inverse_card_str = inverse_card.ToString(colored);
            bool is_equal = (card_str == inverse_card_str);

            std::cout << card_str << " -> " << inverse_card_str << " ";
            std::cout << " EQUAL = ";
            if (is_equal) {
                std::cout << BLUE << "True, " << RESET;
            } else {
                std::cout << RED << "False, " << RESET;
            }

            std::cout << "Index of Card = " << card.GetIndex() << std::endl;
        }
    }

    void TestTableauTensor() {
        std::cout << "\nTestTableauTensor()" << std::endl;
        Pile tableau = Pile(kTableau, kPile1stTableau);
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
        std::cout << tableau.Tensor() << std::endl;
    }

    void TestFoundationTensor() {
        std::cout << "\nTestFoundationTensor()" << std::endl;
        Pile foundation = Pile(kFoundation, kPileSpades);
        std::vector<Card> cards_to_add = {
                Card(false, kS, kA, kFoundation),
                Card(false, kS, k2, kFoundation),
                Card(false, kS, k3, kFoundation),
                Card(false, kS, k4, kFoundation),
                Card(false, kS, k5, kFoundation),
        };
        foundation.Extend(cards_to_add);

        std::cout << foundation.ToString(true) << std::endl;
        std::cout << foundation.Tensor() << std::endl;
    }
    
    void TestWasteTensor() {
        std::cout << "\nTestWasteTensor()" << std::endl;
        Pile waste = Pile(kWaste, kPileWaste);
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
        std::vector<double> tensor = waste.Tensor();

        std::cout << waste.ToString(true) << std::endl;

        auto tensor_index = tensor.begin();

        while (tensor_index != tensor.end()) {
            std::cout << std::vector<double>(tensor_index, tensor_index + 53) << std::endl;
            tensor_index += 53;
        }
    }

    void BasicSolitaireTests() {
        testing::LoadGameTest("solitaire");
        testing::RandomSimTest(*LoadGame("solitaire"), 10);
    }

    /*
    void TestTableauPile() {
        std::cout << "\nTestTableauPile()" << std::endl;

        Pile tableau = Pile(kTableau);
        std::vector<Card> cards_to_add = {
                Card(true),
                Card(true),
                Card(true),
                Card(false, kH, kQ, kTableau),
                Card(false, kS, kJ, kTableau),
                Card(false, kD, kT, kTableau),
        };

        tableau.cards = cards_to_add;
        std::vector<Card> targets = tableau.Targets();
        std::vector<Card> sources = tableau.Sources();

        std::cout << "\nTableau Cards" << std::endl;
        for (auto & card : tableau.cards) {
            std::cout << card.ToString() << std::endl;
        }

        std::cout << "\nTableau Targets" << std::endl;
        for (auto & target : targets) {
            std::cout << target.ToString() << std::endl;
        }

        std::cout << "\nTableau Sources" << std::endl;
        for (auto & source : sources) {
            std::cout << source.ToString() << std::endl;
        }
    }

    void TestFoundationPile() {
        std::cout << "\nTestFoundationPile()" << std::endl;

        Pile foundation = Pile(kFoundation, kH);
        std::vector<Card> cards_to_add = {
                Card(false, kH, kNoRank, kFoundation),
                Card(false, kH, kA, kFoundation),
                Card(false, kH, k2, kFoundation),
        };

        foundation.cards = cards_to_add;
        std::vector<Card> targets = foundation.Targets();
        std::vector<Card> sources = foundation.Sources();

        std::cout << "\nFoundation Cards" << std::endl;
        for (auto &card : foundation.cards) {
            std::cout << card.ToString() << std::endl;
        }

        std::cout << "\nFoundation Targets" << std::endl;
        for (auto &target : targets) {
            std::cout << target.ToString() << std::endl;
        }

        std::cout << "\nFoundation Sources" << std::endl;
        for (auto &source : sources) {
            std::cout << source.ToString() << std::endl;
        }
    }

    void TestEmptyTableauPile() {
        std::cout << "\nTestEmptyTableauPile()" << std::endl;

        Pile tableau = Pile(kTableau);
        std::vector<Card> targets = tableau.Targets();
        std::vector<Card> sources = tableau.Sources();

        std::cout << "\nTableau Cards" << std::endl;
        for (auto & card : tableau.cards) {
            std::cout << card.ToString() << std::endl;
        }

        std::cout << "\nTableau Targets" << std::endl;
        for (auto & target : targets) {
            std::cout << target.ToString() << std::endl;
        }

        std::cout << "\nTableau Sources" << std::endl;
        for (auto & source : sources) {
            std::cout << source.ToString() << std::endl;
        }
    }

    void TestEmptyFoundationPile() {
        std::cout << "\nTestEmptyFoundationPile()" << std::endl;

        Pile foundation = Pile(kFoundation, kH);
        std::vector<Card> targets = foundation.Targets();
        std::vector<Card> sources = foundation.Sources();

        std::cout << "\nFoundation Cards" << std::endl;
        for (auto & card : foundation.cards) {
            std::cout << card.ToString() << std::endl;
        }

        std::cout << "\nFoundation Targets" << std::endl;
        for (auto & target : targets) {
            std::cout << target.ToString() << std::endl;
        }

        std::cout << "\nFoundation Sources" << std::endl;
        for (auto & source : sources) {
            std::cout << source.ToString() << std::endl;
        }
    }

    void TestHiddenTableauPile() {
        std::cout << "\nTestHiddenTableauPile()" << std::endl;

        Pile tableau = Pile(kTableau);
        std::vector<Card> cards_to_add = {
                Card(true),
                Card(true),
                Card(true),
        };

        tableau.cards = cards_to_add;
        std::vector<Card> targets = tableau.Targets();
        std::vector<Card> sources = tableau.Sources();

        std::cout << "\nTableau Cards" << std::endl;
        for (auto & card : tableau.cards) {
            std::cout << card.ToString() << std::endl;
        }

        std::cout << "\nTableau Targets" << std::endl;
        for (auto & target : targets) {
            std::cout << target.ToString() << std::endl;
        }

        std::cout << "\nTableau Sources" << std::endl;
        for (auto & source : sources) {
            std::cout << source.ToString() << std::endl;
        }
    }

    void TestWastePile() {
        std::cout << "\nTestWastePile()" << std::endl;

        Pile waste = Pile(kWaste, );
        std::vector<Card> cards_to_add = {
                Card(false, kH, kQ, kTableau),
                Card(false, kS, k9, kTableau),
                Card(false, kC, kQ, kTableau),
                Card(false, kC, k5, kTableau),
                Card(false, kD, k6, kTableau),
                Card(false, kS, k2, kTableau),
                Card(true),
                Card(true),
                Card(true),
                Card(true),
                Card(true),
                Card(true),
        };

        waste.cards = cards_to_add;
        std::vector<Card> targets = waste.Targets();
        std::vector<Card> sources = waste.Sources();

        std::cout << "\nWaste Cards" << std::endl;
        for (auto & card : waste.cards) {
            std::cout << card.ToString() << " ";
        }

        std::cout << "\nWaste Targets" << std::endl;
        for (auto & target : targets) {
            std::cout << target.ToString() << " ";
        }

        std::cout << "\nWaste Sources" << std::endl;
        for (auto & source : sources) {
            std::cout << source.ToString() << " ";
        }

        std::cout << std::endl;
    }
    */

} // namespace
} // namespace open_spiel::solitaire

int main(int argc, char** argv) {
    /*
    open_spiel::solitaire::TestGetOppositeSuits();
    open_spiel::solitaire::TestDefaultCardConstructor();
    open_spiel::solitaire::TestCardGetIndex();
    open_spiel::solitaire::TestCardToString();
    open_spiel::solitaire::TestLegalChildren();
    open_spiel::solitaire::TestCardConstructorFromIndex();
    open_spiel::solitaire::TestTableauPile();
    open_spiel::solitaire::TestFoundationPile();
    open_spiel::solitaire::TestEmptyTableauPile();
    open_spiel::solitaire::TestEmptyFoundationPile();
    open_spiel::solitaire::TestHiddenTableauPile();
    open_spiel::solitaire::TestWastePile();

    */
    // open_spiel::solitaire::TestTableauTensor();
    // open_spiel::solitaire::TestFoundationTensor();
    // open_spiel::solitaire::TestWasteTensor();
    open_spiel::solitaire::BasicSolitaireTests();
}
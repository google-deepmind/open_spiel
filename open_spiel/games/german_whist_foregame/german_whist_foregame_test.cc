#include "open_spiel/games/GermanWhistForegame_/GermanWhistForegame_.h"

#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace german_whist_foregame {
namespace {

namespace testing = open_spiel::testing;

void BasicGermanWhistForegameTests() {
  testing::LoadGameTest("GermanWhistForegame");
  testing::ChanceOutcomesTest(*LoadGame("GermanWhistForegame"));
  testing::RandomSimTest(*LoadGame("GermanWhistForegame"), 100);
  testing::RandomSimTestWithUndo(*LoadGame("GermanWhistForegame"), 1);
  for (Player players = 3; players <= 5; players++) {
    testing::RandomSimTest(
        *LoadGame("GermanWhistForegame_", {{"players", GameParameter(players)}}), 100);
  }
  auto observer = LoadGame("GermanWhistForegame")
                      ->MakeObserver(kDefaultObsType,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("GermanWhistForegame"), observer);
}

void CountStates() {
  std::shared_ptr<const Game> game = LoadGame("GermanWhistForegame");
  auto states = algorithms::GetAllStates(*game, /*depth_limit=*/-1,
                                         /*include_terminals=*/true,
                                         /*include_chance_states=*/false);
  // 6 deals * 9 betting sequences (-, p, b, pp, pb, bp, bb, pbp, pbb) = 54
  SPIEL_CHECK_EQ(states.size(), 54);
}

void PolicyTest() {
  using PolicyGenerator = std::function<TabularPolicy(const Game& game)>;
  std::vector<PolicyGenerator> policy_generators = {
      GetAlwaysPassPolicy,
      GetAlwaysBetPolicy,
  };

  std::shared_ptr<const Game> game = LoadGame("GermanWhistForegame");
  for (const auto& policy_generator : policy_generators) {
    testing::TestEveryInfostateInPolicy(policy_generator, *game);
    testing::TestPoliciesCanPlay(policy_generator, *game);
  }
}

}  // namespace
}  // namespace GermanWhistForegame_
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::GermanWhistForegame_::BasicGermanWhistForegameTests();
  open_spiel::GermanWhistForegame_::CountStates();
  open_spiel::GermanWhistForegame_::PolicyTest();
  open_spiel::testing::CheckChanceOutcomes(*open_spiel::LoadGame(
      "GermanWhistForegame", {{"players", open_spiel::GameParameter(3)}}));
  open_spiel::testing::RandomSimTest(*open_spiel::LoadGame("GermanWhistForegame"),
                                     /*num_sims=*/10);
  open_spiel::testing::ResampleInfostateTest(
      *open_spiel::LoadGame("GermanWhistForegame"),
      /*num_sims=*/10);
}

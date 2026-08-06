#include "open_spiel/algorithms/is_mcts.h"
#include <random>
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, ttable_path, "",
          "Path to the generated German Whist tablebase file. If empty, "
          "the game falls back to an all-zero tablebase.");

namespace open_spiel {
  namespace {
    constexpr const int kSeed = 9492110;
    void PlayGWhist(int human_player, std::mt19937* rng, int num_rollouts,
                    const std::string& ttable_path) {
      std::string game_string = ttable_path.empty()
      ? "german_whist_foregame"
      : absl::StrCat("german_whist_foregame(ttable_path=", ttable_path, ")");
      std::shared_ptr<const Game> game = LoadGame(game_string);
      std::random_device rd;
      int eval_seed = rd();
      int bot_seed = rd();
      auto evaluator = std::make_shared<algorithms::RandomRolloutEvaluator>(1, eval_seed);
      auto bot = std::make_unique<algorithms::ISMCTSBot>(
        bot_seed, evaluator, 0.7*13, num_rollouts, algorithms::kUnlimitedNumWorldSamples,
        algorithms::ISMCTSFinalPolicyType::kMaxVisitCount,true, false);
      std::unique_ptr<State> state = game->NewInitialState();
      while (!state->IsTerminal()) {
        Action chosen_action = kInvalidAction;
        if (state->IsChanceNode()) {
          chosen_action =
          SampleAction(state->ChanceOutcomes(), absl::Uniform(*rng, 0.0, 1.0))
          .first;
        } else if(state->CurrentPlayer()!=human_player) {
          chosen_action = bot->Step(*state);
        }
        else{
          std::cout<<state->InformationStateString(human_player)<<std::endl;
          auto legal_actions = state->LegalActions();
          for(int i =0;i<legal_actions.size();++i){
            std::cout<<state->ActionToString(legal_actions[i])<<",";
          }
          std::cout<<std::endl;
          std::cout<<"Input action:";
          std::string input;
          std::cin>>input;
          chosen_action = state->StringToAction(input);
          std::cout<<std::endl;
        }
        state->ApplyAction(chosen_action);
      }
      std::cout << "Terminal state:" << std::endl;
      std::cout << state->ToString() << std::endl;
      std::cout << "Returns: " << absl::StrJoin(state->Returns(), " ") << std::endl;
                    }
  }  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  std::random_device rd;
  std::mt19937 rng(rd());
  int human_player;
  int num_rollouts;
  std::cout<<"human_player:";
  std::cin>>human_player;
  std::cout<<"\n";
  std::cout<<"num_rollouts:";
  std::cin>>num_rollouts;
  std::cout<<"\n";
  open_spiel::PlayGWhist(human_player, &rng, num_rollouts,
                         absl::GetFlag(FLAGS_ttable_path));
}

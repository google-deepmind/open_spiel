// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unistd.h>

#include <memory>
#include <random>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/examples/sbr_blotto/fictitious_play.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/init.h"

ABSL_FLAG(int, players, 2, "Number of players.");
ABSL_FLAG(int, coins, 10, "Number of coins to place.");
ABSL_FLAG(int, fields, 3, "Number of coins to place.");

ABSL_FLAG(int, seed, 82368234, "Seed for the random number generator.");
ABSL_FLAG(int, iterations, -1, "Number of iterations.");
ABSL_FLAG(std::string, algorithm, "fp",
          "Algorithm to run (fp|sbr|ibr|meibr|sfp|brpi)");
ABSL_FLAG(int, sbr_b, 10, "Number of base profiles in SBR");
ABSL_FLAG(int, sbr_c, 25, "Number of candidates in SBR");
ABSL_FLAG(int, brpi_n, 1000, "N in BRPI");
ABSL_FLAG(std::string, base_sampler, "uniform", "Base sampler type.");
ABSL_FLAG(std::string, candidates_sampler, "initial", "Cand. sampler type.");
ABSL_FLAG(double, lambda, 1.0, "Lambda for the softmax");
ABSL_FLAG(std::string, logdirpref, "/tmp", "Log prefix");
ABSL_FLAG(std::string, run_name, "run", "Run name");
ABSL_FLAG(bool, enable_log, true, "Whether to enable logging?");
ABSL_FLAG(std::string, game, "", "Game string override (if not blotto)");
ABSL_FLAG(bool, randomize_initial_policies, false,
          "Arbitrary initial policies?");

using open_spiel::algorithms::blotto_fp::BaseSamplerType;
using open_spiel::algorithms::blotto_fp::CandidatesSamplerType;
using open_spiel::algorithms::blotto_fp::FictitiousPlayProcess;

BaseSamplerType GetBaseSamplerType(const std::string& str) {
  if (str == "uniform") {
    return BaseSamplerType::kBaseUniform;
  } else if (str == "latest") {
    return BaseSamplerType::kBaseLatest;
  } else {
    open_spiel::SpielFatalError("Unrecognized base sampler type.");
  }
}

CandidatesSamplerType GetCandidatesSamplerType(const std::string& str) {
  if (str == "initial") {
    return CandidatesSamplerType::kCandidatesInitial;
  } else if (str == "uniform") {
    return CandidatesSamplerType::kCandidatesUniform;
  } else if (str == "latest") {
    return CandidatesSamplerType::kCandidatesLatest;
  } else if (str == "mixedIU") {
    return CandidatesSamplerType::kCandidatesInitialUniform;
  } else if (str == "mixedIL") {
    return CandidatesSamplerType::kCandidatesInitialLatest;
  } else {
    open_spiel::SpielFatalError("Unrecognized candidates sampler type.");
  }
}

int main(int argc, char** argv) {
  open_spiel::Init(argv[0], &argc, &argv, /*remove_flags=*/true);

  int players = absl::GetFlag(FLAGS_players);
  int coins = absl::GetFlag(FLAGS_coins);
  int fields = absl::GetFlag(FLAGS_fields);

  absl::ParseCommandLine(argc, argv);

  std::string game_string = "";
  if (absl::GetFlag(FLAGS_game).empty()) {
    game_string = absl::StrCat(
        "turn_based_simultaneous_game(game=blotto(players=", players,
        ",coins=", coins, ",fields=", fields, "))");
  } else {
    game_string = absl::StrCat(
        "turn_based_simultaneous_game(game=", absl::GetFlag(FLAGS_game), ")");
  }

  std::cout << "game string: " << game_string << std::endl;
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(game_string);
  SPIEL_CHECK_TRUE(game != nullptr);

  std::cout << "NumDistinctActions: " << game->NumDistinctActions()
            << std::endl;
  std::cout << "Size of game: " << std::fixed
            << std::pow(game->NumDistinctActions(), game->NumPlayers())
            << std::endl;

  FictitiousPlayProcess fpp(game, absl::GetFlag(FLAGS_seed),
                            absl::GetFlag(FLAGS_randomize_initial_policies));
  int iterations = absl::GetFlag(FLAGS_iterations);
  std::string algo = absl::GetFlag(FLAGS_algorithm);
  int sbr_num_base = absl::GetFlag(FLAGS_sbr_b);
  int sbr_num_candidates = absl::GetFlag(FLAGS_sbr_c);
  int brpi_n = absl::GetFlag(FLAGS_brpi_n);
  double lambda = absl::GetFlag(FLAGS_lambda);
  BaseSamplerType base_sampler_type =
      GetBaseSamplerType(absl::GetFlag(FLAGS_base_sampler));
  CandidatesSamplerType candidates_sampler_type =
      GetCandidatesSamplerType(absl::GetFlag(FLAGS_candidates_sampler));

  std::unique_ptr<open_spiel::file::File> logfile = nullptr;
  if (absl::GetFlag(FLAGS_enable_log)) {
    std::cout << "Opening log file.." << std::endl;

    std::string dir = absl::StrCat(absl::GetFlag(FLAGS_logdirpref), "/",
                                   absl::GetFlag(FLAGS_run_name));

    std::string filename = absl::StrCat(dir, "/blotto_", players, "_", coins,
                                        "_", fields, "_", algo);

    if (absl::GetFlag(FLAGS_randomize_initial_policies)) {
      absl::StrAppend(&filename, "_rip");
    }

    absl::StrAppend(&filename, "_seed", absl::GetFlag(FLAGS_seed));

    if (algo == "sbr") {
      absl::StrAppend(&filename, "_", sbr_num_base, "_", sbr_num_candidates);
    } else if (algo == "sfp") {
      absl::StrAppend(&filename, "_lambda", lambda);
    } else if (algo == "brpi") {
      absl::StrAppend(&filename, "_", sbr_num_base, "_", sbr_num_candidates,
                      "_", absl::GetFlag(FLAGS_base_sampler), "_",
                      absl::GetFlag(FLAGS_candidates_sampler));
    }

    if (!open_spiel::file::Exists(dir)) {
      std::cout << "Creating log directory " << dir << std::endl;
      SPIEL_CHECK_TRUE(open_spiel::file::Mkdir(dir));
    }

    logfile = std::make_unique<open_spiel::file::File>(filename, "w");
  }

  std::cout << "Starting." << std::endl;

  int next_br_iter = 1;
  for (int i = 1; i < iterations || iterations < 0; ++i) {
    if (algo == "fp") {
      fpp.FullFPIteration();
    } else if (algo == "sbr") {
      fpp.SBRIteration(sbr_num_base, sbr_num_candidates);
    } else if (algo == "ibr") {
      fpp.IBRIteration();
    } else if (algo == "meibr") {
      fpp.MaxEntIBRIteration();
    } else if (algo == "sfp") {
      fpp.SFPIteration(lambda);
    } else if (algo == "brpi") {
      fpp.BRPIIteration(base_sampler_type, candidates_sampler_type,
                        sbr_num_base, sbr_num_candidates, brpi_n);
    } else {
      std::cerr << "Unrecognized algorithm. Exiting...";
      exit(-1);
    }

    if (i == next_br_iter) {
      double cce_dist = fpp.CCEDist();
      double nash_conv = fpp.NashConv();

      absl::Duration total_time = fpp.TotalTime();

      std::string outline =
          absl::StrCat(i, " ", absl::ToDoubleSeconds(total_time), " ", cce_dist,
                       " ", nash_conv, "\n");
      std::cout << outline;
      if (logfile != nullptr) {
        SPIEL_CHECK_TRUE(logfile->Write(outline));
        SPIEL_CHECK_TRUE(logfile->Flush());
      }

      next_br_iter *= 2;
    }
  }
}

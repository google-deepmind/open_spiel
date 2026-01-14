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

#include "open_spiel/algorithms/nfg_writer.h"

#include "open_spiel/normal_form_game.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace {

void BasicNFSWriterTestRPS() {
  constexpr const char* kRPSNFG =
      R"###(NFG 1 R "OpenSpiel export of matrix_rps()"
{ "Player 0" "Player 1" } { 3 3 }

0 0
1 -1
-1 1
-1 1
0 0
1 -1
1 -1
-1 1
0 0
)###";

  std::shared_ptr<const Game> rps = LoadGame("matrix_rps");
  std::string rps_nfg_text = GameToNFGString(*rps);
  SPIEL_CHECK_EQ(rps_nfg_text, kRPSNFG);
}

void BasicNFSWriterTestPD() {
  constexpr const char* kPDNFG = R"###(NFG 1 R "OpenSpiel export of matrix_pd()"
{ "Player 0" "Player 1" } { 2 2 }

5 5
10 0
0 10
1 1
)###";

  std::shared_ptr<const Game> pd = LoadGame("matrix_pd");
  std::string pd_nfg_text = GameToNFGString(*pd);
  SPIEL_CHECK_EQ(pd_nfg_text, kPDNFG);
}

void BasicNFSWriterTestMP3P() {
  constexpr const char* kMP3PNFG =
      R"###(NFG 1 R "OpenSpiel export of matching_pennies_3p()"
{ "Player 0" "Player 1" "Player 2" } { 2 2 2 }

1 1 -1
-1 1 1
-1 -1 -1
1 -1 1
1 -1 1
-1 -1 -1
-1 1 1
1 1 -1
)###";

  std::shared_ptr<const Game> mp3p = LoadGame("matching_pennies_3p");
  std::string mp3p_nfg_text = GameToNFGString(*mp3p);
  SPIEL_CHECK_EQ(mp3p_nfg_text, kMP3PNFG);
}
}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::BasicNFSWriterTestRPS();
  open_spiel::BasicNFSWriterTestPD();
  open_spiel::BasicNFSWriterTestMP3P();
}

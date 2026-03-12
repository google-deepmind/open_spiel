// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/utils/sgf_reader.h"

#include <vector>
#include <string>

#include "open_spiel/utils/status.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace go {
namespace {

void BasicSGFReaderTests() {
  StatusWithValue<std::vector<SgfNode>> status_with_nodes;

  status_with_nodes = ReadSgfString(kExampleGoSgfString);
  SPIEL_CHECK_OK(status_with_nodes);
  std::vector<SgfNode> nodes = status_with_nodes.value();
  SPIEL_CHECK_EQ(nodes.size(), 1);
  SgfNode node = nodes[0];
  SPIEL_CHECK_EQ(node.properties.size(), 9);
  SPIEL_CHECK_EQ(node.children.size(), 1);
  SPIEL_CHECK_EQ(node.children[0].children.size(), 1);
  SPIEL_CHECK_EQ(node.children[0].properties[0].name, "B");
  SPIEL_CHECK_EQ(node.children[0].properties[0].values[0], "pd");
  SPIEL_CHECK_EQ(node.children[0].children[0].children.size(), 1);

  status_with_nodes = ReadSgfString(kExampleGoSgfString2);
  nodes = status_with_nodes.value();
  SPIEL_CHECK_EQ(nodes.size(), 1);
  node = nodes[0];
  SPIEL_CHECK_EQ(node.properties.size(), 6);
  SPIEL_CHECK_EQ(node.children.size(), 2);
  // ;B[de];W[ee]
  SPIEL_CHECK_EQ(node.children[0].properties.size(), 1);
  SPIEL_CHECK_EQ(node.children[0].properties[0].name, "B");
  SPIEL_CHECK_EQ(node.children[0].properties[0].values[0], "de");
  SPIEL_CHECK_EQ(node.children[0].children.size(), 1);
  SPIEL_CHECK_EQ(node.children[0].children[0].properties[0].name, "W");
  SPIEL_CHECK_EQ(node.children[0].children[0].properties[0].values[0], "ee");

  status_with_nodes = ReadSgfString(kExampleGoSgfString3);
  nodes = status_with_nodes.value();
  SPIEL_CHECK_EQ(nodes.size(), 1);
  node = nodes[0];
  SPIEL_CHECK_EQ(node.properties.size(), 8);
  SPIEL_CHECK_EQ(node.children.size(), 0);
  // AB[ik][il][jd][jk][kj][kl][lj]AW[dd][dj][jj])
  SPIEL_CHECK_EQ(node.properties[6].name, "AB");
  SPIEL_CHECK_EQ(node.properties[6].values.size(), 7);
  SPIEL_CHECK_EQ(node.properties[7].name, "AW");
  SPIEL_CHECK_EQ(node.properties[7].values.size(), 3);
}


}  // namespace
}  // namespace go
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::go::BasicSGFReaderTests();
}

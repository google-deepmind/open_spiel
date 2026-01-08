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

#include "open_spiel/evaluation/soft_condorcet_optimization.h"

#include <iostream>  // NOLINT (used by std::cout)
#include <map>
#include <string>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestSimpleCaseSigmoid() {
  std::cout << "TestSimpleCaseSigmoid" << std::endl;
  evaluation::SoftCondorcetOptimizer sco_optimizer({{1, {"a", "b", "c"}}},
                                                   -100.0, 100.0, 4, 1.0);
  sco_optimizer.RunSolver(1000, 0.01);
  std::map<std::string, double> ratings = sco_optimizer.ratings();
  for (const auto& [alt, rating] : ratings) {
    std::cout << alt << ": " << rating << std::endl;
  }
  SPIEL_CHECK_GT(ratings["a"], ratings["b"]);
  SPIEL_CHECK_GT(ratings["b"], ratings["c"]);
}

void TestMeeplePentathlonSigmoid() {
  std::cout << "TestMeeplePentathlonSigmoid" << std::endl;
  evaluation::SoftCondorcetOptimizer sco_optimizer({{1, {"a", "b", "c"}},
                                                    {1, {"a", "c", "b"}},
                                                    {2, {"c", "a", "b"}},
                                                    {1, {"b", "c", "a"}}},
                                                   -100.0, 100.0, 4, 1.0);
  sco_optimizer.RunSolver(1000, 0.01);
  std::map<std::string, double> ratings = sco_optimizer.ratings();
  for (const auto& [alt, rating] : ratings) {
    std::cout << alt << ": " << rating << std::endl;
  }
  SPIEL_CHECK_GT(ratings["c"], ratings["a"]);
  SPIEL_CHECK_GT(ratings["a"], ratings["b"]);
}

void TestSec41ExampleSigmoid() {
  std::cout << "TestSec41ExampleSigmoid" << std::endl;
  evaluation::SoftCondorcetOptimizer sco_optimizer(
      {{2, {"a", "b", "c"}}, {3, {"c", "a", "b"}}}, -100.0, 100.0, 4, 1.0);
  sco_optimizer.RunSolver(10000, 0.01);
  std::map<std::string, double> ratings = sco_optimizer.ratings();
  for (const auto& [alt, rating] : ratings) {
    std::cout << alt << ": " << rating << std::endl;
  }
  SPIEL_CHECK_GT(ratings["c"], ratings["a"]);
  SPIEL_CHECK_GT(ratings["a"], ratings["b"]);
}

void TestSimpleCaseFenchelYoung() {
  std::cout << "TestSimpleCaseFenchelYoung" << std::endl;
  evaluation::FenchelYoungOptimizer fy_optimizer({{1, {"a", "b", "c"}}}, -100.0,
                                                 100.0, 4, 1.0);
  fy_optimizer.RunSolver(1000, 0.01);
  std::map<std::string, double> ratings = fy_optimizer.ratings();
  for (const auto& [alt, rating] : ratings) {
    std::cout << alt << ": " << rating << std::endl;
  }
  SPIEL_CHECK_GT(ratings["a"], ratings["b"]);
  SPIEL_CHECK_GT(ratings["b"], ratings["c"]);
}

void TestMeeplePentathlonFenchelYoung() {
  std::cout << "TestMeeplePentathlonFenchelYoung" << std::endl;
  evaluation::FenchelYoungOptimizer fy_optimizer({{1, {"a", "b", "c"}},
                                                  {1, {"a", "c", "b"}},
                                                  {2, {"c", "a", "b"}},
                                                  {1, {"b", "c", "a"}}},
                                                 -100.0, 100.0, 4, 1.0);
  fy_optimizer.RunSolver(1000, 0.01);
  std::map<std::string, double> ratings = fy_optimizer.ratings();
  for (const auto& [alt, rating] : ratings) {
    std::cout << alt << ": " << rating << std::endl;
  }
  // Not necessarily C > A > B! C ~= A just like with Elo.
  SPIEL_CHECK_GT(ratings["c"], ratings["b"]);
  SPIEL_CHECK_GT(ratings["a"], ratings["b"]);
}

void TestSec41ExampleFenchelYoung() {
  std::cout << "TestSec41ExampleFenchelYoung" << std::endl;
  evaluation::FenchelYoungOptimizer fy_optimizer(
      {{2, {"a", "b", "c"}}, {3, {"c", "a", "b"}}}, -100.0, 100.0, 4, 1.0);
  fy_optimizer.RunSolver(10000, 0.01);
  std::map<std::string, double> ratings = fy_optimizer.ratings();
  for (const auto& [alt, rating] : ratings) {
    std::cout << alt << ": " << rating << std::endl;
  }
  // Like Elo, this should result in A > C > B.
  SPIEL_CHECK_GT(ratings["a"], ratings["c"]);
  SPIEL_CHECK_GT(ratings["c"], ratings["b"]);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestSimpleCaseSigmoid();
  open_spiel::TestMeeplePentathlonSigmoid();
  open_spiel::TestSec41ExampleSigmoid();
  open_spiel::TestSimpleCaseFenchelYoung();
  open_spiel::TestMeeplePentathlonFenchelYoung();
  open_spiel::TestSec41ExampleFenchelYoung();
}

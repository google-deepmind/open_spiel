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

#include "open_spiel/utils/run_python.h"

#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, python_command, open_spiel::kDefaultPythonCommand,
          "The python interpreter command to use.");

namespace open_spiel {
namespace {

inline constexpr const char* kTestModuleName =
    "open_spiel.utils.run_python_test_file";

void TestRunPython() {
  std::string test_module(kTestModuleName);
  std::string python_command = absl::GetFlag(FLAGS_python_command);
  SPIEL_CHECK_TRUE(RunPython(python_command, test_module,
                             {"--return_value", "0"}));
  SPIEL_CHECK_FALSE(RunPython(python_command, test_module,
                              {"--return_value", "1"}));

  SPIEL_CHECK_TRUE(RunPython(python_command, test_module,
                             {"--print_value", "asdf"}));
  SPIEL_CHECK_FALSE(RunPython(python_command, test_module,
                              {"--bogus_flag"}));
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  open_spiel::TestRunPython();
}

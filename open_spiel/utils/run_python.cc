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

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"

namespace open_spiel {

bool RunPython(const std::string& python_command,
               const std::string& module,
               const std::vector<std::string>& args) {
  // If this fails, make sure your PYTHONPATH environment variable is correct.
  return 0 == std::system(absl::StrCat(
      python_command, " -m ", module, " ", absl::StrJoin(args, " ")).c_str());
}


bool RunPython(const std::string& module,
               const std::vector<std::string>& args) {
  return RunPython(kDefaultPythonCommand, module, args);
}

}  // namespace open_spiel


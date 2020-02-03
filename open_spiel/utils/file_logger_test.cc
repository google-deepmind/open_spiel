// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/utils/file_logger.h"

#include <cstdlib>
#include <string>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"

namespace open_spiel::file {
namespace {

std::string GetEnv(const std::string& key, const std::string& default_value) {
    char* val = getenv(key.c_str());
    return ((val != nullptr) ? std::string(val) : default_value);
}

void TestFileLogger() {
  std::string val = std::to_string(std::rand());  // NOLINT
  std::string tmp_dir = GetEnv("TMPDIR", "/tmp");
  std::string dir = tmp_dir + "/open_spiel-test-" + val;
  std::string filename = dir + "/log-test.txt";

  SPIEL_CHECK_TRUE(Exists(tmp_dir));
  SPIEL_CHECK_TRUE(IsDirectory(tmp_dir));
  SPIEL_CHECK_FALSE(Exists(dir));
  SPIEL_CHECK_TRUE(Mkdir(dir));
  SPIEL_CHECK_FALSE(Mkdir(dir));  // already exists
  SPIEL_CHECK_TRUE(Exists(dir));
  SPIEL_CHECK_TRUE(IsDirectory(dir));

  {
    FileLogger logger(dir, "test");
    logger.Print("line 1");
    logger.Print("line %d", 2);
    logger.Print("line %d: %s", 3, "asdf");
  }

  std::string prefix = "hello world ";
  std::string expected = prefix + val + "\n";
  {
    File f(filename, "r");
    std::vector<std::string> lines = absl::StrSplit(f.ReadContents(), '\n');
    SPIEL_CHECK_EQ(lines.size(), 6);
    SPIEL_CHECK_TRUE(absl::StrContains(lines[0], "test started"));
    SPIEL_CHECK_TRUE(absl::StrContains(lines[1], "line 1"));
    SPIEL_CHECK_TRUE(absl::StrContains(lines[2], "line 2"));
    SPIEL_CHECK_TRUE(absl::StrContains(lines[3], "line 3: asdf"));
    SPIEL_CHECK_TRUE(absl::StrContains(lines[4], "Closing the log"));
    SPIEL_CHECK_EQ(lines[5], "");
  }

  SPIEL_CHECK_TRUE(Remove(filename));
  SPIEL_CHECK_TRUE(Remove(dir));
}

}  // namespace
}  // namespace open_spiel::file

int main(int argc, char** argv) {
  open_spiel::file::TestFileLogger();
}

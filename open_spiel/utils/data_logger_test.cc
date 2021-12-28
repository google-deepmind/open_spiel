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

#include "open_spiel/utils/data_logger.h"

#include <cstdlib>
#include <string>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"

namespace open_spiel {
namespace {

void TestDataLogger() {
  std::string val = std::to_string(std::rand());  // NOLINT
  std::string tmp_dir = file::GetTmpDir();
  std::string dir = tmp_dir + "/open_spiel-test-" + val;
  std::string filename = dir + "/data-test.jsonl";

  SPIEL_CHECK_TRUE(file::Exists(tmp_dir));
  SPIEL_CHECK_TRUE(file::IsDirectory(tmp_dir));
  SPIEL_CHECK_FALSE(file::Exists(dir));
  SPIEL_CHECK_TRUE(file::Mkdir(dir));
  SPIEL_CHECK_TRUE(file::Exists(dir));
  SPIEL_CHECK_TRUE(file::IsDirectory(dir));

  {
    DataLoggerJsonLines logger(dir, "data-test");
    logger.Write({{"step", 1}, {"avg", 1.5}});
    logger.Write({{"step", 2}, {"avg", 2.5}});
  }

  {
    file::File f(filename, "r");
    std::vector<std::string> lines = absl::StrSplit(f.ReadContents(), '\n');
    SPIEL_CHECK_EQ(lines.size(), 3);
    SPIEL_CHECK_EQ(lines[2], "");

    json::Object obj1 = json::FromString(lines[0])->GetObject();
    SPIEL_CHECK_EQ(obj1["step"], 1);
    SPIEL_CHECK_EQ(obj1["avg"], 1.5);
    SPIEL_CHECK_TRUE(obj1["time_str"].IsString());
    SPIEL_CHECK_TRUE(obj1["time_abs"].IsDouble());
    SPIEL_CHECK_GT(obj1["time_abs"].GetDouble(), 1'500'000'000);  // July 2017
    SPIEL_CHECK_TRUE(obj1["time_rel"].IsDouble());
    SPIEL_CHECK_GT(obj1["time_rel"].GetDouble(), 0);

    json::Object obj2 = json::FromString(lines[1])->GetObject();
    SPIEL_CHECK_EQ(obj2["step"], 2);
    SPIEL_CHECK_EQ(obj2["avg"], 2.5);

    SPIEL_CHECK_LT(obj1["time_abs"].GetDouble(), obj2["time_abs"].GetDouble());
    SPIEL_CHECK_LT(obj1["time_rel"].GetDouble(), obj2["time_rel"].GetDouble());
  }

  SPIEL_CHECK_TRUE(file::Remove(filename));
  SPIEL_CHECK_TRUE(file::Remove(dir));
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestDataLogger();
}

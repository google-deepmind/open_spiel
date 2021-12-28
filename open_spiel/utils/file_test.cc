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

#include "open_spiel/utils/file.h"

#include <cstdlib>
#include <string>

#include "open_spiel/spiel_utils.h"

namespace open_spiel::file {
namespace {

void TestFile() {
  std::string val = std::to_string(std::rand());  // NOLINT
  std::string tmp_dir = file::GetTmpDir();
  std::string dir = tmp_dir + "/open_spiel-test-" + val;
  std::string filename = dir + "/test-file.txt";

  SPIEL_CHECK_TRUE(Exists(tmp_dir));
  SPIEL_CHECK_TRUE(IsDirectory(tmp_dir));

  SPIEL_CHECK_FALSE(Exists(dir));
  SPIEL_CHECK_TRUE(Mkdir(dir));
  SPIEL_CHECK_FALSE(Mkdir(dir));  // already exists
  SPIEL_CHECK_TRUE(Exists(dir));
  SPIEL_CHECK_TRUE(IsDirectory(dir));

  std::string prefix = "hello world ";
  std::string expected = prefix + val + "\n";
  {
    File f(filename, "w");
    SPIEL_CHECK_EQ(f.Tell(), 0);
    SPIEL_CHECK_TRUE(f.Write(expected));
    SPIEL_CHECK_TRUE(f.Flush());
    SPIEL_CHECK_EQ(f.Tell(), expected.size());
    SPIEL_CHECK_EQ(f.Length(), expected.size());
  }

  SPIEL_CHECK_TRUE(Exists(filename));
  SPIEL_CHECK_FALSE(IsDirectory(filename));

  {
    File f(filename, "r");
    SPIEL_CHECK_EQ(f.Tell(), 0);
    SPIEL_CHECK_EQ(f.Length(), expected.size());
    std::string found = f.ReadContents();
    SPIEL_CHECK_EQ(found, expected);
    SPIEL_CHECK_EQ(f.Tell(), expected.size());
    f.Seek(0);
    SPIEL_CHECK_EQ(f.Read(6), "hello ");
    SPIEL_CHECK_EQ(f.Read(6), "world ");
  }

  { // Test the move constructor/assignment.
    File f(filename, "r");
    File f2 = std::move(f);
    File f3(std::move(f2));
  }

  SPIEL_CHECK_TRUE(Remove(filename));
  SPIEL_CHECK_FALSE(Remove(filename));  // already gone
  SPIEL_CHECK_FALSE(Exists(filename));

  std::string deep_dir = dir + "/1/2/3";
  SPIEL_CHECK_FALSE(IsDirectory(dir + "/1"));
  SPIEL_CHECK_TRUE(Mkdirs(dir + "/1/2/3"));
  SPIEL_CHECK_TRUE(IsDirectory(dir + "/1/2/3"));
  SPIEL_CHECK_TRUE(Remove(dir + "/1/2/3"));
  SPIEL_CHECK_TRUE(Remove(dir + "/1/2"));
  SPIEL_CHECK_TRUE(Remove(dir + "/1"));

  SPIEL_CHECK_TRUE(Remove(dir));
  SPIEL_CHECK_FALSE(Exists(dir));
}

}  // namespace
}  // namespace open_spiel::file

int main(int argc, char** argv) {
  open_spiel::file::TestFile();
}

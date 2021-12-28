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

#ifndef OPEN_SPIEL_UTILS_FILE_H_
#define OPEN_SPIEL_UTILS_FILE_H_

#include <string>
#include <memory>

#include "open_spiel/abseil-cpp/absl/strings/string_view.h"

namespace open_spiel::file {

// A simple file abstraction. Needed for compatibility with Google's libraries.
class File {
 public:
  File(const std::string& filename, const std::string& mode);

  // File is move only.
  File(File&& other);
  File& operator=(File&& other);
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  ~File();  // Flush and Close.

  bool Flush();  // Flush the buffer to disk.

  std::int64_t Tell();  // Offset of the current point in the file.
  bool Seek(std::int64_t offset);  // Move the current point.

  std::string Read(std::int64_t count);  // Read count bytes.
  std::string ReadContents();  // Read the entire file.

  bool Write(absl::string_view str);  // Write to the file.

  std::int64_t Length();  // Length of the entire file.

 private:
  bool Close();  // Close the file. Use the destructor instead.

  class FileImpl;
  std::unique_ptr<FileImpl> fd_;
};

// Reads the file at filename to a string. Dies if this doesn't succeed.
std::string ReadContentsFromFile(const std::string& filename,
                                 const std::string& mode);

bool Exists(const std::string& path);  // Does the file/directory exist?
bool IsDirectory(const std::string& path);  // Is it a directory?
bool Mkdir(const std::string& path, int mode = 0755);  // Make a directory.
bool Mkdirs(const std::string& path, int mode = 0755);  // Mkdir recursively.
bool Remove(const std::string& path);  // Remove/delete the file/directory.

std::string GetEnv(const std::string& key, const std::string& default_value);
std::string GetTmpDir();

}  // namespace open_spiel::file

#endif  // OPEN_SPIEL_UTILS_FILE_H_

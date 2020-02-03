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

#include "open_spiel/utils/file.h"

#include <filesystem>
#include <cstdio>

#include "open_spiel/spiel_utils.h"

namespace open_spiel::file {

class File::FileImpl : public std::FILE {};

File::File(const std::string& filename, const std::string& mode) {
  fd_.reset(static_cast<FileImpl*>(std::fopen(filename.c_str(), mode.c_str())));
  SPIEL_CHECK_TRUE(fd_);
}

File::~File() {
  if (fd_) {
    Flush();
    Close();
  }
}

File::File(File&& other) = default;
File& File::operator=(File&& other) = default;

bool File::Close() { return !std::fclose(fd_.release()); }
bool File::Flush() { return !std::fflush(fd_.get()); }
std::int64_t File::Tell() { return std::ftell(fd_.get()); }
bool File::Seek(std::int64_t offset) {
  return !std::fseek(fd_.get(), offset, SEEK_SET);
}

std::string File::Read(std::int64_t count) {
  std::string out(count, '\0');
  int read = std::fread(out.data(), sizeof(char), count, fd_.get());
  out.resize(read);
  return out;
}

std::string File::ReadContents() {
  Seek(0);
  return Read(Length());
}

bool File::Write(absl::string_view str) {
  return std::fwrite(str.data(), sizeof(char), str.size(), fd_.get()) ==
         str.size();
}

std::int64_t File::Length() {
  std::int64_t current = std::ftell(fd_.get());
  std::fseek(fd_.get(), 0, SEEK_END);
  std::int64_t length = std::ftell(fd_.get());
  std::fseek(fd_.get(), current, SEEK_SET);
  return length;
}

bool Exists(const std::string& path) {
  return std::filesystem::exists(path);
}

bool IsDirectory(const std::string& path) {
  return std::filesystem::is_directory(path);
}

bool Mkdir(const std::string& path) {
  return std::filesystem::create_directory(path);
}

bool Mkdirs(const std::string& path) {
  return std::filesystem::create_directories(path);
}

bool Remove(const std::string& path) {
  return std::filesystem::remove(path);
}

}  // namespace open_spiel::file

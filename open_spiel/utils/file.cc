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

#include <sys/types.h>
#include <sys/stat.h>

#ifdef _WIN32
// https://stackoverflow.com/a/42906151
#include <windows.h>
#include <direct.h>
#include <stdio.h>
#define mkdir(dir, mode) _mkdir(dir)
#define unlink(file) _unlink(file)
#define rmdir(dir) _rmdir(dir)
#else
#include <unistd.h>
#endif

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

std::string ReadContentsFromFile(const std::string& filename,
                                 const std::string& mode) {
  File f(filename, mode);
  return f.ReadContents();
}

bool Exists(const std::string& path) {
  struct stat info;
  return stat(path.c_str(), &info) == 0;
}

bool IsDirectory(const std::string& path) {
  struct stat info;
  return stat(path.c_str(), &info) == 0 && info.st_mode & S_IFDIR;
}

bool Mkdir(const std::string& path, int mode) {
  return mkdir(path.c_str(), mode) == 0;
}

bool Mkdirs(const std::string& path, int mode) {
  struct stat info;
  size_t pos = 0;
  while (pos != std::string::npos) {
    pos = path.find_first_of("\\/", pos + 1);
    std::string sub_path = path.substr(0, pos);
    if (stat(sub_path.c_str(), &info) == 0) {
      if (info.st_mode & S_IFDIR) {
        continue;  // directory already exists
      } else {
        return false;  // is a file?
      }
    }
    if (!Mkdir(sub_path, mode)) {
      return false;  // permission error?
    }
  }
  return true;
}

bool Remove(const std::string& path) {
  if (IsDirectory(path)) {
    return rmdir(path.c_str()) == 0;
  } else {
    return unlink(path.c_str()) == 0;
  }
}

std::string GetEnv(const std::string& key, const std::string& default_value) {
  char* val = std::getenv(key.c_str());
  return ((val != nullptr) ? std::string(val) : default_value);
}

std::string GetTmpDir() { return GetEnv("TMPDIR", "/tmp"); }

}  // namespace open_spiel::file

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

#include "open_spiel/games/ludii/chunk_set.h"

namespace open_spiel {
namespace ludii {

ChunkSet::ChunkSet(JNIEnv *env, jobject chunkset)
    : env(env), chunkset(chunkset) {}

std::string ChunkSet::Print() const {
  jclass chunkSetClass = env->FindClass("util/ChunkSet");
  jmethodID tostring_id =
      env->GetMethodID(chunkSetClass, "toString", "()Ljava/lang/String;");
  jstring string_obj = (jstring)env->CallObjectMethod(chunkset, tostring_id);

  const char *rawString = env->GetStringUTFChars(string_obj, 0);
  std::string cppString(rawString);
  env->ReleaseStringUTFChars(string_obj, rawString);

  return cppString;
}

std::string ChunkSet::ToChunkString() const {
  jclass chunkSetClass = env->FindClass("util/ChunkSet");
  jmethodID toChunkString_id =
      env->GetMethodID(chunkSetClass, "toChunkString", "()Ljava/lang/String;");
  jstring string_obj =
      (jstring)env->CallObjectMethod(chunkset, toChunkString_id);

  const char *rawString = env->GetStringUTFChars(string_obj, 0);
  std::string cppString(rawString);
  env->ReleaseStringUTFChars(string_obj, rawString);

  return cppString;
}

}  // namespace ludii
}  // namespace open_spiel

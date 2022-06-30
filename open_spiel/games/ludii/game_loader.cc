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

#include "open_spiel/games/ludii/game_loader.h"

#include <cstring>
#include <string>

namespace open_spiel {
namespace ludii {

GameLoader::GameLoader(JNIEnv *env) : env(env) {}

std::vector<std::string> GameLoader::ListGames() const {
  std::vector<std::string> gamesVector;

  jclass gameLoader = env->FindClass("player/GameLoader");
  jmethodID mid =
      env->GetStaticMethodID(gameLoader, "listGames", "()[Ljava/lang/String;");
  jobjectArray stringArray =
      (jobjectArray)env->CallStaticObjectMethod(gameLoader, mid);

  int stringCount = env->GetArrayLength(stringArray);

  for (int i = 0; i < stringCount; i++) {
    // get array element and convert it from jstring
    jstring string = (jstring)(env->GetObjectArrayElement(stringArray, i));
    const char *rawString = env->GetStringUTFChars(string, 0);

    std::string cppString(rawString);
    gamesVector.push_back(cppString);

    env->ReleaseStringUTFChars(string, rawString);
  }

  return gamesVector;
}

Game GameLoader::LoadGame(std::string game_name) const {
  jclass gameLoader = env->FindClass("player/GameLoader");
  jmethodID mid = env->GetStaticMethodID(gameLoader, "loadGameFromName",
                                         "(Ljava/lang/String;)Lgame/Game;");

  // convert game name to java string
  jstring j_game_name = env->NewStringUTF(game_name.c_str());
  jobject game_obj = env->CallStaticObjectMethod(gameLoader, mid, j_game_name);

  return Game(env, game_obj, game_name);
}

}  // namespace ludii
}  // namespace open_spiel

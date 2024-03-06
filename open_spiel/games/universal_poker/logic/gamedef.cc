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

#include "open_spiel/games/universal_poker/logic/gamedef.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_replace.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "third_party/cppitertools/filter.hpp"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::universal_poker::logic {

constexpr char kGamedef[] = "gamedef";
constexpr char kEndGamedef[] = "end gamedef";

std::string GamedefToOpenSpielParameters(const std::string& acpc_gamedef) {
  if (acpc_gamedef.empty()) {
    SpielFatalError("Input ACPC gamedef was empty.");
  }

  if (!absl::StrContainsIgnoreCase(acpc_gamedef, kGamedef)) {
    SpielFatalError(absl::StrCat(
        "ACPC gamedef does not have a 'gamedef' line: ", acpc_gamedef));
  }
  if (!absl::StrContainsIgnoreCase(acpc_gamedef, kEndGamedef)) {
    SpielFatalError(absl::StrCat(
        "ACPC gamedef does not have an 'end gamedef' line: ", acpc_gamedef));
  }

  // As per definition of gamedef -> "case is ignored". So we will normalize to
  // lowercase initially / when initially processing it. (Note: we will have to
  // 'correct' the capitalization for all our keys down below at the end. Since
  // OpenSpiel itself *does* care about capitalization, unlike the official ACPC
  // gamedef definition.)
  std::string gamedef_normalized =
      absl::AsciiStrToLower(absl::StripAsciiWhitespace(acpc_gamedef));

  std::vector<std::string> open_spiel_state_args = {};

  // Gamedef's definition states that: "Empty lines or lines with '#' as the
  // very first character will be ignored". (Note that this means we do NOT want
  // to treat '#' like code comments, which normally take affect even in the
  // middle of a line.)
  // Additionally, we want to skip doing anything for the 'gamedef' and
  // 'end gamedef' lines (now that we've verified they appear in it somewhere)
  // because they're not needed for the Open Spiel game state.
  const auto is_useful_line = [](absl::string_view line) {
    return !line.starts_with("#") && !line.empty() && line != kGamedef &&
           line != kEndGamedef;
  };
  for (const auto& line :
       iter::filter(is_useful_line, absl::StrSplit(gamedef_normalized, '\n'))) {
    // EDGE CASE: we should only see exactly one of either 'limit' or 'nolimit',
    // and it should be on its own line. TLDR it's like 'END GAMEDEF' in that
    // it's atypical / has no '=' in it, which would interfere with our
    // processing below. (Hence why we're immediately taking care of it here.)
    if ((line == "limit") || (line == "nolimit")) {
      open_spiel_state_args.push_back(absl::StrCat("betting=", line));
      continue;
    }
    // else line must be of the following form: key[ ]=[ ]val1[ val2 val3 ...]

    if (!absl::StrContains(line, '=')) {
      SpielFatalError(
          absl::StrCat("Gamedef line is missing its '=' character: ", line));
    }
    std::vector<std::string> key_and_values = absl::StrSplit(line, '=');

    if (key_and_values.size() != 2) {
      SpielFatalError(
          absl::StrCat("Gamedef line has wrong number of components: ", line));
    }
    auto key = std::string(absl::StripAsciiWhitespace(key_and_values[0]));
    // Note that "values" is plural on purpose - it has potentially multiple,
    // space-separated things in it!
    auto values = std::string(absl::StripAsciiWhitespace(key_and_values[1]));

    // EDGE CASE:
    // There's a bug with a downstream serializer that gets confused and errors
    // if it receives a single value in places that can potentially be multiple
    // values, e.g. firstPlayer value '1' vs '1 1' (regardless of the actual
    // number of players / betting rounds / etc).
    //
    // With the exception of the 'blind' input, there is typically no meaningful
    // difference between the value appearing a single time, vs the same exact
    // value appearing twice (separated by a space). So, as a workaround we
    // manually convert the former form to the latter.
    //
    // Yes, this is hacky. But it's also the most durable option we have until
    // we can go fix the downstream issue :)
    const std::set<std::string> optionally_multi_round_parameters = {
        "firstplayer", "raisesize", "maxraises", "numboardcards", "stack"};
    if (optionally_multi_round_parameters.contains(key) && !values.empty() &&
        !absl::StrContains(values, " ")) {
      // Note: "values" is a single integer if in this section (hence why we're
      // having this problem to begin with; see above for more details).
      LOG(INFO) << line
                << " has a potentially multi-round value defined in terms of a "
                   "single round. Transforming the value into another that is "
                   "equivalent, but defined multi-round, to prevent downstream "
                   "deserializer errors.";

      values = absl::StrCat(values, " ", values);
      LOG(INFO) << "Transformed value into another that is equivalent, but "
                   "defined as multi-round: "
                << values;
    }

    open_spiel_state_args.push_back(absl::StrCat(key, "=", values));
  }
  std::string lowercase_open_spiel_game_state = absl::StrCat(
      "universal_poker(", absl::StrJoin(open_spiel_state_args, ","), ")");

  // See below - unlike the input ACPC gamedef (where casing is ignored),
  // OpenSpiel will actually error at runtime if the arg keys aren't capitalized
  // in the exact way it expects.
  // (Note: deliberately including things like e.g. bettingAbstraction that are
  // not actually valid params for the ACPC gamedef to avoid future bugs).
  static const char* const kPossibleGameStateKeysCapitalized[] = {
      "betting",       "bettingAbstraction",
      "blind",         "boardCards",
      "firstPlayer",   "gamedef",
      "handReaches",   "maxRaises",
      "numBoardCards", "numHoleCards",
      "numPlayers",    "numRanks",
      "numRounds",     "numSuits",
      "potSize",       "raiseSize",
      "stack",
  };
  std::vector<std::pair<std::string, std::string>> replacements = {};
  for (const std::string& capitalized_key : kPossibleGameStateKeysCapitalized) {
    std::string lowercase_key = absl::AsciiStrToLower(capitalized_key);
    if (capitalized_key == lowercase_key) {
      continue;
    }

    // Regardless of order, at this point we know each parameter either is at
    // the start - and following an open paren - or is comma-separated from
    // the preceding parameter. Hence we can look for a preceding "(" or ",".
    replacements.push_back(std::make_pair(absl::StrCat("(", lowercase_key),
                                          absl::StrCat("(", capitalized_key)));
    replacements.push_back(std::make_pair(absl::StrCat(",", lowercase_key),
                                          absl::StrCat(",", capitalized_key)));
  }
  return absl::StrReplaceAll(lowercase_open_spiel_game_state, replacements);
}

}  // namespace open_spiel::universal_poker::logic

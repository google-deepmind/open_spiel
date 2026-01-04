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

#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_replace.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::universal_poker::logic {

constexpr char kGamedef[] = "gamedef";
constexpr char kEndGamedef[] = "end gamedef";

std::string GamedefToOpenSpielParameters(const std::string& acpc_gamedef) {
  if (acpc_gamedef.empty()) {
    SpielFatalError("Input ACPC gamedef was empty.");
  }

  if (!StrContainsIgnoreCase(acpc_gamedef, kGamedef)) {
    SpielFatalError(absl::StrCat("ACPC gamedef does not contain 'GAMEDEF': ",
                                 acpc_gamedef));
  }

  // Check the GAMEDEF/END GAMEDEF statements are valid and not something like
  // e.g. 'GAMEDEFfoo' or 'SPEND GAMEDEF'.
  //
  // GAMEDEF either is the very first line, in which case it should be followed
  // by an "\n", or it is not, in which case it should be both followed by an
  // "\n" AND also prefixed by another "\n".
  if (!absl::StartsWithIgnoreCase(acpc_gamedef, absl::StrCat(kGamedef, "\n")) &&
      !StrContainsIgnoreCase(acpc_gamedef,
                                   absl::StrCat("\n", kGamedef, "\n"))) {
    SpielFatalError(
        absl::StrCat("ACPC gamedef does not have 'GAMEDEF' on its own line "
                     "(please remove any trailing or prefixed characters, "
                     "including whitespace):",
                     acpc_gamedef));
  }
  // END GAMEDEF either is the very last line, in which case it should be
  // prefixed by an "\n", or it is not, in which case it should be both prefixed
  // by an "\n" AND also followed by another "\n".
  if (!StrContainsIgnoreCase(acpc_gamedef, kEndGamedef)) {
    SpielFatalError(absl::StrCat(
        "ACPC gamedef does not contain 'END GAMEDEF': ", acpc_gamedef));
  }
  if (!absl::EndsWithIgnoreCase(acpc_gamedef,
                                absl::StrCat("\n", kEndGamedef)) &&
      !StrContainsIgnoreCase(acpc_gamedef,
                             absl::StrCat("\n", kEndGamedef, "\n"))) {
    SpielFatalError(
        absl::StrCat("ACPC gamedef does not have an 'END GAMEDEF' on its own "
                     "line (please remove any trailing or prefixed characters, "
                     "including whitespace):",
                     acpc_gamedef));
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
    return !line.empty() && line[0] != '#' && line != kGamedef &&
           line != kEndGamedef;
  };
  std::vector<std::string> lines = absl::StrSplit(gamedef_normalized, '\n');
  for (const auto& line : lines) {
    // Skip lines that are not useful.
    if (!is_useful_line(line)) { continue; }

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
    if (optionally_multi_round_parameters.find(key) !=
        optionally_multi_round_parameters.end() && !values.empty() &&
        !absl::StrContains(values, " ")) {
      // Note: "values" is a single integer if in this section (hence why we're
      // having this problem to begin with; see above for more details).

      // Note: this line has a potentially multi-round value defined in terms of
      // single round. Transforming the value into another that is equivalent,
      // but defined multi-round, to prevent downstream deserializer errors.;

      values = absl::StrCat(values, " ", values);
      // Transformed value into another that is equivalent, but defined as
      // multi-round
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

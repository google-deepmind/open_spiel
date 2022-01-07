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

#include "open_spiel/utils/json.h"

#include <cmath>
#include <cstdint>
#include <string>

#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::json {

namespace {

std::string Escape(const std::string& input) {
  std::string out;
  out.reserve(input.length());
  for (const char c : input) {
    switch (c) {
      case '"': out.append("\\\""); break;
      case '\\': out.append("\\\\"); break;
      case '\b': out.append("\\b"); break;
      case '\f': out.append("\\f"); break;
      case '\n': out.append("\\n"); break;
      case '\r': out.append("\\r"); break;
      case '\t': out.append("\\t"); break;
      default: out.push_back(c); break;
    }
  }
  return out;
}

void ConsumeWhitespace(absl::string_view* str) {
  for (auto p = str->begin(); p < str->end(); ++p) {
    switch (*p) {
      case ' ':
      case '\n':
      case '\r':
      case '\t':
        break;
      default:
        str->remove_prefix(p - str->begin());
        return;
    }
  }
}

absl::nullopt_t ParseError(absl::string_view error, absl::string_view str) {
  // Comment out this check if you want parse errors to return nullopt instead
  // of crash with an error message of where the problem is.
  SPIEL_CHECK_EQ(error, str.substr(0,
                                   std::min(30, static_cast<int>(str.size()))));

  // TODO(author7): Maybe return a variant of error string or Value?
  return absl::nullopt;
}

bool ConsumeToken(absl::string_view* str, absl::string_view token) {
  if (absl::StartsWith(*str, token)) {
    str->remove_prefix(token.size());
    return true;
  }
  return false;
}

template <typename T>
absl::optional<Value> ParseConstant(absl::string_view* str,
                                    absl::string_view token, T value) {
  if (ConsumeToken(str, token)) {
    return value;
  }
  return ParseError("Invalid constant: ", *str);
}

absl::optional<Value> ParseNumber(absl::string_view* str) {
  size_t valid_double =
      std::min(str->find_first_not_of("-+.0123456789eE"), str->size());
  size_t valid_int =
      std::min(str->find_first_not_of("-0123456789"), str->size());
  if (valid_double == valid_int) {
    if (int64_t v; absl::SimpleAtoi(str->substr(0, valid_int), &v)) {
      str->remove_prefix(valid_int);
      return Value(v);
    }
  } else {
    if (double v; absl::SimpleAtod(str->substr(0, valid_double), &v)) {
      str->remove_prefix(valid_double);
      return Value(v);
    }
  }
  return ParseError("Invalid number", *str);
}

absl::optional<std::string> ParseString(absl::string_view* str) {
  if (!ConsumeToken(str, "\"")) {
    return ParseError("Expected '\"'", *str);
  }
  std::string out;
  bool escape = false;
  for (auto p = str->begin(); p < str->end(); ++p) {
    switch (*p) {
      case '\\':
        if (escape) {
          out.push_back('\\');
        }
        escape = !escape;
        break;
      case '"':
        if (escape) {
          out.push_back('"');
          escape = false;
          break;
        } else {
          str->remove_prefix(p - str->begin() + 1);
          return out;
        }
      default:
        if (escape) {
          switch (*p) {
            case 'b': out.append("\b"); break;
            case 'f': out.append("\f"); break;
            case 'n': out.append("\n"); break;
            case 'r': out.append("\r"); break;
            case 't': out.append("\t"); break;
            default: out.push_back(*p); break;
          }
          escape = false;
        } else {
          out.push_back(*p);
        }
        break;
    }
  }
  return ParseError("Unfinished string", *str);
}

absl::optional<Value> ParseValue(absl::string_view* str);

absl::optional<Array> ParseArray(absl::string_view* str) {
  if (!ConsumeToken(str, "[")) {
    return ParseError("Expected '['", *str);
  }
  Array out;
  bool first = true;
  while (!str->empty()) {
    ConsumeWhitespace(str);
    if (ConsumeToken(str, "]")) {
      return out;
    }
    if (!first && !ConsumeToken(str, ",")) {
      return ParseError("Expected ','", *str);
    }
    first = false;
    ConsumeWhitespace(str);
    absl::optional<Value> v = ParseValue(str);
    if (!v) {
      return absl::nullopt;
    }
    out.push_back(*v);
  }
  return ParseError("Unfinished array", *str);
}

absl::optional<Object> ParseObject(absl::string_view* str) {
  if (!ConsumeToken(str, "{")) {
    return ParseError("Expected '{'", *str);
  }
  Object out;
  bool first = true;
  while (!str->empty()) {
    ConsumeWhitespace(str);
    if (ConsumeToken(str, "}")) {
      return out;
    }
    if (!first && !ConsumeToken(str, ",")) {
      return ParseError("Expected ','", *str);
    }
    first = false;
    ConsumeWhitespace(str);
    absl::optional<std::string> key = ParseString(str);
    if (!key) {
      return absl::nullopt;
    }
    ConsumeWhitespace(str);
    if (!ConsumeToken(str, ":")) {
      return ParseError("Expected ':'", *str);
    }
    ConsumeWhitespace(str);
    absl::optional<Value> v = ParseValue(str);
    if (!v) {
      return absl::nullopt;
    }
    out.emplace(*key, *v);
  }
  return ParseError("Unfinished object", *str);
}

absl::optional<Value> ParseValue(absl::string_view* str) {
  ConsumeWhitespace(str);
  if (str->empty()) {
    return ParseError("Empty string", *str);
  }
  switch (str->at(0)) {
    case '-':
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9': return ParseNumber(str);
    case 'n': return ParseConstant(str, "null", Null());
    case 't': return ParseConstant(str, "true", true);
    case 'f': return ParseConstant(str, "false", false);
    case '"': return ParseString(str);
    case '[': return ParseArray(str);
    case '{': return ParseObject(str);
    default: return ParseError("Unexpected char: ", *str);
  }
}

}  // namespace

bool Null::operator==(const Null& o) const { return true; }
bool Null::operator!=(const Null& o) const { return false; }

std::string ToString(const Array& array, bool wrap, int indent) {
  std::string out = "[";
  bool first = true;
  for (const Value& v : array) {
    if (!first) {
      absl::StrAppend(&out, ",");
    }
    if (wrap) {
      absl::StrAppend(&out, "\n", std::string(indent + 2, ' '));
    } else if (!first) {
      absl::StrAppend(&out, " ");
    }
    first = false;
    absl::StrAppend(&out, json::ToString(v, wrap, indent + 2));
  }
  if (wrap) {
    absl::StrAppend(&out, "\n", std::string(indent, ' '));
  }
  absl::StrAppend(&out, "]");
  return out;
}

std::string ToString(const Object& obj, bool wrap, int indent) {
  std::string out = "{";
  bool first = true;
  for (const auto& [key, value] : obj) {
    if (!first) {
      absl::StrAppend(&out, ",");
    }
    if (wrap) {
      absl::StrAppend(&out, "\n", std::string(indent + 2, ' '));
    } else if (!first) {
      absl::StrAppend(&out, " ");
    }
    first = false;
    absl::StrAppend(&out, "\"", Escape(key), "\": ",
                    json::ToString(value, wrap, indent + 2));
  }
  if (wrap) {
    absl::StrAppend(&out, "\n", std::string(indent, ' '));
  }
  absl::StrAppend(&out, "}");
  return out;
}


std::string ToString(const Value& value, bool wrap, int indent) {
  if (value.IsNull()) {
    return "null";
  } else if (value.IsBool()) {
    return (value.GetBool() ? "true" : "false");
  } else if (value.IsInt()) {
    return std::to_string(value.GetInt());
  } else if (value.IsDouble()) {
    double v = value.GetDouble();
    if (std::isfinite(v)) {
      return std::to_string(v);
    } else {
      // It'd be nice to show an error with a path, but at least this is
      // debuggable by looking at the json. Crashing doesn't tell you where
      // the problem is.
      return absl::StrCat("\"", std::to_string(v), "\"");
    }
  } else if (value.IsString()) {
    return absl::StrCat("\"", Escape(value.GetString()), "\"");
  } else if (value.IsArray()) {
    return ToString(value.GetArray(), wrap, indent);
  } else if (value.IsObject()) {
    return ToString(value.GetObject(), wrap, indent);
  } else {
    SpielFatalError("json::ToString is missing a type.");
  }
}

std::ostream& operator<<(std::ostream& os, const Null& n) {
  return os << ToString(n);
}

std::ostream& operator<<(std::ostream& os, const Array& a) {
  return os << ToString(a);
}

std::ostream& operator<<(std::ostream& os, const Object& o) {
  return os << ToString(o);
}

std::ostream& operator<<(std::ostream& os, const Value& v) {
  return os << ToString(v);
}

absl::optional<Value> FromString(absl::string_view str) {
  return ParseValue(&str);
}

}  // namespace open_spiel::json

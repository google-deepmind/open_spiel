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

#include "open_spiel/utils/sgf_reader.h"

#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/utils/status.h"

// SGF spec here:
// - https://homepages.cwi.nl/~aeb/go/misc/sgf.html
// - https://www.red-bean.com/sgf/ff1_3/ff3.html

namespace open_spiel {
namespace {
bool IsUppercaseLetter(char c) {
  return c >= 'A' && c <= 'Z';
}
}

void SgfReader::SkipWhitespace() {
  while (index_ < sgf_string_.length() &&
         absl::ascii_isspace(sgf_string_[index_])) {
    ++index_;
  }
}

StatusWithValue<std::string> SgfReader::ReadPropertyName() {
  std::string property_name;
  while (index_ < sgf_string_.length() &&
         IsUppercaseLetter(sgf_string_[index_])) {
    property_name.push_back(sgf_string_[index_]);
    ++index_;
  }
  return StatusWithValue<std::string>(StatusValue::kOk, "", property_name);
}

StatusWithValue<std::vector<std::string>> SgfReader::ReadPropertyValues() {
  std::vector<std::string> property_values;
  while (index_ < sgf_string_.length() && sgf_string_[index_] == '[') {
    ++index_;
    std::string property_value;
    while (index_ < sgf_string_.length() && sgf_string_[index_] != ']') {
      property_value.push_back(sgf_string_[index_]);
      ++index_;
    }
    property_values.push_back(property_value);
    ++index_;
  }
  return StatusWithValue<std::vector<std::string>>(StatusValue::kOk, "",
                                                   property_values);
}

StatusWithValue<std::vector<SgfProperty>> SgfReader::ReadPropertiesAndValues() {
  std::vector<SgfProperty> properties_and_values;
  SkipWhitespace();
  while (index_ < sgf_string_.length() &&
         sgf_string_[index_] != '(' &&
         sgf_string_[index_] != ')' &&
         sgf_string_[index_] != ';') {
    StatusWithValue<std::string> property_name = ReadPropertyName();
    if (!property_name.ok()) {
      return StatusWithValue<std::vector<SgfProperty>>(
          StatusValue::kError, property_name.message(), properties_and_values);
    }
    StatusWithValue<std::vector<std::string>> property_values =
        ReadPropertyValues();
    if (!property_values.ok()) {
      return StatusWithValue<std::vector<SgfProperty>>(
          StatusValue::kError, property_values.message(),
          properties_and_values);
    }
    properties_and_values.push_back(
        SgfProperty{property_name.value(), property_values.value()});
    SkipWhitespace();
  }
  return StatusWithValue<std::vector<SgfProperty>>(StatusValue::kOk, "",
                                                   properties_and_values);
}


StatusWithValue<SgfNode> SgfReader::ReadNode() {
  SgfNode sgf_node;

  if (index_ >= sgf_string_.length()) {
    return StatusWithValue<SgfNode>(StatusValue::kOk, "", sgf_node);
  }

  // A nodes can start with a parenthesis or a semicolon.
  bool started_with_parenthesis = false;
  if (sgf_string_[index_] == '(') {
    ++index_;
    started_with_parenthesis = true;

    // Special case for the root node.
    SkipWhitespace();
    if (sgf_string_[index_] == ';') {
      ++index_;
    }
  } else if (sgf_string_[index_] == ';') {
    ++index_;
  } else {
    return StatusWithValue<SgfNode>(
        StatusValue::kError,
        "Expected a parenthesis or a semicolon to start a node.", sgf_node);
  }

  StatusWithValue<std::vector<SgfProperty>> properties_and_values =
      ReadPropertiesAndValues();
  if (!properties_and_values.ok()) {
    return StatusWithValue<SgfNode>(
        StatusValue::kError, properties_and_values.message(), sgf_node);
  }
  sgf_node.properties = properties_and_values.value();
  SkipWhitespace();

  // Read the children of the node until we see a closing parenthesis.
  while (true) {
    if (index_ >= sgf_string_.length()) {
      break;
    } else if (sgf_string_[index_] == ')') {
      // See a closing parenthesis, so we are done reading this node.
      // If the node started with a parenthesis, we need to increment the index
      // to get past the closing parenthesis, marking it as consumed.
      if (started_with_parenthesis) {
        ++index_;
      }
      // If we see a closing parenthesis but the node did not start with a
      // parenthesis, then this closing parenthesis is part of the parent node,
      // so we do not consume the character here, because the parent node will
      // handle it. But we are still done reading this node.
      break;
    } else if (sgf_string_[index_] == '(' || sgf_string_[index_] == ';') {
      // bool recurse_child = recurse && sgf_string_[index_] == '(';
      StatusWithValue<SgfNode> child_node = ReadNode();
      if (!child_node.ok()) {
        return StatusWithValue<SgfNode>(
            StatusValue::kError, child_node.message(), sgf_node);
      }
      sgf_node.children.push_back(child_node.value());
    } else {
      return StatusWithValue<SgfNode>(
          StatusValue::kError,
          "Expected a parenthesis or semicolon to start a child node.",
          sgf_node);
    }
    SkipWhitespace();
  }

  return StatusWithValue<SgfNode>(StatusValue::kOk, "", sgf_node);
}

StatusWithValue<std::vector<SgfNode>> SgfReader::ReadAllNodes() {
  SkipWhitespace();
  std::vector<SgfNode> sgf_nodes;
  while (index_ < sgf_string_.length()) {
    StatusWithValue<SgfNode> sgf_node = ReadNode();
    if (!sgf_node.ok()) {
      return StatusWithValue<std::vector<SgfNode>>(
          StatusValue::kError, sgf_node.message(), sgf_nodes);
    }
    sgf_nodes.push_back(sgf_node.value());
    SkipWhitespace();
  }
  return StatusWithValue<std::vector<SgfNode>>(StatusValue::kOk, "", sgf_nodes);
}

StatusWithValue<std::vector<SgfNode>> ReadSgfString(
  const std::string& sgf_string) {
    SgfReader sgf_reader(sgf_string);
  return sgf_reader.ReadAllNodes();
}

}  // namespace open_spiel

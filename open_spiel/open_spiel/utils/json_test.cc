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

#include "open_spiel/utils/json.h"

#include "open_spiel/spiel_utils.h"

namespace open_spiel::json {

namespace {

void TestToString() {
  SPIEL_CHECK_EQ(ToString(Null()), "null");
  SPIEL_CHECK_EQ(ToString(true), "true");
  SPIEL_CHECK_EQ(ToString(false), "false");
  SPIEL_CHECK_EQ(ToString(1), "1");
  SPIEL_CHECK_EQ(ToString(3.1415923), "3.141592");
  SPIEL_CHECK_EQ(ToString("asdf"), "\"asdf\"");
  SPIEL_CHECK_EQ(ToString(std::string("asdf")), "\"asdf\"");
  SPIEL_CHECK_EQ(ToString(Array({"asdf"})), "[\"asdf\"]");
  SPIEL_CHECK_EQ(ToString(Array({1, Null(), "asdf"})), "[1, null, \"asdf\"]");
  SPIEL_CHECK_EQ(ToString(Array({1, 2, 3})), "[1, 2, 3]");
  SPIEL_CHECK_EQ(ToString(Array({1, 2, 3, 4})), "[1, 2, 3, 4]");
  SPIEL_CHECK_EQ(ToString(Object({{"asdf", 1}, {"foo", 2}})),
                          "{\"asdf\": 1, \"foo\": 2}");
  SPIEL_CHECK_EQ(ToString(
      Object({{"asdf", Object({{"bar", 6}})}, {"foo", Array({1, 2, 3})}})),
      "{\"asdf\": {\"bar\": 6}, \"foo\": [1, 2, 3]}");
  SPIEL_CHECK_EQ(ToString(Object({{"asdf", Object({{"bar", 6}})},
                                  {"foo", Array({1, true, false})}}),
                          true),
                 R"({
  "asdf": {
    "bar": 6
  },
  "foo": [
    1,
    true,
    false
  ]
})");
}

void TestFromString() {
  absl::optional<Value> v;

  v = FromString("null");
  SPIEL_CHECK_TRUE(v);
  SPIEL_CHECK_TRUE(v->IsNull());

  v = FromString("true");
  SPIEL_CHECK_TRUE(v);
  SPIEL_CHECK_TRUE(v->IsBool());
  SPIEL_CHECK_TRUE(v->IsTrue());
  SPIEL_CHECK_EQ(v->GetBool(), true);

  v = FromString("false");
  SPIEL_CHECK_TRUE(v);
  SPIEL_CHECK_TRUE(v->IsBool());
  SPIEL_CHECK_TRUE(v->IsFalse());
  SPIEL_CHECK_EQ(v->GetBool(), false);

  v = FromString("1");
  SPIEL_CHECK_TRUE(v);
  SPIEL_CHECK_TRUE(v->IsInt());
  SPIEL_CHECK_EQ(v->GetInt(), 1);

  v = FromString("-163546");
  SPIEL_CHECK_TRUE(v);
  SPIEL_CHECK_TRUE(v->IsInt());
  SPIEL_CHECK_EQ(v->GetInt(), -163546);

  v = FromString("3.5");
  SPIEL_CHECK_TRUE(v);
  SPIEL_CHECK_TRUE(v->IsDouble());
  SPIEL_CHECK_EQ(v->GetDouble(), 3.5);

  v = FromString("\"asdf\"");
  SPIEL_CHECK_TRUE(v);
  SPIEL_CHECK_TRUE(v->IsString());
  SPIEL_CHECK_EQ(v->GetString(), "asdf");

  v = FromString(R"("as \" \\ df")");
  SPIEL_CHECK_TRUE(v);
  SPIEL_CHECK_TRUE(v->IsString());
  SPIEL_CHECK_EQ(v->GetString(), R"(as " \ df)");

  v = FromString("[\"asdf\"]");
  SPIEL_CHECK_TRUE(v);
  SPIEL_CHECK_TRUE(v->IsArray());
  SPIEL_CHECK_EQ(v->GetArray(), Array({"asdf"}));

  v = FromString("[ null, true, 1 , 3.5,  \"asdf\"  ]");
  SPIEL_CHECK_TRUE(v);
  SPIEL_CHECK_TRUE(v->IsArray());
  SPIEL_CHECK_EQ(v->GetArray(), Array({Null(), true, 1, 3.5, "asdf"}));

  v = FromString("{\"asdf\" : 1, \"foo\": 2}");
  SPIEL_CHECK_TRUE(v);
  SPIEL_CHECK_TRUE(v->IsObject());
  SPIEL_CHECK_EQ(v->GetObject(), Object({{"asdf", 1}, {"foo", 2}}));

  v = FromString(R"({
  "asdf": {
    "bar": 6
  },
  "foo": [
    1,
    true,
    false
  ]
})");
  SPIEL_CHECK_TRUE(v);
  SPIEL_CHECK_TRUE(v->IsObject());
  SPIEL_CHECK_EQ(v->GetObject(), Object({{"asdf", Object({{"bar", 6}})},
                                         {"foo", Array({1, true, false})}}));
}

void TestValue() {
  SPIEL_CHECK_EQ(Value(true), Value(true));
  SPIEL_CHECK_EQ(Value(true), true);
  SPIEL_CHECK_EQ(Value(1), Value(1));
  SPIEL_CHECK_EQ(Value(1), 1);
  SPIEL_CHECK_EQ(Value(1.5), Value(1.5));
  SPIEL_CHECK_EQ(Value(1.5), 1.5);

  SPIEL_CHECK_TRUE(Value(Null()).IsNull());
  SPIEL_CHECK_EQ(std::get<Null>(Value(Null())), Null());
  SPIEL_CHECK_EQ(Value(Null()), Null());

  SPIEL_CHECK_TRUE(Value(true).IsBool());
  SPIEL_CHECK_TRUE(Value(true).IsTrue());
  SPIEL_CHECK_TRUE(Value(false).IsBool());
  SPIEL_CHECK_TRUE(Value(false).IsFalse());
  SPIEL_CHECK_FALSE(Value(true).IsFalse());
  SPIEL_CHECK_FALSE(Value(false).IsTrue());
  SPIEL_CHECK_FALSE(Value(1).IsTrue());
  SPIEL_CHECK_FALSE(Value(1).IsFalse());
  SPIEL_CHECK_EQ(std::get<bool>(Value(true)), true);
  SPIEL_CHECK_EQ(Value(true).GetBool(), true);
  SPIEL_CHECK_EQ(Value(true), true);
  SPIEL_CHECK_EQ(Value(false).GetBool(), false);
  SPIEL_CHECK_EQ(Value(false), false);
  SPIEL_CHECK_NE(Value(true), 1);
  SPIEL_CHECK_NE(Value(false), 1.5);

  SPIEL_CHECK_TRUE(Value(1).IsInt());
  SPIEL_CHECK_TRUE(Value(1).IsNumber());
  SPIEL_CHECK_FALSE(Value(true).IsInt());
  SPIEL_CHECK_FALSE(Value(1.5).IsInt());
  SPIEL_CHECK_EQ(std::get<int64_t>(Value(1)), 1);
  SPIEL_CHECK_EQ(Value(1).GetInt(), 1);
  SPIEL_CHECK_EQ(Value(1), 1);
  SPIEL_CHECK_NE(Value(1), 2);

  SPIEL_CHECK_TRUE(Value(1.5).IsDouble());
  SPIEL_CHECK_TRUE(Value(1.5).IsNumber());
  SPIEL_CHECK_FALSE(Value(1.5).IsInt());
  SPIEL_CHECK_FALSE(Value(1.5).IsBool());
  SPIEL_CHECK_EQ(std::get<double>(Value(1.5)), 1.5);
  SPIEL_CHECK_EQ(Value(1.5).GetDouble(), 1.5);
  SPIEL_CHECK_EQ(Value(1.5), 1.5);
  SPIEL_CHECK_NE(Value(1.5), 2.5);

  SPIEL_CHECK_TRUE(Value("asdf").IsString());
  SPIEL_CHECK_TRUE(Value(std::string("asdf")).IsString());
  SPIEL_CHECK_FALSE(Value("asdf").IsArray());
  SPIEL_CHECK_EQ(Value("asdf"), "asdf");
  SPIEL_CHECK_EQ(Value("asdf"), std::string("asdf"));
  SPIEL_CHECK_EQ(Value("asdf").GetString(), "asdf");
  SPIEL_CHECK_EQ(std::get<std::string>(Value("asdf")), "asdf");

  SPIEL_CHECK_EQ(Array({1, 2, 3}), Array({1, 2, 3}));
  SPIEL_CHECK_EQ(CastToArray(std::vector<int>({1, 2, 3})), Array({1, 2, 3}));
  SPIEL_CHECK_EQ(
      TransformToArray(std::vector<unsigned int>({1u, 2u, 3u}),
                       [](unsigned int i) { return static_cast<int>(i); }),
      Array({1, 2, 3}));
  SPIEL_CHECK_TRUE(Value(Array({1, 2, 3})).IsArray());
  SPIEL_CHECK_FALSE(Value(Array({1, 2, 3})).IsObject());
  SPIEL_CHECK_EQ(Value(Array({1, 2, 3})), Array({1, 2, 3}));
  SPIEL_CHECK_EQ(Value(Array({1, 2, 3})).GetArray(), Array({1, 2, 3}));
  SPIEL_CHECK_NE(Value(Array({1, 2, 3})).GetArray(), Array({1, 3, 5}));
  SPIEL_CHECK_EQ(std::get<Array>(Value(Array({1, 2, 3}))), Array({1, 2, 3}));

  SPIEL_CHECK_EQ(Object({{"asdf", 1}, {"bar", 2}}),
                 Object({{"asdf", 1}, {"bar", 2}}));
  SPIEL_CHECK_NE(Object({{"asdf", 1}, {"bar", 2}}),
                 Object({{"asdf", 1}, {"bar", 3}}));
  SPIEL_CHECK_NE(Object({{"asdf", 1}, {"bar", 2}}),
                 Object({{"asdf", 1}, {"foo", 2}}));
  SPIEL_CHECK_EQ(Value(Object({{"asdf", 1}, {"bar", 2}})),
                 Object({{"asdf", 1}, {"bar", 2}}));
  SPIEL_CHECK_NE(Value(Object({{"asdf", 1}, {"bar", 2}})),
                 Object({{"asdf", 1}}));
  SPIEL_CHECK_TRUE(Value(Object({{"asdf", 1}, {"bar", 2}})).IsObject());
  SPIEL_CHECK_FALSE(Value(Object({{"asdf", 1}, {"bar", 2}})).IsArray());
  SPIEL_CHECK_EQ(Value(Object({{"asdf", 1}, {"bar", 2}})).GetObject(),
                 Object({{"asdf", 1}, {"bar", 2}}));
  SPIEL_CHECK_EQ(std::get<Object>(Value(Object({{"asdf", 1}, {"bar", 2}}))),
                 Object({{"asdf", 1}, {"bar", 2}}));
}

}  // namespace

}  // namespace open_spiel::json


int main(int argc, char** argv) {
  open_spiel::json::TestToString();
  open_spiel::json::TestFromString();
  open_spiel::json::TestValue();
}

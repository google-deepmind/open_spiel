#!/bin/bash
#
# Copyright 2022 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Links to sub documents within tables are not properly converted.
# E.g. a reference to a separate markdown table is not converted to the
# corresponding .html in Read the Docs.
#
# This is an open issue with sphinx-markdown-tables, see
# https://github.com/ryanfox/sphinx-markdown-tables/issues/18

if [[ "$READTHEDOCS" = "True" ]]; then
  # Fix the links pre-build. In this case, edit the markdown file rather than
  # the resulting HTML
  FILE="docs/api_reference.md"
  if [[ "$1" != "" ]]; then
    FILE="$1"
  fi
  sed -E 's/\[Python\]\((.*).md\)/\[Python\]\(\1.html\)/g' -i ${FILE}
else
  # Fix the links post-build: rewrite the HTML after it's been generated. Was
  # not able to get this to work on Read the Docs.
  FILE="_build/html/api_reference.html"
  if [[ "$1" != "" ]]; then
    FILE="$1"
  fi
  sed -E 's/a href="(.*)\.md"/a href="\1\.html"/g' -i ${FILE}
fi


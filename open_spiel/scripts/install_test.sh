#!/usr/bin/env bash
# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Tests OPEN_SPIEL_INSTALL_SYSTEM_DEPS in install.sh, offline and without
# changing the machine: install.sh runs in a scratch copy of the scripts it
# needs, with git, sudo, apt-get, brew and the other package commands replaced
# by stubs that record each call. Run from the repository root:
#
#   ./open_spiel/scripts/install_test.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRATCH="$(mktemp -d)"
trap 'rm -r -- "$SCRATCH"' EXIT
FAILURES=0

fail() {
  echo "FAIL: $*" 1>&2
  FAILURES=$((FAILURES + 1))
}

# A fresh checkout holding only install.sh and the scripts it sources, with a
# stub directory first on PATH.
setup() {
  local case_dir="$SCRATCH/$1"
  mkdir -p "$case_dir/repo/open_spiel/scripts" "$case_dir/bin" \
    "$case_dir/repo/open_spiel/games/bridge"
  cp "$ROOT/install.sh" "$case_dir/repo/"
  cp "$ROOT/open_spiel/scripts/install.sh" \
     "$ROOT/open_spiel/scripts/global_variables.sh" \
     "$case_dir/repo/open_spiel/scripts/"
  # git: `clone ... <dir>` makes <dir>; anything else (checkout) succeeds.
  cat > "$case_dir/bin/git" <<'EOF'
#!/usr/bin/env bash
if [[ "$1" == "clone" ]]; then mkdir -p "${@: -1}"; fi
exit 0
EOF
  for cmd in sudo apt-get brew add-apt-repository update-alternatives; do
    printf '#!/usr/bin/env bash\necho "%s $*" >> "%s/calls"\n' \
      "$cmd" "$case_dir" > "$case_dir/bin/$cmd"
  done
  # dpkg-query reports every system package missing, so the apt-get
  # commands are reached on any Debian host, provisioned or not.
  printf '#!/usr/bin/env bash\nexit 1\n' > "$case_dir/bin/dpkg-query"
  printf '#!/usr/bin/env bash\necho "cmake version 3.30.0"\n' \
    > "$case_dir/bin/cmake"
  chmod +x "$case_dir/bin/"*
  : > "$case_dir/calls"
  echo "$case_dir"
}

# Runs install.sh in a case directory, with every optional download off.
run_install() {
  local case_dir="$1"
  shift
  (cd "$case_dir/repo" && env PATH="$case_dir/bin:$PATH" \
     DPKG_QUERY="$case_dir/bin/dpkg-query" \
     OPEN_SPIEL_BUILD_WITH_HANABI=OFF OPEN_SPIEL_BUILD_WITH_ACPC=OFF \
     OPEN_SPIEL_BUILD_WITH_XINXIN=OFF OPEN_SPIEL_BUILD_WITH_ROSHAMBO=OFF \
     OPEN_SPIEL_BUILD_WITH_LIBNOP=OFF OPEN_SPIEL_BUILD_WITH_LIBTORCH=OFF \
     OPEN_SPIEL_BUILD_WITH_ORTOOLS=OFF OPEN_SPIEL_BUILD_WITH_JULIA=OFF \
     CI= TRAVIS= "$@" bash ./install.sh > "$case_dir/out" 2>&1)
}

# Deps only: every source dependency is fetched, no package command runs, and
# each skipped command is listed.
case_dir="$(setup deps_only)"
if run_install "$case_dir" OPEN_SPIEL_INSTALL_SYSTEM_DEPS=0; then
  for dir in pybind11 open_spiel/abseil-cpp open_spiel/json \
             open_spiel/pybind11_json open_spiel/pybind11_abseil \
             open_spiel/games/bridge/double_dummy_solver; do
    [[ -d "$case_dir/repo/$dir" ]] || fail "deps only: $dir was not fetched"
  done
  [[ ! -s "$case_dir/calls" ]] || \
    fail "deps only: package commands ran: $(cat "$case_dir/calls")"
  if [[ "$OSTYPE" == "linux-gnu" || "$OSTYPE" == "darwin"* ]]; then
    grep -q "system package commands were skipped" "$case_dir/out" || \
      fail "deps only: no list of the skipped commands"
  fi
else
  fail "deps only: install.sh failed: $(tail -5 "$case_dir/out")"
fi

# Deps only on a Linux with no apt-get (Fedora, Arch): PATH holds only the
# tools install.sh needs, and it still succeeds and lists what it skipped.
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  case_dir="$(setup no_apt)"
  for cmd in sudo apt-get dpkg-query; do rm -- "$case_dir/bin/$cmd"; done
  mkdir "$case_dir/sys"
  for tool in bash env python3 realpath dirname basename mkdir cp cat head \
              sed sort grep which tail; do
    ln -s "$(command -v "$tool")" "$case_dir/sys/$tool"
  done
  if (cd "$case_dir/repo" && env -i HOME="$HOME" \
        PATH="$case_dir/bin:$case_dir/sys" \
        DPKG_QUERY="$case_dir/bin/dpkg-query" \
        OPEN_SPIEL_BUILD_WITH_HANABI=OFF OPEN_SPIEL_BUILD_WITH_ACPC=OFF \
        OPEN_SPIEL_BUILD_WITH_XINXIN=OFF OPEN_SPIEL_BUILD_WITH_ROSHAMBO=OFF \
        OPEN_SPIEL_BUILD_WITH_LIBNOP=OFF OPEN_SPIEL_BUILD_WITH_LIBTORCH=OFF \
        OPEN_SPIEL_BUILD_WITH_ORTOOLS=OFF OPEN_SPIEL_INSTALL_SYSTEM_DEPS=0 \
        bash ./install.sh > "$case_dir/out" 2>&1); then
    [[ -d "$case_dir/repo/open_spiel/abseil-cpp" ]] || \
      fail "no apt-get: abseil-cpp was not fetched"
    grep -q "system package commands were skipped" "$case_dir/out" || \
      fail "no apt-get: no list of the skipped commands"
    [[ ! -s "$case_dir/calls" ]] || \
      fail "no apt-get: package commands ran: $(cat "$case_dir/calls")"
  else
    fail "no apt-get: install.sh failed: $(tail -5 "$case_dir/out")"
  fi
fi

# A value other than 0 or 1 stops install.sh before it does anything.
case_dir="$(setup bad_value)"
if run_install "$case_dir" OPEN_SPIEL_INSTALL_SYSTEM_DEPS=no; then
  fail "bad value: install.sh succeeded"
else
  grep -q "OPEN_SPIEL_INSTALL_SYSTEM_DEPS must be 0 or 1" "$case_dir/out" || \
    fail "bad value: no message naming the variable"
  [[ ! -e "$case_dir/repo/pybind11" ]] || fail "bad value: it fetched first"
fi

# The default still runs the package commands, each announced before it runs.
case_dir="$(setup default)"
if run_install "$case_dir"; then
  [[ -d "$case_dir/repo/open_spiel/abseil-cpp" ]] || \
    fail "default: abseil-cpp was not fetched"
  while IFS= read -r call; do
    [[ -z "$call" ]] && continue
    grep -qF "install.sh runs a system package command: $call" \
      "$case_dir/out" || fail "default: '$call' ran unannounced"
  done < "$case_dir/calls"
  if [[ "$OSTYPE" == "linux-gnu" ]]; then
    grep -q "^sudo apt-get -y install " "$case_dir/calls" || \
      fail "default: sudo apt-get -y install did not run"
  fi
  if [[ "$OSTYPE" == "darwin"* ]]; then
    grep -q "^brew install virtualenv" "$case_dir/calls" || \
      fail "default: brew install virtualenv did not run"
  fi
  ! grep -q "were skipped" "$case_dir/out" || \
    fail "default: it reported skipped commands"
else
  fail "default: install.sh failed: $(tail -5 "$case_dir/out")"
fi

if [[ "$FAILURES" -gt 0 ]]; then
  echo "install_test.sh: $FAILURES failure(s)" 1>&2
  exit 1
fi
echo "install_test.sh: all cases passed"

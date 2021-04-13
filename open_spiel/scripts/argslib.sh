#!/usr/bin/env bash

# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
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

# An ultra-simple comand-line arguments library for bash. Does not support
# spaces in string arguments or argument names. Also: looking up a flag is
# linear since it's an iterative through the array of names (and parsing a
# value, since its type needs to be looked up); this is in order to support bash
# 3 on MacOS, which does not support associative arrays. Worst case: parsing all
# the arguments is quadratic in the number of arguments. Hence, this is a
# barebones command-line argument library and it should only be used for simple
# use cases. There are better libraries for complex use cases (e.g. shflags);
# please use them instead!
#
# Run script with single --help to print argument helper text.
#
# Example usage:
#   source argslib.sh
#   # Main function is: ArgsLibAddArg name type default_value helper_string
#   #   where type is one of { bool, int, float, string }
#   ArgsLibAddArg arg1 bool true "Arg1 helper text"
#   ArgsLibAddArg arg2 int 4 "Arg2 helper text"
#   ArgsLibAddArg arg3 float 0.3 "Arg3 helper text"
#   ArgsLibAddArg arg4 string helloworld "Arg4 helper text"
#   ArgsLibParse $@
#   ArgsLibPrintAll  # optional!
#   echo $ARG_arg1
#   echo $ARG_arg2
#      .
#      .
#      .
#

if [[ -z ${argslib_n} ]];
then
  argslib_n=0
  declare -a argslib_names
  declare -a argslib_types
  declare -a argslib_defaults
  declare -a argslib_values
  declare -a argslib_desc
fi

function _die {
  echo "$1"
  exit -1
}

function _print_usage_exit {
  echo "$0 arguments:"
  echo ""
  for (( i=0; i<$argslib_n; i++ ))
  do
    j=`expr $i - 1`
    echo -n "--${argslib_names[$i]} ("
    echo -n "${argslib_types[$i]}): "
    echo "[defval=${argslib_defaults[$i]}]"
    echo "  ${argslib_desc[$i]}"
    echo ""
  done
  exit
}

function _check_parse_value {
  # type value
  # TODO: check the values based on the type
  case $1 in
    bool)
      if [ "$2" != "false" -a "$2" != "true" ]
      then
        _die "Invalid boolean value: $2"
      fi
      ;;
    int)
      if ! [[ "$2" =~ ^[-+]?[0-9]+$ ]];
      then
        _die "Invalid integer value: $2"
      fi
      ;;
    float)
      if ! [[ "$2" =~ ^[-+]?[0-9]+[\.]?[0-9]*$ ]];
      then
        _die "Invalid float value: $2"
      fi
      ;;
    string)
      # Anything goes
      ;;
    *)
      _die "Unrecognized argument type: $1"
      ;;
  esac
  return 0
}

function _parse_arg {
  # one argment: --name=value
  IFS="=" read -ra parts <<< $@
  [ ${#parts[@]} -eq 2 ] || _die "Incorrect syntax: $@"
  for (( i=0; i<$argslib_n; i++ ))
  do
    if [ "${parts[0]}" = "--${argslib_names[$i]}" ]
    then
      _check_parse_value ${argslib_types[$i]} ${parts[1]}
      argslib_values[$i]=${parts[1]}
      setvalcmd="ARG_${argslib_names[$i]}=${parts[1]}"
      # echo $setvalcmd
      eval $setvalcmd
      return 0
    fi
  done
  _die "Argument not defined: ${parts[0]}"
  return 1
}

function ArgsLibAddArg {
  [ ${#@} -eq 4 ] || _die "Incorrect number of arguments for AddArg"
  _check_parse_value $2 $3

  # Not found? Append it to the end.
  argslib_names[$argslib_n]=$1
  argslib_types[$argslib_n]=$2
  argslib_defaults[$argslib_n]=$3
  argslib_desc[$argslib_n]=$4
  let argslib_n=argslib_n+1

  _parse_arg "--$1=$3"
}

function ArgsLibPrintAll {
  echo "$argslib_n command-line argument(s)"
  for (( i=0; i<$argslib_n; i++ ))
  do
    echo -n "  ${argslib_names[$i]} ("
    echo -n "${argslib_types[$i]}): "
    echo -n "${argslib_values[$i]} "
    echo "[defval=${argslib_defaults[$i]}]"
  done
}

function ArgsLibParse {
  if [ "$1" = "--help" ]
  then
    _print_usage_exit
  fi

  # Parse arguments in the form --name=value
  for arg in $@
  do
    [[ $arg == --* ]] || _die "Invalid syntax for argument name: $arg"
    _parse_arg $arg
  done
}

function ArgsLibGet {
  [ ${#@} -eq 1 ] || _die "Incorrect number of arguments for ArgsLibGet"
  for (( i=0; i<$argslib_n; i++ ))
  do
    if [ "$1" = ${argslib_names[$i]} ]
    then
      echo ${argslib_values[$i]}
      return 0
    fi
  done
  echo ""
  return 1
}

function ArgsLibSet {
  [ ${#@} -eq 2 ] || _die "Incorrect number of arguments for ArgsLibSet"
  for (( i=0; i<$argslib_n; i++ ))
  do
    if [ "$1" = ${argslib_names[$i]} ]
    then
      _parse_arg "--$1=$2"
      return 0
    fi
  done
  echo "Arg $1 not found"
  return 1
}


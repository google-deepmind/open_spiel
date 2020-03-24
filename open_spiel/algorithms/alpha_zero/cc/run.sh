#!/bin/bash

set -e  # Exit on any error.

source gbash.sh || exit 1

DEFINE_string build_config "dmtf_cuda" "Which build config to use."

GBASH_PASSTHROUGH_UNKNOWN_FLAGS=1
gbash::init_google "$@"
set -- "${GBASH_ARGV[@]}"

blaze run --config "${FLAGS_build_config}" \
  //learning/deepmind/research/agents/open_spiel_alpha_zero:main -- "$@"

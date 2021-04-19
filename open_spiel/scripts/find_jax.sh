#!/usr/bin/env bash

read -r -d '' TESTSCRIPT << EOT
import jax
import jaxlib
import haiku
import chex
import optax
print(jax.__version__)
EOT

PY_EXEC=$(which $1)
if [[ ! -x $PY_EXEC ]]
then
  echo "Python executable: $PY_EXEC not found or not executable."
  exit -1
fi

echo "$TESTSCRIPT" | $PY_EXEC
exit $?

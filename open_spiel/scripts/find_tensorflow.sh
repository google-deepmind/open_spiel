#!/usr/bin/env bash

read -r -d '' TESTSCRIPT << EOT
import tensorflow as tf
import tensorflow_probability
print(tf.__version__)
EOT

PY_EXEC=$(which $1)
if [[ ! -x $PY_EXEC ]]
then
  echo "Python executable: $PY_EXEC not found or not executable."
  exit -1
fi

echo "$TESTSCRIPT" | $PY_EXEC
exit $?

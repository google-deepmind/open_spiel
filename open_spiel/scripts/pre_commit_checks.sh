#!/bin/bash
CHANGED_FILES=$(git diff --cached --name-only --diff-filter=ACMR)
if [[ -z "$CHANGED_FILES" ]]; then
    echo "No staged files."
    exit 0
fi
pre-commit run --files $CHANGED_FILES

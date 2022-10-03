#!/usr/bin/env sh

### Run all linters used by this project. This script is used in CI.
### Its main use is that it runs _all_ linters, even if one of them errors.

if [ $# -eq 0 ]; then
    echo >&2 "Usage $0 SRC_PATH"
    echo >&2 ''
    echo >&2 'Run all linters on given path, even if one fails.'
    echo >&2 'error: no source path passed (usually ./src/)'
    exit 1
fi

exit_code=0
for cmd in "mypy" "black --check" "isort --check" "pylint"; do
  echo "Running $cmd ..."
  python -m $cmd "$@"
  # Use bitwise OR instead of addition to accumulate exit codes.
  # Exit codes are 8 bits wide and must not overflow to zero.
  exit_code="$((exit_code | $?))"
done

exit "$exit_code"

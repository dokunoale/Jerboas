#!/usr/bin/env bash
#
# Run the test suite in a container:  ./test.sh [pytest args...]
#
# The same Dockerfile the use cases are served from, built with every extra so
# torch and the pandas conversion are exercised too. The source is mounted
# rather than copied, so a run tests the working tree and not the last build --
# only the dependency layer is baked, and it is cached until pyproject changes.
set -euo pipefail

cd "$(dirname "$0")"

IMAGE=jerboas-test

container build --build-arg EXTRAS=api,torch,dev,pandas -t "$IMAGE" . >/dev/null

exec container run --rm \
    -v "$PWD/jerboas:/app/jerboas" \
    -v "$PWD/tests:/app/tests" \
    -v "$PWD/pytest.ini:/app/pytest.ini" \
    -w /app \
    "$IMAGE" \
    python -m pytest "${@:-tests}" -q

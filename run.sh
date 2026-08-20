#!/usr/bin/env bash
#
# Serve one use case:  ./run.sh coldstart
#
# Every directory under usecase/ is one, and the image holds them all -- the
# argument only picks which one this container serves, so switching between them
# rebuilds nothing and starting one replaces whatever was running. The container
# takes the use case's name, which with a local DNS domain registered is also
# its address: <usecase>.test.
set -euo pipefail

cd "$(dirname "$0")"

IMAGE=jerboas
DOMAIN=test

# A guest gets 1GB unless told otherwise, which spotify outgrows about ten
# seconds into loading its graph -- and the kernel killing it leaves no
# traceback, just a startup that stops mid-sentence. 4GB is roughly three times
# what the default subgraph peaks at while fitting; the larger ones want more:
#
#     MEMORY=12g ./run.sh spotify
MEMORY="${MEMORY:-4g}"

available() {
    find usecase -mindepth 2 -maxdepth 2 -name app.py -exec dirname {} \; | xargs -n1 basename
}

USECASE="${1:-}"
if [ ! -f "usecase/${USECASE}/app.py" ]; then
    echo "usage: $0 <usecase>" >&2
    echo "available: $(available | tr '\n' ' ')" >&2
    exit 2
fi

container build -t "$IMAGE" .

# The domain has to be created once, as an administrator, so this only says how.
# Not resolving is a missing convenience rather than a failure: localhost works
# either way.
if ! container system dns ls | tail -n +2 | grep -qx "$DOMAIN"; then
    echo
    echo "note: ${USECASE}.${DOMAIN} will not resolve until the domain exists:"
    echo "      sudo container system dns create ${DOMAIN}"
fi

# one service at a time: whoever holds the port gives it up
for name in $(available); do
    container stop "$name" >/dev/null 2>&1 || true
done

echo
echo "serving ${USECASE} on http://localhost:8000 and http://${USECASE}.${DOMAIN}:8000"
echo

# data/ and checkpoints/ are mounted rather than built in: the datasets are not
# ours to redistribute, and a fitted model living in the image would be refitted
# by every rebuild.
exec container run --rm \
    --name "$USECASE" \
    --dns-domain "$DOMAIN" \
    -m "$MEMORY" \
    -e "USECASE=$USECASE" \
    -p 8000:8000 \
    -v "$PWD/data:/app/data" \
    -v "$PWD/checkpoints:/app/checkpoints" \
    "$IMAGE"

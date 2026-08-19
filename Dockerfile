FROM python:3.12-slim

WORKDIR /app

# Dependency layer, cached separately from the source: editing jerboas/ or a use
# case must not re-download torch. pyproject.toml is the only thing copied in,
# so this layer invalidates when the dependency list changes and at no other
# time -- the files it names (README, LICENSE, the package itself) are stubbed,
# because nothing here is kept except what pip installed.
#
# Stubbing in /tmp and throwing it away, rather than in /app, matters: setuptools
# leaves a build/ directory behind and will not re-copy a source file that is no
# older than the one already in it. Built in place, the stub's empty
# __init__.py outlives the real one and ends up in the installed package.
COPY pyproject.toml /tmp/deps/
RUN cd /tmp/deps \
    && touch README.md LICENSE NOTICE \
    && mkdir jerboas && touch jerboas/__init__.py \
    && pip install --no-cache-dir "/tmp/deps[api,torch]" \
    && rm -rf /tmp/deps

# Real source, installed without re-resolving dependencies. --force-reinstall
# because the stub is already installed under this exact version, and pip would
# otherwise call the requirement satisfied and keep serving the empty one.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY jerboas ./jerboas
RUN pip install --no-cache-dir --no-deps --force-reinstall .

# Every use case, so one image serves any of them: which one is a run-time
# choice (USECASE), not a build-time one. `--app-dir` puts that directory on the
# path without making it a package, so a use case stays a plain directory that
# can hold whatever modules of its own it likes.
COPY usecase ./usecase
ENV USECASE=coldstart

# Dataset is bind-mounted at runtime (see run.sh) rather than baked into the
# image, since data/ is excluded from the build context via .dockerignore.

# The embedding is fitted on first boot and loaded afterwards. Mount this to keep
# it across restarts; without a mount every start refits it -- about 10s on CPU,
# which is quicker here than MPS, the model being small enough that kernel launch
# overhead dominates the arithmetic.
VOLUME /app/checkpoints

EXPOSE 8000
# shell form, so USECASE expands; `exec` keeps uvicorn as PID 1 and signals reach it
CMD exec uvicorn app:app --app-dir "usecase/$USECASE" --host 0.0.0.0 --port 8000

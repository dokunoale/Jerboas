FROM python:3.12-slim

# uv, for parallel, resumable, retrying installs (pip's single-stream download
# timed out on the big wheels -- gradio ~31 MB, polars-runtime ~46 MB, scipy
# ~34 MB -- over a slow link). The uv image is tiny and only ships the binary.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# uv's own stalls: read timeout and retries, so a slow link does not abort
ENV UV_HTTP_TIMEOUT=300

WORKDIR /app

# Dependency layer, cached separately from the source: editing jerboas/ or a use
# case must not re-download torch. pyproject.toml is the only thing copied in,
# so this layer invalidates when the dependency list changes and at no other
# time -- the files it names (README, LICENSE, the package itself) are stubbed,
# because nothing here is kept except what uv installed. EXTRAS picks what the
# image is for: serving a use case by default, or the whole of it when the image
# is being built to run the tests (see test.sh).
ARG EXTRAS=api,torch,ui

# Torch, from the CPU index rather than PyPI. The default wheel for aarch64 is a
# CUDA build -- 2.9GB of nvidia-* and triton that nothing here can reach, there
# being no GPU passthrough: torch.cuda.is_available() is False in this image and
# device() picks "cpu" anyway. Seeded before the install below, which then finds
# torch>=2.5 already satisfied and leaves it alone, so pyproject stays the only
# place a version is bound. Guarded, because EXTRAS need not ask for torch.
RUN case ",${EXTRAS}," in *,torch,*) \
        uv pip install --system --no-cache \
            --index-url https://download.pytorch.org/whl/cpu torch ;; \
    esac

COPY pyproject.toml /tmp/deps/
RUN cd /tmp/deps \
    && touch README.md LICENSE NOTICE \
    && mkdir jerboas && touch jerboas/__init__.py \
    && uv pip install --system --no-cache "/tmp/deps[${EXTRAS}]" \
    && rm -rf /tmp/deps

# Real source, installed without re-resolving dependencies. --reinstall because
# the stub is already installed under this exact version, and uv would otherwise
# call the requirement satisfied and keep serving the empty one.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY jerboas ./jerboas
RUN uv pip install --system --no-deps --no-cache --reinstall .

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
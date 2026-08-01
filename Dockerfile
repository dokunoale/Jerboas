FROM python:3.12-slim

WORKDIR /app

# Dependency layer, cached separately from the source: editing example.py or the
# package must not re-download torch. README.md and LICENSE come along because
# pyproject.toml reads both for its metadata.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY jerboas ./jerboas
RUN pip install --no-cache-dir ".[api,torch]"

COPY example.py ./
COPY data ./data

# The embedding is fitted on first boot and loaded afterwards. Mount this to keep
# it across restarts; without a mount every start refits it -- about 10s on CPU,
# which is quicker here than MPS, the model being small enough that kernel launch
# overhead dominates the arithmetic.
VOLUME /app/checkpoints

EXPOSE 8000
CMD ["uvicorn", "example:app", "--host", "0.0.0.0", "--port", "8000"]

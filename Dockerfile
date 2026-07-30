FROM python:3.12-slim

WORKDIR /app

# dependency layer cached separately from source, so editing example.py/jerboas
# doesn't force a reinstall of numpy/scipy on the next build
COPY pyproject.toml ./
COPY jerboas ./jerboas
RUN pip install --no-cache-dir ".[api]"

COPY example.py ./
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "example:app", "--host", "0.0.0.0", "--port", "8000"]

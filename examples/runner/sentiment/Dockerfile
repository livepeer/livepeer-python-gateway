FROM python:3.11-slim

WORKDIR /app

# SDK install (in-repo source until livepeer-gateway publishes; will collapse
# to a single `pip install livepeer-gateway` line once on PyPI).
COPY pyproject.toml README ./
COPY src /app/src
RUN pip install --no-cache-dir /app

# Pipeline-specific deps. The requirements.txt sets --extra-index-url to
# pull the CPU-only torch wheel (~200 MB vs ~5 GB for the CUDA variant).
COPY examples/runner/sentiment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake model weights at build time so setup() loads from local disk in
# milliseconds.
COPY examples/runner/sentiment/prepare_models.py /app/prepare_models.py
RUN python /app/prepare_models.py

# Pipeline code last so edits don't invalidate the bake layer above.
COPY examples/runner/sentiment/pipeline.py /app/pipeline.py

EXPOSE 5000

CMD ["python", "/app/pipeline.py"]

FROM python:3.11-slim

WORKDIR /app

# SDK install (in-repo source until livepeer-gateway publishes; will collapse
# to a single `pip install livepeer-gateway` line once on PyPI).
COPY pyproject.toml README ./
COPY src /app/src
RUN pip install --no-cache-dir /app

# Pipeline code last so edits don't invalidate the SDK install layer.
COPY examples/runner/live_tint/pipeline.py /app/pipeline.py

EXPOSE 5000

CMD ["python", "/app/pipeline.py"]

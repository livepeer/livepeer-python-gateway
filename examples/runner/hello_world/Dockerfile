FROM python:3.11-slim

WORKDIR /app

# Install the package from source — pulls aiohttp, grpcio, protobuf, av per
# pyproject.toml. Build context is the repo root.
COPY pyproject.toml README ./
COPY src /app/src
RUN pip install --no-cache-dir /app

COPY examples/runner/hello_world/pipeline.py /app/pipeline.py

EXPOSE 5000

CMD ["python", "/app/pipeline.py"]

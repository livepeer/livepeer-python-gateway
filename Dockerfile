# ---- Builder stage ----
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir . fastapi uvicorn[standard] python-multipart pillow protobuf

# ---- Runtime stage ----
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home appuser

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY web/ web/
COPY browser/ browser/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "web.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--timeout-keep-alive", "75", \
     "--workers", "1"]

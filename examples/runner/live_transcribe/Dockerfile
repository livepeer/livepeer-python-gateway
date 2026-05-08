FROM python:3.11-slim

WORKDIR /app

# SDK install (in-repo source until livepeer-gateway publishes; will collapse
# to a single `pip install livepeer-gateway` line once on PyPI).
COPY pyproject.toml README ./
COPY src /app/src
RUN pip install --no-cache-dir /app

# Whisper deps + model. faster-whisper pulls in CTranslate2 wheels (CPU build);
# pre-downloading the model avoids cold-start delay on first stream.
RUN pip install --no-cache-dir faster-whisper>=1.0.0 \
 && python -c "from faster_whisper import WhisperModel; \
WhisperModel('tiny.en', device='cpu', compute_type='int8')"

# Pipeline code last so edits don't invalidate the SDK install layer.
COPY examples/runner/live_transcribe/pipeline.py /app/pipeline.py

EXPOSE 5000

CMD ["python", "/app/pipeline.py"]

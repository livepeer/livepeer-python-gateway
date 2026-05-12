FROM python:3.11-slim

WORKDIR /app

# OpenCV (ultralytics dep) needs libgl1 + libglib2.0-0 at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# SDK install (in-repo source until livepeer-gateway publishes; will collapse
# to a single `pip install livepeer-gateway` line once on PyPI).
COPY pyproject.toml README ./
COPY src /app/src
RUN pip install --no-cache-dir /app

# Pipeline-specific deps + pre-download both models. Whisper via faster-whisper
# (CTranslate2 CPU build), detection via ultralytics YOLO. Baking the weights
# avoids cold-start network fetches at stream-start.
RUN pip install --no-cache-dir \
    faster-whisper>=1.0.0 \
    "ultralytics>=8.0.0,<9" \
 && python -c "from faster_whisper import WhisperModel; \
WhisperModel('tiny.en', device='cpu', compute_type='int8')" \
 && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Pipeline code last so edits don't invalidate the model bake layer above.
COPY examples/runner/live_detect/pipeline.py /app/pipeline.py

EXPOSE 5000

CMD ["python", "/app/pipeline.py"]

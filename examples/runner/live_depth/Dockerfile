FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# SDK install (in-repo source until livepeer-gateway publishes; will collapse
# to a single `pip install livepeer` line once on PyPI).
COPY pyproject.toml README ./
COPY src /app/src
RUN pip install --no-cache-dir /app

# Bake the depth model into the image so first stream skips the download.
# Switch variant by editing this line and `_MODEL_ID` in pipeline.py.
#
# transformers<4.49 because 4.49+ uses torch.library.custom_op signatures
# that torch 2.4 (our base) can't parse. Lift after upgrading to torch>=2.5.
RUN pip install --no-cache-dir \
        "transformers>=4.45,<4.49" \
        accelerate>=0.34 \
        opencv-python-headless>=4.10 \
        pillow>=10.0 \
 && python -c "from transformers import pipeline; \
pipeline(task='depth-estimation', model='depth-anything/Depth-Anything-V2-Base-hf')"

# Pipeline code last so edits don't invalidate the SDK install layer.
COPY examples/runner/live_depth/pipeline.py /app/pipeline.py

EXPOSE 5000

CMD ["python", "/app/pipeline.py"]

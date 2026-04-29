"""Download model weights into the local HF cache at build time.

Invoked by the Dockerfile so ``setup()`` loads from local disk in
milliseconds instead of pulling from HF Hub on every container start.
"""

from transformers import pipeline

pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

"""Download model weights into the local HF cache at build time.

Invoked by the Dockerfile so ``setup()`` loads from local disk in
milliseconds instead of pulling from HF Hub on every container start.
"""

from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

model_id = "caidas/swin2SR-classical-sr-x2-64"
Swin2SRImageProcessor.from_pretrained(model_id)
Swin2SRForImageSuperResolution.from_pretrained(model_id)

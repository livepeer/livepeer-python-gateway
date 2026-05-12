"""Image upscale BYOC pipeline. Run via ``docker compose up`` — see README.md."""

import base64
import io
import logging

import numpy as np
import torch
from PIL import Image
from pydantic import Base64Bytes, BaseModel, Field
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

from livepeer_gateway.runner import Pipeline, serve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


class UpscaleInput(BaseModel):
    image: Base64Bytes = Field(description="Source image, base64-encoded PNG/JPEG")


class UpscaleOutput(BaseModel):
    image: str = Field(description="Upscaled image, base64-encoded PNG")
    width: int
    height: int


class ImageUpscaler(Pipeline):
    def setup(self):
        # Loads from local HF cache populated at Docker build time.
        model_id = "caidas/swin2SR-classical-sr-x2-64"
        self.processor = Swin2SRImageProcessor.from_pretrained(model_id)
        self.model = Swin2SRForImageSuperResolution.from_pretrained(model_id)

    def run(self, params: UpscaleInput) -> UpscaleOutput:
        src = Image.open(io.BytesIO(params.image)).convert("RGB")

        inputs = self.processor(images=src, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # CHW float [0, 1] → HWC uint8
        chw = outputs.reconstruction.squeeze().clamp(0, 1).cpu().numpy()
        hwc = np.moveaxis(chw, 0, -1)
        upscaled = Image.fromarray((hwc * 255.0).round().astype(np.uint8))

        buf = io.BytesIO()
        upscaled.save(buf, format="PNG")
        return UpscaleOutput(
            image=base64.b64encode(buf.getvalue()).decode(),
            width=upscaled.width,
            height=upscaled.height,
        )


if __name__ == "__main__":
    serve(ImageUpscaler())

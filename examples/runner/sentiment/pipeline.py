"""Sentiment-analysis BYOC pipeline. Run via ``docker compose up`` — see README.md."""

from typing import Literal

from pydantic import BaseModel, Field
from transformers import pipeline as hf_pipeline

from livepeer_gateway.runner import Pipeline, serve


class SentimentInput(BaseModel):
    text: str = Field(description="Text to classify", examples=["I love this!"])


class SentimentOutput(BaseModel):
    label: Literal["POSITIVE", "NEGATIVE"]
    score: float = Field(ge=0.0, le=1.0)
    text: str


class SentimentAnalyzer(Pipeline):
    def setup(self):
        # Loads from local HF cache populated at Docker build time.
        self.model = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

    def run(self, params: SentimentInput) -> SentimentOutput:
        result = self.model(params.text)[0]
        return SentimentOutput(
            label=result["label"],
            score=float(result["score"]),
            text=params.text,
        )


if __name__ == "__main__":
    serve(SentimentAnalyzer())

"""Sentiment-analysis BYOC pipeline. Run via ``docker compose up`` — see README.md."""

from livepeer_gateway.runner import Pipeline, serve
from transformers import pipeline as hf_pipeline


class SentimentAnalyzer(Pipeline):
    def setup(self):
        # Loads from local HF cache populated at Docker build time.
        self.model = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

    def predict(self, text: str = "Livepeer is great") -> dict:
        result = self.model(text)[0]
        return {
            "label": result["label"],
            "score": float(result["score"]),
            "text": text,
        }


if __name__ == "__main__":
    serve(SentimentAnalyzer())

"""LLM chat BYOC pipeline using HuggingFace transformers. Streams tokens via SSE."""

from threading import Thread
from typing import Iterator

from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from livepeer_gateway.runner import Pipeline, serve


class ChatInput(BaseModel):
    prompt: str = Field(description="User message")
    system: str = Field(default="You are a helpful assistant.", description="System prompt")
    max_tokens: int = Field(default=256, ge=1, le=1024, description="Max tokens to generate")


class ChatChunk(BaseModel):
    token: str


class ChatPipeline(Pipeline):
    def setup(self):
        # Loads from local HF cache populated at Docker build time.
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def run(self, params: ChatInput) -> Iterator[ChatChunk]:
        messages = [
            {"role": "system", "content": params.system},
            {"role": "user", "content": params.prompt},
        ]
        # Two-step (template → text → tokenize) avoids a BatchEncoding
        # wrapper that model.generate() can't unpack.
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        # TextIteratorStreamer feeds tokens into a thread-safe queue as the
        # model generates them. We drive .generate() in a background thread
        # so this generator can yield chunks as they arrive.
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        Thread(
            target=self.model.generate,
            kwargs={
                **inputs,
                "max_new_tokens": params.max_tokens,
                "streamer": streamer,
            },
        ).start()

        for chunk in streamer:
            if chunk:
                yield ChatChunk(token=chunk)


if __name__ == "__main__":
    serve(ChatPipeline())

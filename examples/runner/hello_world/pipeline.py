"""Hello-world BYOC pipeline. Run via ``docker compose up``."""

import json
import time
from typing import Iterator

import uvicorn
from fastapi import Request
from fastapi.responses import StreamingResponse

from livepeer_gateway.runner import Pipeline, make_app


class HelloWorld(Pipeline):
    def predict(self, name: str = "world") -> dict:
        return {"message": f"hello, {name}"}


def _sse_hello(name: str) -> Iterator[str]:
    yield "event: message\n"
    yield f"data: {json.dumps({'message': f'hello, {name}'})}\n\n"
    time.sleep(0.1)
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    app = make_app(HelloWorld())

    @app.post("/predict-sse", summary="Run one streaming inference")
    async def predict_sse(request: Request) -> StreamingResponse:
        body = await request.json()
        name = body.get("name", "world") if isinstance(body, dict) else "world"
        return StreamingResponse(_sse_hello(str(name)), media_type="text/event-stream")

    uvicorn.run(app, host="0.0.0.0", port=5000)

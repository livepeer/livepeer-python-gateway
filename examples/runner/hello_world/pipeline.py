"""Hello-world BYOC pipeline. Run via ``docker compose up``."""

import logging

from livepeer_gateway.runner import Pipeline, serve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


class HelloWorld(Pipeline):
    def run(self, name: str = "world") -> dict:
        return {"message": f"hello, {name}"}


if __name__ == "__main__":
    serve(HelloWorld())

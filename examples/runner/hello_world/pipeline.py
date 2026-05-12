"""Hello-world BYOC pipeline. Run via ``docker compose up``."""

from livepeer_gateway.runner import Pipeline, serve


class HelloWorld(Pipeline):
    def run(self, name: str = "world") -> dict:
        return {"message": f"hello, {name}"}


if __name__ == "__main__":
    serve(HelloWorld())

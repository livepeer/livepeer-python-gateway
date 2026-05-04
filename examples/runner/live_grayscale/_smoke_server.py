"""Trickle-aware fixture server for test.sh's real-data smoke.

Serves _fixtures/0 with `Lp-Trickle-Seq: 0` for /-2/-1/0 (matching
TrickleSubscriber's `start_seq=-2`). 404 elsewhere — natural EOS.
"""

import http.server
import os

_FIXTURE = open(os.path.join(os.path.dirname(__file__), "_fixtures", "0"), "rb").read()


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/-2", "/-1", "/0"):
            self.send_response(200)
            self.send_header("Content-Type", "video/MP2T")
            self.send_header("Content-Length", str(len(_FIXTURE)))
            self.send_header("Lp-Trickle-Seq", "0")
            self.end_headers()
            self.wfile.write(_FIXTURE)
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    http.server.HTTPServer(("0.0.0.0", 8080), _Handler).serve_forever()

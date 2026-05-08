"""Pretty-print BYOC data-channel SSE records, one per line.

Reads JSON-shaped data lines from stdin (one record per line, as emitted
by `pipeline.emit_data` and proxied through the gateway's SSE endpoint)
and renders each one for human-readable terminal output. Used by demo.sh.
"""

# TODO: see README — migration to the Python client SDK.

from __future__ import annotations

import json
import sys


def main() -> None:
    for line in sys.stdin:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = rec.get("type")
        if kind == "transcript":
            print("[{:3d}] {}".format(rec["index"], rec["text"]), flush=True)
        elif kind in ("ready", "stopped"):
            print("  ({})".format(kind), flush=True)


if __name__ == "__main__":
    main()

"""Register the image-upscale capability with the orchestrator.

Wire format follows
https://github.com/livepeer/go-livepeer/blob/main/doc/byoc.md
(handler in ``byoc/job_orchestrator.go``).
"""

import os
import sys
import time

import requests
import urllib3

# Orchestrator's HTTPS endpoint uses a self-signed cert.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ORCH_URL = os.environ.get("ORCH_URL", "https://orchestrator:8935")
ORCH_SECRET = os.environ.get("ORCH_SECRET", "orch-secret")
CAPABILITY_NAME = os.environ.get("CAPABILITY_NAME", "image-upscale")
CAPABILITY_URL = os.environ.get("CAPABILITY_URL", "http://image_upscale:5000")
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "30"))

data = {
    "name": CAPABILITY_NAME,
    "url": CAPABILITY_URL,
    "capacity": 1,
    "price_per_unit": 0,
    "price_scaling": 1,
    "currency": "wei",
}
headers = {"Authorization": ORCH_SECRET}

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        r = requests.post(
            f"{ORCH_URL}/capability/register",
            json=data,
            headers=headers,
            verify=False,
            timeout=5,
        )
        if r.status_code == 200:
            print(f"registered {CAPABILITY_NAME} -> {CAPABILITY_URL}")
            sys.exit(0)
        print(f"attempt {attempt}: status={r.status_code} body={r.text!r}")
    except Exception as exc:
        print(f"attempt {attempt}: {exc}")
    time.sleep(2)

print("registration failed after timeout")
sys.exit(1)

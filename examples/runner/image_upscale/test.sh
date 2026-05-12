#!/usr/bin/env bash
# E2E: send a request through the gateway, assert the upscaled image
# (2x of the 32x32 fixture = 64x64) comes back through the orchestrator.

# TODO: see README — migration to the Python client SDK.

set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"
TEST_IMAGE="${TEST_IMAGE:-test_image.png}"
INPUT_WIDTH="${INPUT_WIDTH:-64}"
INPUT_HEIGHT="${INPUT_HEIGHT:-64}"

echo "Waiting for capability registration..."
# SDK self-registers inside the pipeline container; look for the log line
# emitted by livepeer_gateway.runner.registration.register().
for _ in $(seq 30); do
    if docker logs image_upscale 2>&1 | grep -q "registered capability=image-upscale"; then
        echo "  registered."
        break
    fi
    sleep 1
done
if ! docker logs image_upscale 2>&1 | grep -q "registered capability=image-upscale"; then
    echo "FAIL: image_upscale container hasn't logged registration success." >&2
    echo "Make sure 'docker compose up -d --wait --build' completed first." >&2
    exit 1
fi

INPUT_B64=$(base64 -w0 < "${TEST_IMAGE}")
LIVEPEER_HDR=$(printf '%s' '{"request":"{}","parameters":"{}","capability":"image-upscale","timeout_seconds":60}' | base64 -w0)

echo "Sending request through gateway..."
RESPONSE=$(curl -fsS -X POST "${GATEWAY_URL}/process/request/run" \
    -H "Livepeer: ${LIVEPEER_HDR}" \
    -H "Content-Type: application/json" \
    -d "{\"image\":\"${INPUT_B64}\"}")

# Trim the base64 image from the echoed response — keeps stdout readable.
echo "Response (image truncated): $(echo "${RESPONSE}" | sed 's/\("image":"\)[^"]*/\1<base64>/')"

WIDTH=$(echo "${RESPONSE}" | grep -oE '"width"[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+$')
HEIGHT=$(echo "${RESPONSE}" | grep -oE '"height"[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+$')

# The Swin2SR processor pads inputs to its window size before upscaling, so
# output is at least 2x input but may be slightly larger.
if [ "${WIDTH}" -ge $((INPUT_WIDTH * 2)) ] && [ "${HEIGHT}" -ge $((INPUT_HEIGHT * 2)) ]; then
    echo "PASS (${WIDTH}x${HEIGHT}, >=2x of ${INPUT_WIDTH}x${INPUT_HEIGHT})"
    exit 0
fi

echo "FAIL: expected >=${INPUT_WIDTH}x${INPUT_HEIGHT} doubled, got ${WIDTH}x${HEIGHT}"
exit 1

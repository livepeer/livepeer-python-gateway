#!/usr/bin/env bash
# E2E test for the LivePipeline real-time runner.
# Pushes a synthetic colored stream through the BYOC stack and asserts
# the runner produces non-empty output bytes.
#
# Path: ffmpeg push (RTMP) → mediamtx → gateway → orch → runner → orch → mediamtx → ffmpeg pull (RTMP)
#
# TODO: post-PR-#6, replace the gateway+mediamtx ingestion with
# `start_byoc_job` from `livepeer_gateway.byoc` (same chroma assertion,
# customer-flow E2E without the deprecated gateway endpoints).

set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"
OUTPUT_FILE="${OUTPUT_FILE:-/tmp/live_grayscale_output.ts}"

echo "Waiting for capability registration..."
for _ in $(seq 1 60); do
    if docker logs register_capability 2>&1 | grep -q "registered live-video-to-video"; then
        echo "  registered."
        break
    fi
    sleep 2
done

# `parameters` is a stringified JSON; enable_video_{ingress,egress}
# drive trickle channel creation (go-livepeer byoc/types.go).
LIVEPEER_HDR=$(printf '%s' \
  '{"request":"{}","parameters":"{\"enable_video_ingress\":true,\"enable_video_egress\":true}","capability":"live-video-to-video","timeout_seconds":60}' \
  | base64 -w0)

echo "Starting stream session..."
RESPONSE=$(curl -fsS -X POST "${GATEWAY_URL}/process/stream/start" \
    -H "Livepeer: ${LIVEPEER_HDR}" -d '{}')

STREAM_ID=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin)['stream_id'])")
RTMP_IN=$(echo "${RESPONSE}"  | python3 -c "import json,sys; print(json.load(sys.stdin)['rtmp_url'])")
RTMP_OUT=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin)['rtmp_output_url'].split(',')[0])")

echo "  stream_id=${STREAM_ID}"
echo "  rtmp_in =${RTMP_IN}"
echo "  rtmp_out=${RTMP_OUT}"

trap 'curl -fsS -X POST "${GATEWAY_URL}/process/stream/${STREAM_ID}/stop" -H "Livepeer: ${LIVEPEER_HDR}" -d "{}" >/dev/null 2>&1 || true; rm -f "${OUTPUT_FILE}"' EXIT

echo "Pushing synthetic colored stream..."
ffmpeg -loglevel error -re \
       -f lavfi -i "testsrc=size=320x240:rate=30:duration=5" \
       -c:v libx264 -preset ultrafast -tune zerolatency \
       -f flv "${RTMP_IN}" &

echo "Pulling processed stream..."
ffmpeg -loglevel error -i "${RTMP_OUT}" -t 5 -c:v copy "${OUTPUT_FILE}"

if [ ! -s "${OUTPUT_FILE}" ]; then
    echo "FAIL: no bytes received from gateway egress"
    exit 1
fi

OUTPUT_SIZE=$(stat -c %s "${OUTPUT_FILE}" 2>/dev/null || stat -f %z "${OUTPUT_FILE}")
echo "PASS (${OUTPUT_SIZE} bytes received)"

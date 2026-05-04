#!/usr/bin/env bash
# E2E lifecycle smoke test for the LivePipeline real-time runner.
#
# Asserts that a stream session establishes through the full BYOC
# stack: caller (curl) → gateway → orchestrator → runner /stream/start,
# and tears down cleanly via /process/stream/{id}/stop.
#
# Does NOT push media. Pushing ffmpeg-generated MP2T and asserting
# grayscale output is a follow-up (see TODO at bottom).

set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"

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
    -H "Livepeer: ${LIVEPEER_HDR}" \
    -H "Content-Type: application/json" \
    -d '{}')

echo "Response: ${RESPONSE}"

STREAM_ID=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin)['stream_id'])")
WHIP_URL=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('whip_url',''))")
WHEP_URL=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('whep_url',''))")

echo "  stream_id=${STREAM_ID}"
echo "  whip=${WHIP_URL}"
echo "  whep=${WHEP_URL}"

if [ -z "${STREAM_ID}" ] || [ -z "${WHIP_URL}" ] || [ -z "${WHEP_URL}" ]; then
    echo "FAIL: missing stream URLs in response"
    exit 1
fi

# Give the runner a moment to receive the orchestrator's /stream/start.
sleep 1

echo "Asserting runner received /stream/start..."
# uvicorn access log format: 'POST /stream/start HTTP/1.1" 200 OK'.
# We assert 200 specifically so a 422 / 4xx still fails the test.
if ! docker logs live_grayscale 2>&1 | grep -qE 'POST /stream/start HTTP/1\.1" 200'; then
    echo "FAIL: runner didn't accept /stream/start (200 not seen)"
    docker logs live_grayscale 2>&1 | tail -20
    exit 1
fi

echo "Stopping stream..."
curl -fsS -X POST "${GATEWAY_URL}/process/stream/${STREAM_ID}/stop" \
    -H "Livepeer: ${LIVEPEER_HDR}" \
    -d '{}'

sleep 1

echo "Asserting runner received /stream/stop..."
if ! docker logs live_grayscale 2>&1 | grep -qE 'POST /stream/stop HTTP/1\.1" 200'; then
    echo "FAIL: runner didn't accept /stream/stop (200 not seen)"
    docker logs live_grayscale 2>&1 | tail -20
    exit 1
fi

echo "PASS"
exit 0

# TODO: extend with ffmpeg-driven media verification — push MP2T to WHIP,
# pull from WHEP, assert UV chroma planes are flat (= grayscale). Spike-
# risk; landing the lifecycle smoke test first proves the wire and lets
# us iterate on media verification independently.

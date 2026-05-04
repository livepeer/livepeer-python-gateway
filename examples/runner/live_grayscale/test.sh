#!/usr/bin/env bash
# E2E smoke test for the LivePipeline real-time runner. Two parts:
#   - Lifecycle through the BYOC stack (gateway → orch → runner).
#   - Real-data feed direct to runner with a pre-created MP2T segment.
#
# Grayscale correctness is verified by the demo.py script.

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
# Match status 200 explicitly — naive grep would pass on 4xx too.
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

# Phase 2: real-data smoke — feed a pre-created MP2T segment direct to the
# runner. Verifies bytes were fetched + no frame-processor errors.

# Trickle-aware fixture server on :8080 (handles start_seq=-2 + Lp-Trickle-Seq).
# Runner reaches the host via host.docker.internal.
SERVER_LOG=$(mktemp)
python3 -u _smoke_server.py > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap "kill ${SERVER_PID} 2>/dev/null; rm -f ${SERVER_LOG}" EXIT
sleep 0.5

echo "Starting real-data session against host fixtures server..."
curl -fsS -X POST -H "Content-Type: application/json" \
    -d '{"gateway_request_id":"smoke","control_url":"http://nope/c","events_url":"http://nope/e","subscribe_url":"http://host.docker.internal:8080","publish_url":"http://nope/p","params":null}' \
    http://localhost:5000/stream/start
echo

# Give the runner time to GET /-2 (start_seq=-2), decode, run process_video,
# and hit /1=404 = EOS.
sleep 3

echo "Asserting fixtures server received GET /-2..."
if ! grep -qE '"GET /-2 HTTP/1\.1" 200' "${SERVER_LOG}"; then
    echo "FAIL: fixtures server didn't serve GET /-2 — runner didn't fetch the segment"
    cat "${SERVER_LOG}"
    exit 1
fi

echo "Asserting no frame-processor errors in runner logs..."
if docker logs live_grayscale 2>&1 | grep -qE "process_video failed|process_audio failed"; then
    echo "FAIL: runner reported a frame-processor failure"
    docker logs live_grayscale 2>&1 | grep -E "process_video failed|process_audio failed" | tail -5
    exit 1
fi

echo "Stopping real-data session..."
curl -fsS -X POST http://localhost:5000/stream/stop
echo

echo "PASS"
exit 0

#!/usr/bin/env bash
# E2E test for the LivePipeline real-time runner. Pushes a synthetic
# colored stream through the full BYOC stack, asserts non-empty output,
# and opens the captured grayscale clip in a video player (skip with
# SKIP_VIEWER=1 in CI).
#
# Path: ffmpeg push (RTMP) → mediamtx → gateway → orch → runner → orch → mediamtx → ffmpeg pull (RTMP)
#
# TODO: post-PR-#6, replace the gateway+mediamtx ingestion with
# `start_byoc_job` from `livepeer_gateway.byoc` (customer-flow E2E
# without the deprecated gateway endpoints).

set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"
OUTPUT_FILE="${OUTPUT_FILE:-/tmp/live_grayscale_output.mts}"

echo "Waiting for capability registration..."
if ! docker logs register_capability 2>&1 | grep -q "registered live-video-to-video"; then
    echo "FAIL: register_capability hasn't logged success."
    echo "Make sure 'docker compose up -d --wait --build' completed first."
    exit 1
fi
echo "  registered."

# `parameters` is a stringified JSON; enable_video_{ingress,egress} drive
# trickle channel creation (go-livepeer byoc/types.go).
LIVEPEER_HDR=$(printf '%s' \
  '{"request":"{}","parameters":"{\"enable_video_ingress\":true,\"enable_video_egress\":true}","capability":"live-video-to-video","timeout_seconds":60}' \
  | base64 -w0)

# Best-effort session cleanup; registered early to catch Ctrl-C.
# `${STREAM_ID:-}` so an early failure (before stream/start succeeded) doesn't
# trip `set -u` when the trap fires.
trap 'curl -fsS -X POST "${GATEWAY_URL}/process/stream/${STREAM_ID:-}/stop" -H "Livepeer: ${LIVEPEER_HDR}" -d "{}" >/dev/null 2>&1 || true' EXIT

echo "Starting stream session..."
RESPONSE=$(curl -fsS -X POST "${GATEWAY_URL}/process/stream/start" \
    -H "Livepeer: ${LIVEPEER_HDR}" -d '{}')

STREAM_ID=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin)['stream_id'])")
RTMP_IN=$(echo  "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin)['rtmp_url'])")
RTMP_OUT=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin)['rtmp_output_url'].split(',')[0])")
echo "  stream_id=${STREAM_ID}"
echo "  rtmp_in =${RTMP_IN}"
echo "  rtmp_out=${RTMP_OUT}"

# Gateway URLs use docker-DNS name `mediamtx`, only resolvable inside compose.
RTMP_IN="${RTMP_IN/mediamtx:/localhost:}"
RTMP_OUT="${RTMP_OUT/mediamtx:/localhost:}"

# Push synthetic testsrc into the gateway. `-g 30` = 1s GOP (else first
# segment lags 8s+); duration outlasts cold-start; `</dev/null` stops
# ffmpeg fighting the script's stdin and "hanging" until Enter.
echo "Pushing synthetic colored stream..."
ffmpeg -loglevel error -re \
       -f lavfi -i "testsrc=size=320x240:rate=30:duration=60" \
       -c:v libx264 -preset ultrafast -tune zerolatency -g 30 \
       -f flv "${RTMP_IN}" </dev/null 2>/dev/null &
PUSH_PID=$!

# Retry until mediamtx serves egress; low-latency flags = faster first packet.
echo -n "Pulling processed stream"
PULL_OK=0
for _ in $(seq 1 "${RETRIES:-20}"); do
    if ffmpeg -y -loglevel error \
              -fflags nobuffer -flags low_delay \
              -probesize 32 -analyzeduration 0 \
              -i "${RTMP_OUT}" -t 5 -c:v copy "${OUTPUT_FILE}" </dev/null 2>/dev/null; then
        echo " ok."
        PULL_OK=1
        break
    fi
    echo -n "."
    sleep 2
done

# SIGINT (not SIGTERM/SIGKILL) lets ffmpeg flush its RTMP trailer cleanly.
kill -INT ${PUSH_PID} 2>/dev/null || true
wait ${PUSH_PID} 2>/dev/null || true

if [ ${PULL_OK} -ne 1 ]; then
    echo
    echo "FAIL: pull never succeeded after retries"
    exit 1
fi

if [ ! -s "${OUTPUT_FILE}" ]; then
    echo "FAIL: no bytes received from gateway egress"
    exit 1
fi

# Verify chroma ≈ 128 (else a no-op passthrough would also pass).
IFS=, read -r U V < <(ffprobe -v error -f lavfi -i "movie=${OUTPUT_FILE},signalstats" \
    -show_entries frame_tags=lavfi.signalstats.UAVG,lavfi.signalstats.VAVG \
    -of csv=p=0 -read_intervals "%+#1")
awk -v u="${U:-0}" -v v="${V:-0}" 'BEGIN { exit !(u>123 && u<133 && v>123 && v<133) }' \
    || { echo "FAIL: chroma not zeroed (U=${U} V=${V}, expected ≈128)"; exit 1; }

OUTPUT_SIZE=$(stat -c %s "${OUTPUT_FILE}" 2>/dev/null || stat -f %z "${OUTPUT_FILE}")
echo "PASS (${OUTPUT_SIZE} bytes, U=${U} V=${V} → ${OUTPUT_FILE})"

# Open in ffplay (skipped in non-TTY or via SKIP_VIEWER=1).
if [ -t 1 ] && [ "${SKIP_VIEWER:-0}" != "1" ]; then
    echo "Opening captured clip..."
    ffplay -loglevel error -window_title "live_grayscale (captured)" \
           -x 640 -y 480 -autoexit "${OUTPUT_FILE}"
fi

#!/usr/bin/env bash
# E2E test for the LivePipeline real-time runner. One stream session,
# two captures: first asserts the default tint (chroma ≈ 128 → grayscale),
# then POSTs new params via /process/stream/{id}/update and asserts the
# second capture reflects them. This exercises both `process_video` and
# `on_params_update` end-to-end through the BYOC stack. Skip the viewer
# with SKIP_VIEWER=1 in CI.
#
# Path: ffmpeg push (RTMP) → mediamtx → gateway → orch → runner → orch → mediamtx → ffmpeg pull (RTMP)
#
# TODO: post-PR-#6, replace the gateway+mediamtx ingestion with
# `start_byoc_job` from `livepeer_gateway.byoc` (customer-flow E2E
# without the deprecated gateway endpoints).

# TODO: see README — migration to the Python client SDK.

set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"
OUTPUT_FILE="${OUTPUT_FILE:-/tmp/live_tint_output.mts}"
OUTPUT_FILE_GRAY="${OUTPUT_FILE%.mts}_gray.mts"
OUTPUT_FILE_TINT="${OUTPUT_FILE%.mts}_tint.mts"

# Tint we POST after the first capture. Picked far enough from 128 that the
# ±5 tolerance below can't accidentally accept a stale grayscale capture.
TINT_U=80
TINT_V=180

echo "Waiting for capability registration..."
if ! docker logs register_capability 2>&1 | grep -q "registered live-tint"; then
    echo "FAIL: register_capability hasn't logged success."
    echo "Make sure 'docker compose up -d --wait --build' completed first."
    exit 1
fi
echo "  registered."

# `parameters` is a stringified JSON; enable_video_{ingress,egress} drive
# trickle channel creation (go-livepeer byoc/types.go).
LIVEPEER_HDR=$(printf '%s' \
  '{"request":"{}","parameters":"{\"enable_video_ingress\":true,\"enable_video_egress\":true}","capability":"live-tint","timeout_seconds":120}' \
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
# segment lags 8s+); duration outlasts both captures + the post-update
# settle window. `</dev/null` stops ffmpeg fighting the script's stdin.
echo "Pushing synthetic colored stream..."
ffmpeg -loglevel error -re \
       -f lavfi -i "testsrc=size=320x240:rate=30:duration=60" \
       -c:v libx264 -preset ultrafast -tune zerolatency -g 30 \
       -f flv "${RTMP_IN}" </dev/null 2>/dev/null &
PUSH_PID=$!

# Capture #1 — default tint (grayscale). Retried until mediamtx serves egress;
# low-latency flags = faster first packet.
echo -n "Capture #1 (default tint)"
PULL_OK=0
for _ in $(seq 1 "${RETRIES:-20}"); do
    if ffmpeg -y -loglevel error \
              -fflags nobuffer -flags low_delay \
              -probesize 32 -analyzeduration 0 \
              -i "${RTMP_OUT}" -t 4 -c:v copy "${OUTPUT_FILE_GRAY}" </dev/null 2>/dev/null; then
        echo " ok."
        PULL_OK=1
        break
    fi
    echo -n "."
    sleep 2
done

if [ ${PULL_OK} -ne 1 ]; then
    kill -INT ${PUSH_PID} 2>/dev/null || true
    wait ${PUSH_PID} 2>/dev/null || true
    echo
    echo "FAIL: pull never succeeded after retries"
    exit 1
fi

# POST the new tint and give the runner a moment for the next encoded
# segment to land on the egress channel. 2s @ 1s GOP = at least one fresh
# I-frame after the update arrives.
echo "Posting tint update u=${TINT_U} v=${TINT_V}..."
curl -fsS -X POST "${GATEWAY_URL}/process/stream/${STREAM_ID}/update" \
    -H "Livepeer: ${LIVEPEER_HDR}" \
    -H "Content-Type: application/json" \
    -d "{\"u\": ${TINT_U}, \"v\": ${TINT_V}}" >/dev/null
sleep 2

# Capture #2 — same egress stream, new tint should be live.
echo -n "Capture #2 (after update)"
ffmpeg -y -loglevel error \
       -fflags nobuffer -flags low_delay \
       -probesize 32 -analyzeduration 0 \
       -i "${RTMP_OUT}" -t 4 -c:v copy "${OUTPUT_FILE_TINT}" </dev/null 2>/dev/null
echo " ok."

# SIGINT (not SIGTERM/SIGKILL) lets ffmpeg flush its RTMP trailer cleanly.
kill -INT ${PUSH_PID} 2>/dev/null || true
wait ${PUSH_PID} 2>/dev/null || true

if [ ! -s "${OUTPUT_FILE_GRAY}" ] || [ ! -s "${OUTPUT_FILE_TINT}" ]; then
    echo "FAIL: empty capture(s) (gray=$(stat -c %s "${OUTPUT_FILE_GRAY}" 2>/dev/null || stat -f %z "${OUTPUT_FILE_GRAY}"), tint=$(stat -c %s "${OUTPUT_FILE_TINT}" 2>/dev/null || stat -f %z "${OUTPUT_FILE_TINT}"))"
    exit 1
fi

# Read mean U/V from a single decoded frame in each capture. signalstats
# emits per-frame averages; we only need one good frame for an assertion.
read_uv() {
    ffprobe -v error -f lavfi -i "movie=$1,signalstats" \
        -show_entries frame_tags=lavfi.signalstats.UAVG,lavfi.signalstats.VAVG \
        -of csv=p=0 -read_intervals "%+#1"
}

IFS=, read -r U_GRAY V_GRAY < <(read_uv "${OUTPUT_FILE_GRAY}")
IFS=, read -r U_TINT V_TINT < <(read_uv "${OUTPUT_FILE_TINT}")

# Default capture: chroma must be ≈128 (else a no-op passthrough would also pass).
awk -v u="${U_GRAY:-0}" -v v="${V_GRAY:-0}" 'BEGIN { exit !(u>123 && u<133 && v>123 && v<133) }' \
    || { echo "FAIL: default tint not neutral (U=${U_GRAY} V=${V_GRAY}, expected ≈128)"; exit 1; }

# Post-update capture: chroma must track the new (u,v) within ±5 (round-trip
# encode/decode jitter). The 80/180 picks are >40 away from 128, so this
# can't be confused with the grayscale assertion.
awk -v u="${U_TINT:-0}" -v v="${V_TINT:-0}" -v want_u="${TINT_U}" -v want_v="${TINT_V}" \
    'BEGIN { exit !(u>want_u-5 && u<want_u+5 && v>want_v-5 && v<want_v+5) }' \
    || { echo "FAIL: tint update not reflected (U=${U_TINT} V=${V_TINT}, expected ≈${TINT_U}/${TINT_V})"; exit 1; }

# Concatenate the two clips for the optional viewer step (gray → tint).
ffmpeg -y -loglevel error -i "concat:${OUTPUT_FILE_GRAY}|${OUTPUT_FILE_TINT}" \
       -c copy "${OUTPUT_FILE}" </dev/null

OUTPUT_SIZE=$(stat -c %s "${OUTPUT_FILE}" 2>/dev/null || stat -f %z "${OUTPUT_FILE}")
echo "PASS (${OUTPUT_SIZE} bytes; gray U=${U_GRAY} V=${V_GRAY} → tint U=${U_TINT} V=${V_TINT} → ${OUTPUT_FILE})"

# Open in ffplay (skipped in non-TTY or via SKIP_VIEWER=1).
if [ -t 1 ] && [ "${SKIP_VIEWER:-0}" != "1" ]; then
    echo "Opening captured clip (gray → tint)..."
    ffplay -loglevel error -window_title "live_tint (captured)" \
           -x 640 -y 480 -autoexit "${OUTPUT_FILE}"
fi

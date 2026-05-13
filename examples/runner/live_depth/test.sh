#!/usr/bin/env bash
# E2E test for live_depth — pushes a synthetic colored stream through the
# full BYOC stack, asserts the egress is grayscale (chroma zeroed) AND that
# the luma plane has spatial variance (i.e., the depth model produced a
# non-uniform output rather than a fixed grayscale fill).
#
# Path: ffmpeg push (RTMP) → mediamtx → gateway → orch → runner → orch → mediamtx → ffmpeg pull (RTMP)

# TODO: see README — migration to the Python client SDK.

set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"
OUTPUT_FILE="${OUTPUT_FILE:-/tmp/live_depth_output.mts}"

echo "Waiting for capability registration..."
# SDK self-registers inside the pipeline container; look for the log line
# emitted by livepeer_gateway.runner.registration.register().
# TODO: switch to `curl /status` once the SDK exposes a status endpoint
# (Phase 2 of auto-registration). Structured check beats log grep.
for _ in $(seq 30); do
    if docker logs live_depth 2>&1 | grep -q "registered capability=live-depth"; then
        echo "  registered."
        break
    fi
    sleep 1
done
if ! docker logs live_depth 2>&1 | grep -q "registered capability=live-depth"; then
    echo "FAIL: live_depth container hasn't logged registration success." >&2
    echo "Make sure 'docker compose up -d --wait --build' completed first." >&2
    exit 1
fi

# `parameters` is a stringified JSON; enable_video_{ingress,egress} drive
# trickle channel creation (go-livepeer byoc/types.go).
LIVEPEER_HDR=$(printf '%s' \
  '{"request":"{}","parameters":"{\"enable_video_ingress\":true,\"enable_video_egress\":true}","capability":"live-depth","timeout_seconds":60}' \
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

RTMP_IN="${RTMP_IN/mediamtx:/localhost:}"
RTMP_OUT="${RTMP_OUT/mediamtx:/localhost:}"

# Depth-rich clip from the DepthAnything V2 repo (cached under assets/ on
# first run).
SAMPLE_URL="${SAMPLE_URL:-https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/assets/examples_video/basketball.mp4}"
SAMPLE_FILE="${SAMPLE_FILE:-assets/sample.mp4}"
if [ ! -f "${SAMPLE_FILE}" ]; then
    echo "Downloading sample video..."
    mkdir -p "$(dirname "${SAMPLE_FILE}")"
    curl -fsSL -o "${SAMPLE_FILE}" "${SAMPLE_URL}"
fi

# `-stream_loop -1` loops the clip until the `-t` budget is hit.
echo "Pushing depth-rich sample..."
ffmpeg -loglevel error -re -stream_loop -1 -i "${SAMPLE_FILE}" \
       -vf "scale=640:480" -an \
       -c:v libx264 -preset ultrafast -tune zerolatency -g 30 \
       -t 60 -f flv "${RTMP_IN}" </dev/null 2>/dev/null &
PUSH_PID=$!

# Re-encode (not `-c:v copy`) so the capture starts with fresh SPS/PPS.
# A mid-stream RTMP pull lands after the originals, leaving a copied stream undecodable.
echo -n "Pulling processed stream"
PULL_OK=0
for _ in $(seq 1 "${RETRIES:-20}"); do
    if ffmpeg -y -loglevel error \
              -i "${RTMP_OUT}" -t 5 \
              -c:v libx264 -preset ultrafast -an \
              "${OUTPUT_FILE}" </dev/null 2>/dev/null; then
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

# Validate the depth pipeline actually ran:
#   1. Extract a PNG frame (direct ffprobe on the .mts trips on mid-stream H.264 issues).
#   2. Read luma min/max + chroma averages via signalstats.
#      ffprobe emits CSV in Y*→U*→V* order regardless of -show_entries; match `read` to that.
#   3. Assert U ≈ V ≈ 128 (chroma was zeroed).
#   4. Assert YHIGH - YLOW > 50 (luma non-uniform — real depth, else a chroma-only no-op
#      would pass since real depth maps span ~150+ while uniform fill spans 0).
FRAME_PNG="${OUTPUT_FILE%.mts}_frame.png"
ffmpeg -v error -i "${OUTPUT_FILE}" -frames:v 1 -y "${FRAME_PNG}" </dev/null 2>/dev/null \
    || { echo "FAIL: could not extract a frame from ${OUTPUT_FILE}"; exit 1; }
IFS=, read -r YLOW YHIGH U V < <(ffprobe -v error -f lavfi \
    -i "movie=${FRAME_PNG},signalstats" \
    -show_entries frame_tags=lavfi.signalstats.YLOW,lavfi.signalstats.YHIGH,lavfi.signalstats.UAVG,lavfi.signalstats.VAVG \
    -of csv=p=0)
awk -v u="${U:-0}" -v v="${V:-0}" 'BEGIN { exit !(u>123 && u<133 && v>123 && v<133) }' \
    || { echo "FAIL: chroma not zeroed (U=${U} V=${V}, expected ≈128)"; exit 1; }
RANGE=$(awk -v hi="${YHIGH:-0}" -v lo="${YLOW:-0}" 'BEGIN { print hi - lo }')
awk -v r="${RANGE:-0}" 'BEGIN { exit !(r > 50.0) }' \
    || { echo "FAIL: luma plane too uniform (YHIGH-YLOW=${RANGE}); depth model may not be producing varied output"; exit 1; }

OUTPUT_SIZE=$(stat -c %s "${OUTPUT_FILE}" 2>/dev/null || stat -f %z "${OUTPUT_FILE}")
echo "PASS (${OUTPUT_SIZE} bytes, U=${U} V=${V} YHIGH-YLOW=${RANGE} → ${OUTPUT_FILE})"

# Open in ffplay (skipped in non-TTY or via SKIP_VIEWER=1).
if [ -t 1 ] && [ "${SKIP_VIEWER:-0}" != "1" ]; then
    echo "Opening captured clip..."
    ffplay -loglevel error -window_title "live_depth (captured)" \
           -x 640 -y 480 -autoexit "${OUTPUT_FILE}"
fi

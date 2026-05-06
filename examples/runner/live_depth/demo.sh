#!/usr/bin/env bash
# Live demo — push your webcam through the grayscale pipeline and watch
# the result in ffplay. Minimal complement to test.sh (which uses synthetic
# testsrc + capture-to-file for CI assertions).
#
# Path: webcam → ffmpeg push (RTMP) → mediamtx → gateway → orch → runner →
#       orch → mediamtx → ffplay pull (RTMP)
#
# Prerequisites:
#   - Stack up: `docker compose up -d --wait --build`
#   - ffmpeg + ffplay on host
#   - Webcam: /dev/video0 (Linux) or first AVFoundation device (macOS)

set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"

# 640x480 @ 15fps — sustains real-time on any 30/40-series GPU.
WEBCAM_RES="${WEBCAM_RES:-640x480}"
WEBCAM_FPS="${WEBCAM_FPS:-15}"

case "$(uname -s)" in
    Linux*)
        WEBCAM_DEVICE="${WEBCAM_DEVICE:-/dev/video0}"
        # `-thread_queue_size` absorbs v4l2 driver hiccups; `+discardcorrupt`
        # drops bad-buffer frames USB webcams emit at startup.
        INPUT_FLAGS=(-thread_queue_size 1024 -fflags +discardcorrupt
                     -f v4l2 -framerate "${WEBCAM_FPS}" -video_size "${WEBCAM_RES}"
                     -i "${WEBCAM_DEVICE}")
        ;;
    Darwin*)
        # `0` = first AVFoundation video device. Use `ffmpeg -f avfoundation -list_devices true -i ""` to enumerate.
        WEBCAM_DEVICE="${WEBCAM_DEVICE:-0}"
        INPUT_FLAGS=(-thread_queue_size 1024 -fflags +discardcorrupt
                     -f avfoundation -framerate "${WEBCAM_FPS}" -video_size "${WEBCAM_RES}"
                     -i "${WEBCAM_DEVICE}")
        ;;
    *)
        echo "Unsupported platform $(uname -s); only Linux and macOS are wired up." >&2
        exit 1
        ;;
esac

echo "Waiting for capability registration..."
if ! docker logs register_capability 2>&1 | grep -q "registered live-video-to-video"; then
    echo "FAIL: register_capability hasn't logged success."
    echo "Make sure 'docker compose up -d --wait --build' completed first."
    exit 1
fi
echo "  registered."

# `parameters` is a stringified JSON; enable_video_{ingress,egress} drive
# trickle channel creation (go-livepeer byoc/types.go). 600s timeout for long demos.
LIVEPEER_HDR=$(printf '%s' \
  '{"request":"{}","parameters":"{\"enable_video_ingress\":true,\"enable_video_egress\":true}","capability":"live-video-to-video","timeout_seconds":600}' \
  | base64 -w0)

# Best-effort session cleanup; registered early to catch Ctrl-C.
# `${STREAM_ID:-}` so an early failure (before stream/start succeeded) doesn't
# trip `set -u` when the trap fires.
trap 'kill -INT "${PUSH_PID:-}" 2>/dev/null || true; curl -fsS -X POST "${GATEWAY_URL}/process/stream/${STREAM_ID:-}/stop" -H "Livepeer: ${LIVEPEER_HDR}" -d "{}" >/dev/null 2>&1 || true' EXIT

echo "Starting stream session..."
RESPONSE=$(curl -fsS -X POST "${GATEWAY_URL}/process/stream/start" \
    -H "Livepeer: ${LIVEPEER_HDR}" -d '{}')

STREAM_ID=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin)['stream_id'])")
RTMP_IN=$(echo  "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin)['rtmp_url'])")
RTMP_OUT=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin)['rtmp_output_url'].split(',')[0])")

# Gateway URLs use docker-DNS name `mediamtx`, only resolvable inside compose.
RTMP_IN="${RTMP_IN/mediamtx:/localhost:}"
RTMP_OUT="${RTMP_OUT/mediamtx:/localhost:}"

echo "  stream_id=${STREAM_ID}"
echo "  webcam =${WEBCAM_DEVICE} @ ${WEBCAM_RES} ${WEBCAM_FPS}fps"
echo "  rtmp_in=${RTMP_IN}"
echo "  rtmp_out=${RTMP_OUT}"

echo "Pushing webcam to gateway..."
# `-g 30` = 1s GOP so first segment lands quickly (else 8s+ latency).
# `-vf format=yuv420p` normalizes YUYV/NV12 before x264.
# `-loglevel error` mutes the cosmetic v4l2 startup warning.
ffmpeg -loglevel error -re \
    "${INPUT_FLAGS[@]}" \
    -vf format=yuv420p \
    -c:v libx264 -preset ultrafast -tune zerolatency -g 30 \
    -f flv "${RTMP_IN}" </dev/null 2>/dev/null &
PUSH_PID=$!

# Retry until mediamtx serves egress; low-latency flags = faster first packet.
echo -n "Waiting for processed stream"
for _ in $(seq 1 "${RETRIES:-30}"); do
    # Normal probe — readiness needs reliable confirmation, not low latency.
    if ffmpeg -loglevel error -y \
              -i "${RTMP_OUT}" -t 0.1 -f null - </dev/null 2>/dev/null; then
        echo " ok."
        break
    fi
    echo -n "."
    sleep 1
done

echo "Opening live viewer (close the window or Ctrl-C to stop)..."
# `nobuffer` + `low_delay` for low playback latency; no `probesize 32` —
# too aggressive for RTMP (causes I/O error on open before ffplay locks).
ffplay -loglevel warning \
    -fflags nobuffer -flags low_delay \
    -window_title "live_depth (webcam)" \
    "${RTMP_OUT}"

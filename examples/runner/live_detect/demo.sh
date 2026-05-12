#!/usr/bin/env bash
# Live demo — push your webcam + microphone through the BYOC stack and watch
# both detection and transcript records arrive in real time. Detections fire
# every ~1s (one YOLO inference per second of video); transcripts fire every
# ~3s (one whisper window).
#
# Path:  webcam + mic → ffmpeg push (RTMP) → mediamtx → gateway → orch →
#                       runner → emit_data → orch trickle (data) → gateway
#                       SSE proxy → curl subscriber → pretty-printed records
#
# Prerequisites:
#   - Stack up: `docker compose up -d --wait --build`
#   - ffmpeg + curl + python3 on host
#   - Webcam + microphone:
#       Linux  — pulseaudio default mic + /dev/video0; override via
#                PULSE_SOURCE=... and VIDEO_DEVICE=...
#       macOS  — first AVFoundation video + audio device (`:0`); override
#                via MAC_VIDEO_DEVICE=... MAC_AUDIO_DEVICE=...

# TODO: see README — migration to the Python client SDK.

set -euo pipefail
cd "$(dirname "$0")"

# Use the system ffmpeg — capture demuxers (pulse/alsa/avfoundation/v4l2)
# must be compiled in. The transcoding-only ffmpeg that ships with go-
# livepeer lacks them, so a Livepeer dev's $PATH ffmpeg often won't open
# the webcam/mic.
FFMPEG="${FFMPEG:-${HOMEBREW_PREFIX:+$HOMEBREW_PREFIX/bin/ffmpeg}}"
[ -x "${FFMPEG:-}" ] || FFMPEG="/usr/bin/ffmpeg"
[ -x "${FFMPEG}" ] || FFMPEG="$(command -v ffmpeg)"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"
DURATION="${DURATION:-60}"

case "$(uname -s)" in
    Linux*)
        VIDEO_INPUT=(-f v4l2 -framerate 15 -video_size 640x480 -i "${VIDEO_DEVICE:-/dev/video0}")
        AUDIO_INPUT=(-f pulse -i "${PULSE_SOURCE:-default}")
        ;;
    Darwin*)
        # AVFoundation: <video>:<audio>. Enumerate via:
        # `ffmpeg -f avfoundation -list_devices true -i ""`.
        VIDEO_INPUT=(-f avfoundation -framerate 15 -video_size 640x480 \
                     -i "${MAC_VIDEO_DEVICE:-0}:${MAC_AUDIO_DEVICE:-0}")
        AUDIO_INPUT=()
        ;;
    *)
        echo "Unsupported platform $(uname -s); only Linux and macOS are wired up." >&2
        exit 1
        ;;
esac

echo "Waiting for capability registration..."
if ! docker logs register_capability 2>&1 | grep -q "registered live-detect"; then
    echo "FAIL: register_capability hasn't logged success."
    echo "Make sure 'docker compose up -d --wait --build' completed first."
    exit 1
fi
echo "  registered."

# enable_data_output makes the gateway create the data trickle channel and
# proxy it as SSE on /process/stream/{id}/data. Long timeout for a chatty demo.
LIVEPEER_HDR=$(printf '%s' \
  '{"request":"{}","parameters":"{\"enable_video_ingress\":true,\"enable_video_egress\":true,\"enable_data_output\":true}","capability":"live-detect","timeout_seconds":600}' \
  | base64 -w0)

# Best-effort cleanup; registered early to catch Ctrl-C.
trap 'kill -INT "${PUSH_PID:-}" 2>/dev/null || true; kill "${SUB_PID:-}" 2>/dev/null || true; curl -fsS -X POST "${GATEWAY_URL}/process/stream/${STREAM_ID:-}/stop" -H "Livepeer: ${LIVEPEER_HDR}" -d "{}" >/dev/null 2>&1 || true' EXIT

echo "Starting stream session..."
RESPONSE=$(curl -fsS -X POST "${GATEWAY_URL}/process/stream/start" \
    -H "Livepeer: ${LIVEPEER_HDR}" -d '{}')

STREAM_ID=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin)['stream_id'])")
RTMP_IN=$(echo  "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin)['rtmp_url'])")
RTMP_IN="${RTMP_IN/mediamtx:/localhost:}"
DATA_URL="${GATEWAY_URL}/process/stream/${STREAM_ID}/data"

echo "  stream_id=${STREAM_ID}"
echo "  rtmp_in =${RTMP_IN}"
echo "  data_url=${DATA_URL}"

# Subscribe before pushing so we don't miss early records.
echo
echo "Listening for detections + transcripts. Show things to the camera"
echo "and speak — Ctrl-C to stop early."
echo "------------------------------------------------------------"
curl -fsS -N --max-time "$((DURATION + 30))" "${DATA_URL}" 2>/dev/null \
  | grep --line-buffered '^data: {' \
  | sed -u 's/^data: //' \
  | python3 -u _format_records.py &
SUB_PID=$!

# Tiny gap so curl's connection is established before ffmpeg starts pushing.
sleep 1

# Push webcam video + mic audio. macOS combines video+audio in a single
# avfoundation input; Linux uses separate v4l2 + pulse inputs.
"${FFMPEG}" -loglevel error -re \
    "${VIDEO_INPUT[@]}" \
    "${AUDIO_INPUT[@]}" \
    -map 0:v -map "${#AUDIO_INPUT[@]:+1}:a" \
    -c:v libx264 -preset ultrafast -tune zerolatency -g 15 \
    -c:a aac -ar 48000 \
    -t "${DURATION}" -f flv "${RTMP_IN}" </dev/null 2>/dev/null &
PUSH_PID=$!

wait "${PUSH_PID}" 2>/dev/null || true

# Let on_stream_stop's flush propagate through the SSE proxy, then close out.
sleep 4
curl -fsS -X POST "${GATEWAY_URL}/process/stream/${STREAM_ID}/stop" \
    -H "Livepeer: ${LIVEPEER_HDR}" -d '{}' >/dev/null 2>&1 || true
wait "${SUB_PID}" 2>/dev/null || true
SUB_PID=""

echo "------------------------------------------------------------"
echo "Done."

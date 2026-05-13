#!/usr/bin/env bash
# Live demo — speak into your microphone and watch transcripts appear in
# the terminal as whisper finishes each 3-second window. Minimal complement
# to test.sh (which uses the canned JFK clip for CI assertions).
#
# Path:  mic → ffmpeg push (RTMP) → mediamtx → gateway → orch → runner →
#               emit_data → orch trickle (data) → gateway SSE proxy →
#               this script's curl subscriber → pretty-printed transcripts
#
# Prerequisites:
#   - Stack up: `docker compose up -d --wait --build`
#   - ffmpeg + curl + python3 on host
#   - Microphone:
#       Linux  — pulseaudio (default source); override via PULSE_SOURCE=...
#                or `pactl list short sources` to find a name
#       macOS  — first AVFoundation audio device (`:0`); override via
#                MAC_AUDIO_DEVICE=...

# TODO: see README — migration to the Python client SDK.

set -euo pipefail
cd "$(dirname "$0")"

# Use the system ffmpeg — capture demuxers (pulse/alsa/avfoundation) must
# be compiled in. The transcoding-only ffmpeg that ships with go-livepeer
# lacks them, so a Livepeer dev's $PATH ffmpeg often won't open the mic.
FFMPEG="${FFMPEG:-${HOMEBREW_PREFIX:+$HOMEBREW_PREFIX/bin/ffmpeg}}"
[ -x "${FFMPEG:-}" ] || FFMPEG="/usr/bin/ffmpeg"
[ -x "${FFMPEG}" ] || FFMPEG="$(command -v ffmpeg)"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"
DURATION="${DURATION:-60}"

case "$(uname -s)" in
    Linux*)
        AUDIO_INPUT=(-f pulse -i "${PULSE_SOURCE:-default}")
        ;;
    Darwin*)
        # `:0` = first AVFoundation audio device. Enumerate via:
        # `ffmpeg -f avfoundation -list_devices true -i ""`.
        AUDIO_INPUT=(-f avfoundation -i ":${MAC_AUDIO_DEVICE:-0}")
        ;;
    *)
        echo "Unsupported platform $(uname -s); only Linux and macOS are wired up." >&2
        exit 1
        ;;
esac

echo "Waiting for capability registration..."
# SDK self-registers inside the pipeline container; look for the log line
# emitted by livepeer_gateway.runner.registration.register().
# TODO: switch to `curl /status` once the SDK exposes a status endpoint
# (Phase 2 of auto-registration). Structured check beats log grep.
for _ in $(seq 30); do
    if docker logs live_transcribe 2>&1 | grep -q "registered capability=live-transcribe"; then
        echo "  registered."
        break
    fi
    sleep 1
done
if ! docker logs live_transcribe 2>&1 | grep -q "registered capability=live-transcribe"; then
    echo "FAIL: live_transcribe container hasn't logged registration success." >&2
    echo "Make sure 'docker compose up -d --wait --build' completed first." >&2
    exit 1
fi

# enable_data_output makes the gateway create the data trickle channel and
# proxy it as SSE on /process/stream/{id}/data. Long timeout for a chatty demo.
LIVEPEER_HDR=$(printf '%s' \
  '{"request":"{}","parameters":"{\"enable_video_ingress\":true,\"enable_video_egress\":true,\"enable_data_output\":true}","capability":"live-transcribe","timeout_seconds":600}' \
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
echo "Listening for transcripts. Speak now — Ctrl-C to stop early."
echo "------------------------------------------------------------"
curl -fsS -N --max-time "$((DURATION + 30))" "${DATA_URL}" 2>/dev/null \
  | grep --line-buffered '^data: {' \
  | sed -u 's/^data: //' \
  | python3 -u _format_records.py &
SUB_PID=$!

# Tiny gap so curl's connection is established before ffmpeg starts pushing.
sleep 1

# Push black video + mic. Black satisfies the live-transcribe capability;
# the pipeline only consumes the audio track. `-g 15` = 1s GOP for low first-
# segment latency.
"${FFMPEG}" -loglevel error -re \
    -f lavfi -i "color=size=320x240:rate=15" \
    "${AUDIO_INPUT[@]}" \
    -map 0:v -map 1:a \
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

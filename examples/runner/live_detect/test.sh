#!/usr/bin/env bash
# E2E test for live_detect — pushes OpenCV's `vtest.avi` plaza-pedestrian
# clip + JFK's "Ask not..." audio through the BYOC stack, subscribes to
# the gateway's data SSE proxy, and asserts both:
#   - detection records arrive containing "person"
#   - transcript records arrive containing "country"
# Validates that both `process_video` and `process_audio` hooks fire on the
# same pipeline, with two emit_data streams ({"type": "detection"} and
# {"type": "transcript"}) interleaving cleanly on the data channel.
#
# Path:  ffmpeg push (RTMP) → mediamtx → gateway → orch → runner → emit_data
#                                               ↑                              │
#         curl SSE subscribe ─── gateway data proxy ──────────────────────────┘
#                                  (GET /process/stream/{id}/data)

# TODO: see README — migration to the Python client SDK.

set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"

echo "Waiting for capability registration..."
if ! docker logs register_capability 2>&1 | grep -q "registered live-detect"; then
    echo "FAIL: register_capability hasn't logged success."
    echo "Make sure 'docker compose up -d --wait --build' completed first."
    exit 1
fi
echo "  registered."

LIVEPEER_HDR=$(printf '%s' \
  '{"request":"{}","parameters":"{\"enable_video_ingress\":true,\"enable_video_egress\":true,\"enable_data_output\":true}","capability":"live-detect","timeout_seconds":120}' \
  | base64 -w0)

trap 'kill ${SUB_PID:-} 2>/dev/null || true; curl -fsS -X POST "${GATEWAY_URL}/process/stream/${STREAM_ID:-}/stop" -H "Livepeer: ${LIVEPEER_HDR}" -d "{}" >/dev/null 2>&1 || true' EXIT

# Audio fixture: JFK "Ask not..." (~11s) — same source as live_transcribe.
AUDIO_URL="${AUDIO_URL:-https://github.com/openai/whisper/raw/main/tests/jfk.flac}"
AUDIO_FILE="${AUDIO_FILE:-assets/jfk.flac}"
# Video fixture: OpenCV's canonical `vtest.avi` — ~80s plaza-pedestrian
# scene (5–9 simultaneous people walking + a parked truck). Multi-class,
# real-world, COCO-trained YOLO scores `person` 0.7–0.9 every frame.
VIDEO_URL="${VIDEO_URL:-https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi}"
VIDEO_FILE="${VIDEO_FILE:-assets/vtest.avi}"

mkdir -p assets
[ -f "${AUDIO_FILE}" ] || { echo "Downloading audio fixture..."; curl -fsSL -o "${AUDIO_FILE}" "${AUDIO_URL}"; }
[ -f "${VIDEO_FILE}" ] || { echo "Downloading video fixture..."; curl -fsSL -o "${VIDEO_FILE}" "${VIDEO_URL}"; }

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

SSE_OUT=$(mktemp -t live_detect.XXXXXX.sse)
echo "Subscribing to data channel SSE..."
curl -fsS -N --max-time 30 "${DATA_URL}" > "${SSE_OUT}" 2>/dev/null &
SUB_PID=$!
sleep 1

# 15s push: vtest.avi (looped, normalized to 15 fps / 640×480) + JFK
# audio (also looped). Both inputs use `-stream_loop -1` so a short
# fixture covers the full push duration.
echo "Pushing video+audio (15s)..."
ffmpeg -loglevel error -re \
       -stream_loop -1 -i "${VIDEO_FILE}" \
       -stream_loop -1 -i "${AUDIO_FILE}" \
       -map 0:v -map 1:a \
       -c:v libx264 -preset ultrafast -tune zerolatency -g 15 \
       -vf "fps=15,scale=640:480" \
       -c:a aac -ar 48000 \
       -t 15 -f flv "${RTMP_IN}" </dev/null 2>/dev/null

# Let the final in-stream window + whisper inference (~3.5s) complete
# before /stream/stop fires. The gateway closes its SSE proxy on stop
# without draining, so we wait here, not after.
sleep 4
curl -fsS -X POST "${GATEWAY_URL}/process/stream/${STREAM_ID}/stop" \
    -H "Livepeer: ${LIVEPEER_HDR}" -d '{}' >/dev/null 2>&1 || true
# Buffer for the SSE subscriber to flush before we cancel the curl reader.
sleep 5
wait "${SUB_PID}" 2>/dev/null || true
SUB_PID=""

# SSE format: `data: <payload>\n\n` per event.
RECORDS=$(grep '^data: {' "${SSE_OUT}" | sed 's/^data: //' || true)

if [ -z "${RECORDS}" ]; then
    echo "FAIL: no records arrived on the data channel." >&2
    echo "Raw SSE output (head):" >&2
    head -20 "${SSE_OUT}" | sed 's/^/  /' >&2 || true
    echo "Recent runner activity:" >&2
    docker logs --tail 20 live_detect 2>&1 | sed 's/^/  /' >&2
    exit 1
fi

DETECTIONS=$(echo "${RECORDS}" | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    if r.get('type') == 'detection':
        print(json.dumps(r))
")
TRANSCRIPTS=$(echo "${RECORDS}" | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    if r.get('type') == 'transcript':
        print(json.dumps(r))
")

# YOLO should detect pedestrians on every inference pass. vtest's plaza
# scene is dense enough that "person" hits 5–9× per frame, so this assert
# is robust to alternate VIDEO_URLs that contain people.
if ! echo "${DETECTIONS}" | grep -qE '"label"[[:space:]]*:[[:space:]]*"person"'; then
    echo "FAIL: no person detections — got:" >&2
    echo "${DETECTIONS}" | sed 's/^/  /' >&2
    exit 1
fi

# Whisper should transcribe "country" — the JFK speech says it three times.
if ! echo "${TRANSCRIPTS}" | grep -qi "country"; then
    echo "FAIL: transcripts don't mention 'country' — got:" >&2
    echo "${TRANSCRIPTS}" | sed 's/^/  /' >&2
    exit 1
fi

DET_COUNT=$(echo "${DETECTIONS}" | grep -c '^{' || true)
TRX_COUNT=$(echo "${TRANSCRIPTS}" | grep -c '^{' || true)
echo "PASS (${DET_COUNT} detection record(s), ${TRX_COUNT} transcript record(s)):"
echo "${RECORDS}" | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    t = r.get('type')
    if t == 'detection':
        objs = ', '.join(f\"{o['label']}({o['conf']:.2f})\" for o in r['objects'])
        print(f\"  detection[{r['index']}] frame={r['frame']}: {objs}\")
    elif t == 'transcript':
        print(f\"  transcript[{r['index']}]: {r['text']}\")
"

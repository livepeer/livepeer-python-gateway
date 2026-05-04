#!/usr/bin/env bash
# Push your webcam through the GrayscaleFilter and view the output.
# Run after `docker compose up -d --wait --build`.
#
# Webcam input depends on OS — set WEBCAM_FLAGS to override the default:
#   Linux (default):  WEBCAM_FLAGS="-f v4l2 -i /dev/video0"
#   macOS:            WEBCAM_FLAGS="-f avfoundation -i 0"
#   Windows:          WEBCAM_FLAGS='-f dshow -i video=YourCameraName'
#
# TODO: post-PR-#6, replace the gateway+mediamtx ingestion with
# `start_byoc_job` from `livepeer_gateway.byoc`.

set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"
WEBCAM_FLAGS="${WEBCAM_FLAGS:--f v4l2 -i /dev/video0}"

LIVEPEER_HDR=$(printf '%s' \
  '{"request":"{}","parameters":"{\"enable_video_ingress\":true,\"enable_video_egress\":true}","capability":"live-video-to-video","timeout_seconds":300}' \
  | base64 -w0)

RESPONSE=$(curl -fsS -X POST "${GATEWAY_URL}/process/stream/start" \
    -H "Livepeer: ${LIVEPEER_HDR}" -d '{}')

STREAM_ID=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin)['stream_id'])")
RTMP_IN=$(echo "${RESPONSE}"  | python3 -c "import json,sys; print(json.load(sys.stdin)['rtmp_url'])")
RTMP_OUT=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin)['rtmp_output_url'].split(',')[0])")

trap 'curl -fsS -X POST "${GATEWAY_URL}/process/stream/${STREAM_ID}/stop" -H "Livepeer: ${LIVEPEER_HDR}" -d "{}" >/dev/null 2>&1 || true' EXIT

echo "Pushing webcam → ${RTMP_IN}"
echo "Opening viewer  ← ${RTMP_OUT}"
echo "Press 'q' in the player window (or Ctrl-C here) to stop."

ffmpeg -loglevel error ${WEBCAM_FLAGS} \
       -c:v libx264 -preset ultrafast -tune zerolatency \
       -f flv "${RTMP_IN}" &

ffplay -loglevel error "${RTMP_OUT}"

#!/usr/bin/env bash
# E2E: send a chat request through the gateway, assert the LLM streams
# tokens back via SSE and terminates with [DONE].

set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"
PROMPT="${PROMPT:-Say hello in three words}"

echo "Waiting for capability registration..."
if ! docker logs register_capability 2>&1 | grep -q "registered llm"; then
    echo "FAIL: register_capability hasn't logged success."
    echo "Make sure 'docker compose up -d --wait --build' completed first."
    exit 1
fi
echo "  registered."

# TODO: swap curl for a livepeer_gateway batch caller (post PR #6) — drops
# the gateway service from compose.
LIVEPEER_HDR=$(printf '%s' '{"request":"{}","parameters":"{}","capability":"llm","timeout_seconds":120}' | base64 -w0)

echo "Sending chat request through gateway (streaming)..."
# -N disables curl output buffering so chunks arrive as they're generated.
RESPONSE=$(curl -fsSN -X POST "${GATEWAY_URL}/process/request/predict" \
    -H "Livepeer: ${LIVEPEER_HDR}" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":\"${PROMPT}\"}")

echo "Response (first events):"
echo "${RESPONSE}" | head -10

# Verify SSE format: at least one token event + the [DONE] terminator.
if echo "${RESPONSE}" | grep -q '^data: {"token":' \
   && echo "${RESPONSE}" | grep -q '^data: \[DONE\]'; then
    echo "PASS"
    exit 0
fi

echo "FAIL: expected token events and [DONE] terminator"
exit 1

#!/usr/bin/env bash
# E2E: send a chat request through the gateway, assert the LLM streams
# tokens back via SSE and terminates with [DONE].

# TODO: see README — migration to the Python client SDK.

set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"
PROMPT="${PROMPT:-Say hello in three words}"

echo "Waiting for capability registration..."
# SDK self-registers inside the pipeline container; look for the log line
# emitted by livepeer_gateway.runner.registration.register().
# TODO: switch to `curl /status` once the SDK exposes a status endpoint
# (Phase 2 of auto-registration). Structured check beats log grep.
for _ in $(seq 30); do
    if docker logs llm 2>&1 | grep -q "registered capability=llm"; then
        echo "  registered."
        break
    fi
    sleep 1
done
if ! docker logs llm 2>&1 | grep -q "registered capability=llm"; then
    echo "FAIL: llm container hasn't logged registration success." >&2
    echo "Make sure 'docker compose up -d --wait --build' completed first." >&2
    exit 1
fi
LIVEPEER_HDR=$(printf '%s' '{"request":"{}","parameters":"{}","capability":"llm","timeout_seconds":120}' | base64 -w0)

echo "Sending chat request through gateway (streaming)..."
# -N disables curl output buffering so chunks arrive as they're generated.
RESPONSE=$(curl -fsSN -X POST "${GATEWAY_URL}/process/request/run" \
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

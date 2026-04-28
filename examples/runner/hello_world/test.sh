#!/usr/bin/env bash
# E2E: send a request through the gateway, assert the response from the
# hello_world container comes back through the orchestrator.

set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"
NAME="${NAME:-livepeer}"
EXPECTED_MSG="hello, ${NAME}"

echo "Waiting for capability registration..."
for i in $(seq 1 60); do
    if docker logs register_capability 2>&1 | grep -q "registered hello-world"; then
        echo "  registered."
        break
    fi
    sleep 2
done

# TODO: swap curl for a livepeer_gateway batch caller (post PR #6) — drops
# the gateway service from compose.
LIVEPEER_HDR=$(printf '%s' '{"request":"{}","parameters":"{}","capability":"hello-world","timeout_seconds":30}' | base64 -w0)

echo "Sending request through gateway..."
RESPONSE=$(curl -fsS -X POST "${GATEWAY_URL}/process/request/predict" \
    -H "Livepeer: ${LIVEPEER_HDR}" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"${NAME}\"}")

echo "Response: ${RESPONSE}"

if echo "${RESPONSE}" | grep -qE "\"message\"[[:space:]]*:[[:space:]]*\"${EXPECTED_MSG}\""; then
    echo "PASS"
    exit 0
fi

echo "FAIL"
exit 1

#!/usr/bin/env bash
# E2E: send a request through the gateway, assert the SentimentAnalyzer
# response (label + score) comes back through the orchestrator.

# TODO: see README — migration to the Python client SDK.

set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9935}"
TEXT="${TEXT:-Livepeer makes decentralized inference effortless}"
EXPECTED_LABEL="${EXPECTED_LABEL:-POSITIVE}"

echo "Waiting for capability registration..."
# SDK self-registers inside the pipeline container; look for the log line
# emitted by livepeer_gateway.runner.registration.register().
for _ in $(seq 30); do
    if docker logs sentiment 2>&1 | grep -q "registered capability=sentiment"; then
        echo "  registered."
        break
    fi
    sleep 1
done
if ! docker logs sentiment 2>&1 | grep -q "registered capability=sentiment"; then
    echo "FAIL: sentiment container hasn't logged registration success." >&2
    echo "Make sure 'docker compose up -d --wait --build' completed first." >&2
    exit 1
fi
LIVEPEER_HDR=$(printf '%s' '{"request":"{}","parameters":"{}","capability":"sentiment","timeout_seconds":30}' | base64 -w0)

echo "Sending request through gateway..."
RESPONSE=$(curl -fsS -X POST "${GATEWAY_URL}/process/request/run" \
    -H "Livepeer: ${LIVEPEER_HDR}" \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"${TEXT}\"}")

echo "Response: ${RESPONSE}"

if echo "${RESPONSE}" | grep -qE "\"label\"[[:space:]]*:[[:space:]]*\"${EXPECTED_LABEL}\""; then
    echo "PASS"
    exit 0
fi

echo "FAIL"
exit 1

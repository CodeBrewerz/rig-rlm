#!/bin/bash
# Deploy rig-rlm agent to Restate (local development).
#
# Usage: ./examples/restate_deploy.sh
#
# Prerequisites:
#   - restate-server running (`restate-server --dev`)
#   - cargo build completed

set -euo pipefail

RESTATE_ADMIN="${RESTATE_ADMIN:-http://localhost:9070}"
APP_PORT="${APP_PORT:-9091}"
APP_URI="http://localhost:${APP_PORT}"

echo "🔨 Building restate-server..."
cargo build --bin restate-server

echo "🚀 Starting agent workflow server on port ${APP_PORT}..."
RUST_LOG=info cargo run --bin restate-server &
SERVER_PID=$!
sleep 3

echo "📡 Registering with Restate at ${RESTATE_ADMIN}..."
curl -s -X POST "${RESTATE_ADMIN}/deployments" \
  -H 'Content-Type: application/json' \
  -d "{\"uri\": \"${APP_URI}\"}" | jq .

echo ""
echo "✅ Ready! Try:"
echo "   curl -X POST http://localhost:8080/AgentWorkflow/test-1/run \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"task\": \"What is 2+2?\"}'"
echo ""
echo "   # Check status:"
echo "   curl http://localhost:8080/AgentWorkflow/test-1/status"
echo ""
echo "Press Ctrl+C to stop."

wait $SERVER_PID

#!/bin/bash
# Configure Restate service with retry policy + register deployment.
#
# Usage:
#   ./scripts/configure_restate.sh [agent_endpoint]
#
# Defaults:
#   agent_endpoint = http://localhost:9091
#   restate_admin  = http://localhost:9070

set -euo pipefail

AGENT_ENDPOINT="${1:-http://localhost:9091}"
RESTATE_ADMIN="${RESTATE_ADMIN:-http://localhost:9070}"
SERVICE_NAME="AgentWorkflow"

echo "🔧 Configuring Restate for $SERVICE_NAME"
echo "   Agent endpoint: $AGENT_ENDPOINT"
echo "   Restate admin:  $RESTATE_ADMIN"
echo ""

# 1. Register deployment
echo "📦 Registering deployment..."
DEPLOY_RESULT=$(curl -s -X POST "$RESTATE_ADMIN/deployments" \
  -H 'Content-Type: application/json' \
  -d "{\"uri\": \"$AGENT_ENDPOINT\"}" 2>&1)

if echo "$DEPLOY_RESULT" | grep -q '"id"'; then
  DEPLOY_ID=$(echo "$DEPLOY_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null || echo "unknown")
  echo "   ✅ Registered: $DEPLOY_ID"
else
  echo "   ⚠️  Registration response: $DEPLOY_RESULT"
  echo "   (May already be registered — continuing...)"
fi
echo ""

# 2. Configure retry policy
echo "🔄 Configuring retry policy..."
RETRY_CONFIG=$(cat <<'EOF'
{
  "retry_policy": {
    "type": "Exponential",
    "initial_interval": "1s",
    "factor": 2.0,
    "max_interval": "30s",
    "max_attempts": 10
  },
  "abort_timeout": "600s",
  "inactivity_timeout": "120s"
}
EOF
)

# Apply via admin API (PATCH /services/{service_name})
HTTP_CODE=$(curl -s -o /tmp/restate_config_response.json -w "%{http_code}" \
  -X PATCH "$RESTATE_ADMIN/services/$SERVICE_NAME" \
  -H 'Content-Type: application/json' \
  -d "$RETRY_CONFIG" 2>&1)

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "202" ]; then
  echo "   ✅ Retry policy applied:"
  echo "      initial-interval: 1s"
  echo "      factor: 2.0x exponential backoff"
  echo "      max-interval: 30s"
  echo "      max-attempts: 10"
  echo "      abort-timeout: 600s (10 min)"
  echo "      inactivity-timeout: 120s"
else
  echo "   ⚠️  HTTP $HTTP_CODE — config may need different format"
  echo "   Response: $(cat /tmp/restate_config_response.json 2>/dev/null)"
  echo ""
  echo "   Fallback: use 'restate services config edit $SERVICE_NAME'"
fi

echo ""
echo "🎯 Done! Service ready at:"
echo "   POST http://localhost:8080/$SERVICE_NAME/{id}/run"
echo "   GET  http://localhost:8080/$SERVICE_NAME/{id}/status"

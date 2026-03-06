#!/bin/bash
# Test the health check endpoint
# Usage: ./scripts/test_health.sh [host:port]

HOST_PORT=${1:-"localhost:8001"}
echo "Testing health check on http://$HOST_PORT/health"

curl -s -X GET "http://$HOST_PORT/health" | jq

#!/bin/bash
# Test the list agents endpoint
# Usage: ./scripts/test_agents.sh [host:port]

HOST_PORT=${1:-"localhost:8001"}
echo "Testing list agents on http://$HOST_PORT/agents"

curl -s -X GET "http://$HOST_PORT/agents" | jq

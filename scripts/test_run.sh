#!/bin/bash
# Test the create run endpoint (AG-UI protocol)
# Usage: ./scripts/test_run.sh [question] [host:port] [page_context]

QUESTION=${1:-"What are the available agents?"}
HOST_PORT=${2:-"localhost:8001"}
PAGE_CONTEXT=${3:-"search-relevance"}

THREAD_ID="test-thread-$(date +%s)"
RUN_ID="test-run-$(date +%s)"
MESSAGE_ID="msg-$(date +%s)"

echo "Starting run on http://$HOST_PORT/runs"
echo "Question: $QUESTION"
echo "Thread ID: $THREAD_ID"
echo "Run ID: $RUN_ID"
echo "Message ID: $MESSAGE_ID"
echo "Page Context: $PAGE_CONTEXT"
echo "----------------------------------------"

curl -N -X POST "http://$HOST_PORT/runs" \
  -H "Content-Type: application/json" \
  -d "{
    \"threadId\": \"$THREAD_ID\",
    \"runId\": \"$RUN_ID\",
    \"messages\": [
      {
        \"id\": \"$MESSAGE_ID\",
        \"role\": \"user\",
        \"content\": [
          {
            \"type\": \"text\",
            \"text\": \"$QUESTION\"
          }
        ]
      }
    ],
    \"state\": {},
    \"context\": [
      {
        \"type\": \"page\",
        \"appId\": \"$PAGE_CONTEXT\",
        \"description\": \"Current page context\",
        \"value\": \"$PAGE_CONTEXT\"
      }
    ]
  }"

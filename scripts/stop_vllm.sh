#!/bin/bash
#
# Usage:
#   bash scripts/stop_vllm.sh [PORT or --port PORT]
#
# Description:
#   Gracefully shuts down the background vLLM server running on a specific port.
#   Falls back to a hard SIGKILL if the grace period expires.
#
# Examples:
#   bash scripts/stop_vllm.sh 8000
#   bash scripts/stop_vllm.sh --port 8001
#

# Configuration
TIMEOUT=15  
PORT=8000   # Default fallback port

# 1. Parse port from arguments (accepts raw number or --port flag)
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == "--port" ]]; then
        PORT="${ARGS[i+1]}"
    elif [[ "${ARGS[i]}" =~ --port=(.*) ]]; then
        PORT="${BASH_REMATCH[1]}"
    elif [[ "${ARGS[i]}" =~ ^[0-9]+$ ]]; then
        PORT="${ARGS[i]}"
    fi
done

PID_FILE="vllm_${PORT}.pid"

# 2. Check if the targeted PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "⚠️ No tracking file '$PID_FILE' found. Is a server running on port $PORT?"
    exit 1
fi

VLLM_PID=$(cat "$PID_FILE")

# 3. Check if the process is actually alive
if kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Stopping vLLM server on port $PORT gracefully (PID: $VLLM_PID)..."
    kill "$VLLM_PID" # Sends SIGTERM
    
    # Dynamic wait loop based on the TIMEOUT parameter
    ELAPSED=0
    while [ $ELAPSED -lt $TIMEOUT ]; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "✅ vLLM server on port $PORT stopped successfully."
            rm -f "$PID_FILE"
            exit 0
        fi
        sleep 1
        ((ELAPSED++))
    done

    # Force kill if timeout is reached
    echo "⚠️ Server on port $PORT didn't stop within $TIMEOUT seconds. Forcing shutdown..."
    kill -9 "$VLLM_PID"
    echo "☠️ vLLM server force-killed."
else
    echo "ℹ️ The vLLM process ($VLLM_PID) on port $PORT was already dead."
fi

# Clean up the tracking file artifact
rm -f "$PID_FILE"

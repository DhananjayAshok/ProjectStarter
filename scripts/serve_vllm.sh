#!/bin/bash
#
# Usage: 
#   bash scripts/serve_vllm.sh [VLLM_ARGS...]
#
# Description:
#   A blocking wrapper around 'vllm serve'. Launches the server in the 
#   background, dynamically routes tracking files by port, and holds execution 
#   until the health check passes or the process crashes.
#
# Examples:
#   bash scripts/serve_vllm.sh --model facebook/opt-125m --port 8000
#   bash scripts/serve_vllm.sh --model mistralai/Mistral-7B-Instruct-v0.3 --port 8001
#
# Outputs:
#   vllm_[PORT].pid         - Stores the active background process ID
#   vllm_server_[PORT].log  - Dedicated standard output and error logs
#

PORT=8000
TIMEOUT=1200  # 20 min: hybrid/Mamba models can take ~16+ min to warm up

# 1. Dynamically extract the port from the forwarded arguments
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == "--port" ]]; then
        PORT="${ARGS[i+1]}"
    elif [[ "${ARGS[i]}" =~ --port=(.*) ]]; then
        PORT="${BASH_REMATCH[1]}"
    fi
done

# 2. Assign port-specific filenames
LOG_FILE="vllm_server_${PORT}.log"
PID_FILE="vllm_${PORT}.pid"

echo "Launching vLLM on port $PORT..."
echo "Tracking logs via: $LOG_FILE"

# Start vLLM in its own session/process group so the whole tree (APIServer +
# EngineCore workers, which hold the VRAM) can be killed together later.
setsid nohup vllm serve "$@" > "$LOG_FILE" 2>&1 &
VLLM_PID=$!

# Block until the health endpoint responds or the process crashes
END_TIME=$((SECONDS + TIMEOUT))
while [ $SECONDS -lt $END_TIME ]; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ ERROR: vLLM process crashed. Check $LOG_FILE"
        exit 1
    fi

    if curl -s -f http://localhost:${PORT}/health > /dev/null; then
        echo "✅ SUCCESS: vLLM is healthy on port $PORT!"
        echo $VLLM_PID > "$PID_FILE"
        exit 0
    fi

    sleep 2
done

echo "❌ ERROR: Timed out waiting for vLLM to start on port $PORT."
# Kill the entire process group so EngineCore workers don't orphan and leak VRAM.
# Guard: never group-kill our own group (would take down an interactive shell).
PGID=$(ps -o pgid= -p "$VLLM_PID" | tr -d ' ')
MYPGID=$(ps -o pgid= -p $$ | tr -d ' ')
if [ -n "$PGID" ] && [ "$PGID" != "$MYPGID" ]; then
    kill -TERM -"$PGID" 2>/dev/null
    sleep 5
    kill -KILL -"$PGID" 2>/dev/null
else
    kill -TERM "$VLLM_PID" 2>/dev/null
    sleep 5
    kill -KILL "$VLLM_PID" 2>/dev/null
fi
exit 1

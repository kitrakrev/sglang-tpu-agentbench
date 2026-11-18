#!/bin/bash
# Run ON the TPU VM: nohup scripts/idle_delete.sh NAME ZONE [PROJECT] &
# Deletes this TPU VM after IDLE_MIN (default 120) min with no server/bench python process.
# Hard cap: MAX_MIN (default 240) regardless.
# Fix 2026-11-18: old `pgrep -f "sgl_jax|sglang|agentbench"` matched this script's own command line
# (its path contains "sglang-tpu-agentbench"), so the VM never counted as idle. Now only python
# processes count, and our own PID/parent are excluded.
NAME=$1; ZONE=$2; PROJECT=${3:-torchtpu-sglang-kr}; IDLE=0; UP=0
MAX_MIN=${MAX_MIN:-240}; IDLE_MIN=${IDLE_MIN:-120}; STEP=${STEP:-5}
busy() {
  pgrep -f "python[0-9.]* .*(sgl_jax|agentbench)" | grep -vx -e "$$" -e "$PPID" | grep -q .
}
while true; do
  sleep $((STEP*60)); UP=$((UP+STEP))
  if busy; then IDLE=0; else IDLE=$((IDLE+STEP)); fi
  echo "$(date -u +%FT%TZ) up=${UP}m idle=${IDLE}m"
  if [ $IDLE -ge "$IDLE_MIN" ] || [ $UP -ge "$MAX_MIN" ]; then
    gcloud compute tpus tpu-vm delete "$NAME" --zone="$ZONE" --project="$PROJECT" --quiet
  fi
done

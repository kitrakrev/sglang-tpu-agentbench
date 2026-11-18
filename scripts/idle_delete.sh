#!/bin/bash
# Run ON the TPU VM: nohup scripts/idle_delete.sh NAME ZONE [PROJECT] &
# Deletes this TPU VM after 120 min with no python/sglang process. Hard cap: MAX_MIN (default 240) regardless.
NAME=$1; ZONE=$2; PROJECT=${3:-torchtpu-sglang-kr}; IDLE=0; UP=0; MAX_MIN=${MAX_MIN:-240}
while true; do
  sleep 300; UP=$((UP+5))
  if pgrep -f "sgl_jax|sglang|agentbench" >/dev/null; then IDLE=0; else IDLE=$((IDLE+5)); fi
  if [ $IDLE -ge 120 ] || [ $UP -ge "$MAX_MIN" ]; then
    gcloud compute tpus tpu-vm delete "$NAME" --zone="$ZONE" --project="$PROJECT" --quiet
  fi
done

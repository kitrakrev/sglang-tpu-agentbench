#!/bin/bash
# Run on the TPU VM (nohup ./idle_delete.sh NAME ZONE &). Deletes the TPU VM after 2h with TensorCore duty cycle ~0.
NAME=$1; ZONE=$2; IDLE=0
while true; do
  sleep 300
  if pgrep -f "python|sglang" >/dev/null; then IDLE=0; else IDLE=$((IDLE+5)); fi
  if [ $IDLE -ge 120 ]; then gcloud compute tpus tpu-vm delete "$NAME" --zone="$ZONE" --quiet; fi
done

#!/usr/bin/env bash
# Create a TPU VM for the benchmark. Delete it when done (see idle_delete.sh).
# Usage: scripts/tpu_up.sh [accelerator] [zone]   e.g. v6e-1 us-east5-b
set -euo pipefail
PROJECT=${PROJECT:-torchtpu-sglang-kr}
ACCEL=${1:-v6e-1}
ZONE=${2:-us-east5-b}          # check v6e availability: gcloud compute tpus accelerator-types list --zone=$ZONE
NAME=${NAME:-agentbench-${ACCEL}}
RUNTIME=${RUNTIME:-v2-alpha-tpuv6e}
SPOT=${SPOT---spot}           # set SPOT= (empty) for on-demand

gcloud compute tpus tpu-vm create "$NAME" --project "$PROJECT" --zone "$ZONE" \
  --accelerator-type "$ACCEL" --version "$RUNTIME" $SPOT
echo "ssh: gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --project $PROJECT"
echo "delete: gcloud compute tpus tpu-vm delete $NAME --zone $ZONE --project $PROJECT --quiet"

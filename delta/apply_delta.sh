#!/bin/bash
# ============================================================
# Apply delta to YOLOX-S model on the host VM filesystem.
#
# Model lives at /opt/models/yolox_s.pth on the host, mounted
# into the container at /models — no pod exec or oc cp needed.
# After applying the delta the pod is recreated to pick up the
# updated weights.
#
# Usage:
#   bash apply_delta.sh <delta_file>
#
# Example:
#   bash apply_delta.sh delta_20260625_222149.pth
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DELTA_FILE="${1}"
NAMESPACE="yolox-edge"
MODEL_DIR="/opt/models"
BACKUP_DIR="/opt/backup/models"
MODEL_PATH="${MODEL_DIR}/yolox_s.pth"
WEIGHT_DELTA="${SCRIPT_DIR}/weight_delta.py"

# ── Validate input ────────────────────────────────────────────────────────────
if [ -z "$DELTA_FILE" ]; then
    echo "Usage: bash apply_delta.sh <path_to_delta.pth>"
    echo "Example: bash apply_delta.sh delta_20260625_222149.pth"
    exit 1
fi

if [ ! -f "$DELTA_FILE" ]; then
    echo "   Delta file not found: $DELTA_FILE"
    exit 1
fi

if [ ! -f "$WEIGHT_DELTA" ]; then
    echo "   weight_delta.py not found at ${WEIGHT_DELTA}"
    echo "   Ensure weight_delta.py is in the same directory as this script."
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "   Base model not found at ${MODEL_PATH}"
    exit 1
fi

DELTA_BASENAME=$(basename ${DELTA_FILE})
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo " YOLOX-S Delta Apply"
echo "============================================================"
echo " Delta     : ${DELTA_FILE}"
echo " Base model: ${MODEL_PATH}"
echo " Backup dir: ${BACKUP_DIR}"
echo " Namespace : ${NAMESPACE}"
echo "============================================================"

# ── Backup current model ──────────────────────────────────────────────────────
echo ""
echo "   Backing up current model..."
sudo mkdir -p ${BACKUP_DIR}
BACKUP_PATH="${BACKUP_DIR}/yolox_s_${TIMESTAMP}.pth"
sudo cp ${MODEL_PATH} ${BACKUP_PATH}
echo "   Backed up to ${BACKUP_PATH}"

# ── Apply delta on host using weight_delta.py ─────────────────────────────────
echo ""
echo "    Applying delta..."
python3 - << PYEOF
import sys
import torch

sys.path.insert(0, '${SCRIPT_DIR}')
from weight_delta import apply_head_delta

BASE_MODEL   = '${MODEL_PATH}'
DELTA_FILE   = '${DELTA_FILE}'
OUTPUT_MODEL = '${MODEL_PATH}'   # overwrite in-place

print(f'  Loading base model from {BASE_MODEL}...')
checkpoint = torch.load(BASE_MODEL, map_location='cpu')

print(f'  Loading delta from {DELTA_FILE}...')
delta_payload = torch.load(DELTA_FILE, map_location='cpu')

metadata = delta_payload.get('metadata', {})
print('')
print('  Delta metadata:')
print(f'    Dataset   : {metadata.get("dataset", "unknown")}')
print(f'    Timestamp : {metadata.get("extraction_timestamp", "unknown")}')
print(f'    Layers    : {metadata.get("num_layers", "unknown")}')
print(f'    Notes     : {metadata.get("notes", "")}')
print(f'    Mean |Δ|  : {metadata.get("mean_abs_delta", 0):.6f}')
print(f'    Max  |Δ|  : {metadata.get("max_abs_delta", 0):.6f}')
print('')

print('  Applying delta...')
apply_head_delta(checkpoint, delta_payload['delta'])

print(f'  Saving updated model to {OUTPUT_MODEL}...')
torch.save(checkpoint, OUTPUT_MODEL)

import os
size_mb = os.path.getsize(OUTPUT_MODEL) / 1024 / 1024
print(f'     Done — {size_mb:.1f} MB written to {OUTPUT_MODEL}')
PYEOF

# ── Recreate pod to pick up updated model ────────────────────────────────────
echo ""
echo "   Recreating inference pod..."
POD=$(oc get pods -n ${NAMESPACE} -l app=yolox-s \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -n "$POD" ]; then
    oc delete pod -n ${NAMESPACE} ${POD}
    echo "   Deleted pod: ${POD}"
else
    echo "   No running pod found — skipping delete"
fi

echo "   Waiting for new pod to be ready..."
oc rollout status deployment/yolox-s -n ${NAMESPACE} --timeout=180s

# ── Verify health ─────────────────────────────────────────────────────────────
echo ""
echo "   Checking inference server health..."
NODE_IP=$(oc get nodes -o jsonpath='{.items[0].status.addresses[0].address}')
sleep 5
HEALTH=$(curl -s http://${NODE_IP}:30800/health)
echo "   Health: ${HEALTH}"

echo ""
echo "============================================================"
echo "   Delta applied successfully!"
echo "   Delta   : ${DELTA_BASENAME}"
echo "   Backup  : ${BACKUP_PATH}"
echo "   Model   : ${MODEL_PATH}"
echo ""
echo "Test inference:"
echo "   curl -X POST http://${NODE_IP}:30800/predict \\"
echo "     -F 'file=@/tmp/test.jpg' | python3 -m json.tool"
echo ""
echo "Rollback if needed:"
echo "   sudo cp ${BACKUP_PATH} ${MODEL_PATH}"
echo "   oc delete pod -n ${NAMESPACE} -l app=yolox-s"
echo "============================================================"
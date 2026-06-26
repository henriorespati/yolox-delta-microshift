#!/bin/bash
# ============================================================
# Apply delta to YOLOX-S model inside the running inference pod
# Uses apply_head_delta() from weight_delta.py (Option B)
#
# Usage:
#   bash apply_delta_to_pod.sh <delta_file>
#
# Example:
#   bash apply_delta_to_pod.sh /tmp/delta_20260625_222149.pth
# ============================================================

set -e

NAMESPACE="yolox-edge"
DELTA_FILE="${1}"

# ── Validate input ────────────────────────────────────────────────────────────
if [ -z "$DELTA_FILE" ]; then
    echo "   Usage: bash apply_delta_to_pod.sh <path_to_delta.pth>"
    echo "   Example: bash apply_delta_to_pod.sh /tmp/delta_20260625_222149.pth"
    exit 1
fi

if [ ! -f "$DELTA_FILE" ]; then
    echo "   Delta file not found: $DELTA_FILE"
    exit 1
fi

# ── Get running pod name ──────────────────────────────────────────────────────
echo "🔍 Finding running inference pod..."
POD=$(oc get pods -n ${NAMESPACE} -l app=yolox-s \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$POD" ]; then
    echo "   No running pod found with label app=yolox-s in namespace ${NAMESPACE}"
    exit 1
fi

echo "   Found pod: $POD"

# ── Verify weight_delta.py is available in the pod ───────────────────────────
echo "🔍 Checking weight_delta.py in pod..."
if ! oc exec -n ${NAMESPACE} ${POD} -- ls /opt/weight-delta/weight_delta.py > /dev/null 2>&1; then
    echo "   /opt/weight-delta/weight_delta.py not found in pod."
    echo "   Rebuild the image with the updated Dockerfile and redeploy."
    exit 1
fi
echo "   weight_delta.py found at /opt/weight-delta/weight_delta.py"

# ── Check base model exists in pod ───────────────────────────────────────────
echo "🔍 Checking base model in pod..."
if ! oc exec -n ${NAMESPACE} ${POD} -- ls /models/yolox_s.pth > /dev/null 2>&1; then
    echo "   /models/yolox_s.pth not found in pod. Run the model-init Job first."
    exit 1
fi

MODEL_SIZE=$(oc exec -n ${NAMESPACE} ${POD} -- \
    python3 -c "import os; print(f'{os.path.getsize(\"/models/yolox_s.pth\") / 1024 / 1024:.1f} MB')")
echo "   Base model found: ${MODEL_SIZE}"

# ── Copy delta file into pod ──────────────────────────────────────────────────
DELTA_BASENAME=$(basename ${DELTA_FILE})
echo ""
echo "📦 Copying delta file into pod..."
echo "   ${DELTA_FILE} → /models/${DELTA_BASENAME}"
oc cp ${DELTA_FILE} ${NAMESPACE}/${POD}:/models/${DELTA_BASENAME}
echo "   Delta copied"

# ── Apply delta inside pod ────────────────────────────────────────────────────
echo ""
echo "⚙️  Applying delta inside pod..."
oc exec -n ${NAMESPACE} ${POD} -- python3 - << PYEOF
import sys
import torch

sys.path.insert(0, '/opt/weight-delta')
from weight_delta import apply_head_delta

BASE_MODEL  = '/models/yolox_s.pth'
DELTA_FILE  = '/models/${DELTA_BASENAME}'
OUTPUT_MODEL = '/models/yolox_s.pth'   # overwrite in-place

print(f'Loading base model from {BASE_MODEL}...')
checkpoint = torch.load(BASE_MODEL, map_location='cpu')

print(f'Loading delta from {DELTA_FILE}...')
delta_payload = torch.load(DELTA_FILE, map_location='cpu')

metadata = delta_payload.get('metadata', {})
print('')
print('Delta metadata:')
print(f'  Dataset   : {metadata.get("dataset", "unknown")}')
print(f'  Timestamp : {metadata.get("extraction_timestamp", "unknown")}')
print(f'  Layers    : {metadata.get("num_layers", "unknown")}')
print(f'  Notes     : {metadata.get("notes", "")}')
print(f'  Mean |Δ|  : {metadata.get("mean_abs_delta", 0):.6f}')
print(f'  Max  |Δ|  : {metadata.get("max_abs_delta", 0):.6f}')
print('')

print('Applying delta...')
apply_head_delta(checkpoint, delta_payload['delta'])

print(f'Saving updated model to {OUTPUT_MODEL}...')
torch.save(checkpoint, OUTPUT_MODEL)

import os
size_mb = os.path.getsize(OUTPUT_MODEL) / 1024 / 1024
print(f'   Model updated — {size_mb:.1f} MB saved to {OUTPUT_MODEL}')
PYEOF

# ── Clean up delta file from pod ─────────────────────────────────────────────
echo ""
echo "🧹 Cleaning up delta file from pod..."
oc exec -n ${NAMESPACE} ${POD} -- rm -f /models/${DELTA_BASENAME}

# ── Hot-reload inference server ───────────────────────────────────────────────
echo ""
echo "🔄 Hot-reloading inference server..."
NODE_IP=$(oc get nodes -o jsonpath='{.items[0].status.addresses[0].address}')
RELOAD_RESPONSE=$(curl -s -X POST http://${NODE_IP}:30800/reload-model)
echo "   Response: ${RELOAD_RESPONSE}"

# ── Verify health ─────────────────────────────────────────────────────────────
echo ""
echo "🏥 Checking inference server health..."
sleep 3
HEALTH=$(curl -s http://${NODE_IP}:30800/health)
echo "   Health: ${HEALTH}"

echo ""
echo "============================================================"
echo "   Delta applied successfully!"
echo "   Pod      : ${POD}"
echo "   Delta    : ${DELTA_BASENAME}"
echo "   Reload   : ${RELOAD_RESPONSE}"
echo ""
echo "Test inference:"
echo "   curl -X POST http://${NODE_IP}:30800/predict \\"
echo "     -F 'file=@/tmp/test.jpg' | python3 -m json.tool"
echo "============================================================"

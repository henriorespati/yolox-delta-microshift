# YOLOX-S on MicroShift — Edge Inference Pipeline

YOLOX-S object detection deployed on Red Hat MicroShift (OpenShift on a single EC2 node), with a Mothership-to-Edge delta update pipeline for continuous model improvement without distributing full model weights.

---

## Repository layout

```
microshift/
├── Containerfile              ← Container image (ubi9/python-314-minimal + PyTorch CPU + YOLOX)
├── app/
│   └── inference_server.py    ← FastAPI inference server (COCO 80 classes)
├── ocp/
│   ├── namespace.yml          ← Namespace: yolox-edge
│   ├── configmap.yml          ← Inference thresholds, model path, delta server URL
│   ├── pvc.yml                ← PersistentVolumeClaim for model weights (lvms-operator)
│   ├── model-init-job.yml     ← One-shot Job: downloads yolox_s.pth from Megvii release
│   ├── deployment.yml         ← Deployment: inference pod with init container
│   └── service.yml            ← NodePort service on port 30800
└── delta/
    ├── apply_delta.sh         ← Script: applies a delta .pth to the running pod
    └── delta_sample.pth       ← Sample delta (~3.7 MB, float16 compressed)
```

---

## Prerequisites

- RHEL 9 EC2 instance with MicroShift installed and running
- Podman for building the container image
- `oc` CLI configured with MicroShift kubeconfig
- EBS volume attached and mounted (for container storage)

---

## Quick start

### 1. Build and load the container image

```bash
# Build
podman build -f Containerfile -t yolox-s-inference:latest .

# Load into MicroShift's CRI-O runtime (no registry needed)
podman save yolox-s-inference:latest -o /tmp/yolox-s-inference.tar
sudo skopeo copy \
  docker-archive:/tmp/yolox-s-inference.tar \
  containers-storage:localhost/yolox-s-inference:latest
```

### 2. Deploy to MicroShift

```bash
oc apply -f ocp/namespace.yml
oc apply -f ocp/ -n yolox-edge

# Wait for model download Job to complete
oc wait -n yolox-edge job/yolox-model-init --for=condition=complete --timeout=120s

# Watch the inference pod come up (~90s for PyTorch to load)
oc get pods -n yolox-edge -w
```

### 3. Test inference

```bash
NODE_IP=$(oc get nodes -o jsonpath='{.items[0].status.addresses[0].address}')

# Health check
curl http://${NODE_IP}:30800/health

# Run inference on an image
curl -X POST http://${NODE_IP}:30800/predict \
  -F "file=@/path/to/image.jpg" | python3 -m json.tool
```

---

## Applying a delta update

Model updates are distributed as small delta `.pth` files (~3.7 MB) rather than full model weights (~43 MB). A delta captures only the changes to the detection head after fine-tuning.

```bash
# Apply a delta to the running pod and hot-reload
bash delta/apply_delta.sh delta/delta_sample.pth
```

The script:
1. Copies the delta into the pod via base64 transfer
2. Runs `apply_head_delta()` from `weight_delta.py` inside the pod
3. Saves the updated `yolox_s.pth` back to the PVC
4. Calls `POST /reload-model` to hot-swap weights without restarting the pod

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — returns model load status |
| `GET` | `/status` | Stats — avg inference ms, total inference count |
| `POST` | `/predict` | Single image inference — returns bounding boxes and class labels |
| `POST` | `/predict/batch` | Batch inference — multiple images in one request |
| `POST` | `/reload-model` | Hot-reload weights from disk after delta update |

---

## Container image notes

Base image: `registry.access.redhat.com/ubi9/python-314-minimal`

Key decisions:
- **No C extension compilation** — YOLOX's `fast_cocoeval` extension requires Python dev headers not available in the minimal image. It is only used for COCO evaluation, not inference. YOLOX source is added to `PYTHONPATH` instead.
- **Free UBI repos only** — `microdnf` is restricted to `ubi-9-baseos-rpms` and `ubi-9-appstream-rpms` to avoid 403 errors from entitled RHEL CDN repos on unregistered hosts.
- **Non-root user 1001** — required by MicroShift's Security Context Constraints.
- **`weight_delta.py` included** — cloned from `ganeshmurthy/yolo-weight-delta` at `/opt/weight-delta/` so delta application runs inside the same pod environment as inference.

---

## Fine-tuning the model (Mothership)

To generate a new delta from a retrained model:

```bash
python weight_delta.py \
  --base-model    models/yolox_s.pth \
  --trained-model YOLOX_outputs/yolox_s_finetune/latest_ckpt.pth \
  --output        delta/delta_$(date +%Y%m%d_%H%M%S).pth \
  --dataset       "your_dataset_name" \
  --notes         "Description of what changed"
```

Model source: [Megvii YOLOX 0.1.1rc0](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0) — COCO 80 classes, 40.5 mAP.
Delta pipeline: [ganeshmurthy/yolo-weight-delta](https://github.com/ganeshmurthy/yolo-weight-delta)

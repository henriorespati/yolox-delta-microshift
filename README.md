# YOLOX-S on MicroShift — Delta Model Deployment

Deploy YOLOX-S object detection on Red Hat MicroShift without rebuilding container images. Delta weights are merged with the base model on the host and made available to the inference pod via a `hostPath` volume mount at `/opt/models`. Updating the model is a matter of running `apply_delta.sh` on the host and cycling the deployment replicas.

---

## Repository layout

```
yolox-delta-microshift/
├── Containerfile              ← Container image (ubi9/python-314-minimal + PyTorch CPU + YOLOX)
├── app/
│   └── inference_server.py    ← FastAPI inference server (COCO 80 classes)
├── ocp/
│   ├── namespace.yml          ← Namespace: yolox-edge
│   ├── deployment.yml         ← Deployment: hostPath mount from /opt/models (read-only)
│   └── service.yml            ← NodePort service on port 30800
└── delta/
    ├── apply_delta.sh         ← Script: merges delta weights with base model on the host
    └── delta_sample.pth       ← Sample delta (~3.7 MB, float16 compressed)
```

---

## How it works

```
                   MicroShift Node
┌────────────────────────────────────────────────────────┐
│                                                        │
│   /opt/models/yolox_s.pth   ◄── apply_delta.sh merges  │
│          │                       delta + base model    │
│          │ hostPath (readOnly)                         │
│          ▼                                             │
│   ┌──────────────────────┐                             │
│   │  yolox-s pod         │                             │
│   │  /models/yolox_s.pth │──► FastAPI :8000            │
│   └──────────────────────┘       │                     │
│                                  │ NodePort 30800      │
└──────────────────────────────────┼─────────────────────┘
                                   ▼
                              Inference clients
```

1. The base model (`yolox_s.pth`) is pre-placed at `/opt/models` on the MicroShift host.
2. The deployment mounts `/opt/models` into the pod as a read-only `hostPath` volume.
3. When a new delta is available, `apply_delta.sh` merges it with the base model on the host, producing an updated `/opt/models/yolox_s.pth`.
4. Scale the deployment down and back up to pick up the new model — no image rebuild required.

---

## Prerequisites

- MicroShift installed and running
- Podman for building the container image
- `oc` CLI configured with MicroShift kubeconfig
- Base model (`yolox_s.pth`) present at `/opt/models/yolox_s.pth` on the MicroShift host
- **For delta merging on the host**: Python 3, PyTorch, and [yolo-weight-delta](https://github.com/ganeshmurthy/yolo-weight-delta) installed on the MicroShift host

---

## Quick start

### 1. Prepare the host model directory

Ensure the base YOLOX-S model is available on the MicroShift host:

```bash
sudo mkdir -p /opt/models

# Download the base model (if not already present)
sudo curl -L -o /opt/models/yolox_s.pth \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

### 2. Build and load the container image

```bash
# Build
podman build -f Containerfile -t yolox-s-inference:latest .

# Load into MicroShift's CRI-O runtime (no registry needed)
podman save yolox-s-inference:latest -o /tmp/yolox-s-inference.tar
sudo skopeo copy \
  docker-archive:/tmp/yolox-s-inference.tar \
  containers-storage:localhost/yolox-s-inference:latest
```

### 3. Deploy to MicroShift

```bash
oc apply -f ocp/namespace.yml

# Grant hostmount-anyuid SCC to default service account for hostPath access
oc adm policy add-scc-to-user hostmount-anyuid -z default -n yolox-edge

oc apply -f ocp/deployment.yml
oc apply -f ocp/service.yml

# Watch the inference pod come up (~90s for PyTorch to load)
oc get pods -n yolox-edge -w
```

### 4. Test inference

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

### Merge the delta and redeploy

```bash
# 1. Merge delta weights with the base model on the host
bash delta/apply_delta.sh delta/delta_sample.pth

# 2. Scale down the deployment (stops the running pod)
oc scale deployment/yolox-s -n yolox-edge --replicas=0

# 3. Scale back up (pod starts with the updated model)
oc scale deployment/yolox-s -n yolox-edge --replicas=1
```

That's it — no image rebuild, no container registry push, no PVC provisioning.

> **Note:** The `apply_delta.sh` script currently contains logic for in-pod delta application (via `oc cp` / `oc exec`). It needs to be updated to run the merge directly on the host against `/opt/models/yolox_s.pth`. The merge logic uses `apply_head_delta()` from [yolo-weight-delta](https://github.com/ganeshmurthy/yolo-weight-delta).

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — returns model load status |
| `GET` | `/status` | Stats — avg inference ms, total inference count |
| `POST` | `/predict` | Single image inference — returns bounding boxes and class labels |
| `POST` | `/predict/batch` | Batch inference — multiple images in one request |
| `POST` | `/reload-model` | Hot-reload weights from disk without restarting the pod |

---

## Container image notes

Base image: `registry.access.redhat.com/ubi9/python-314-minimal`

- **No C extension compilation** — YOLOX's `fast_cocoeval` extension requires Python dev headers not available in the minimal image. It is only used for COCO evaluation, not inference. YOLOX source is added to `PYTHONPATH` instead.
- **Free UBI repos only** — `microdnf` is restricted to `ubi-9-baseos-rpms` and `ubi-9-appstream-rpms` to avoid 403 errors from entitled RHEL CDN repos on unregistered hosts.
- **Non-root user 1001** — required by MicroShift's Security Context Constraints.
- **`weight_delta.py` included** — cloned from `ganeshmurthy/yolo-weight-delta` at `/opt/weight-delta/` for in-pod delta operations.

---

## Generating deltas (Mothership)

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

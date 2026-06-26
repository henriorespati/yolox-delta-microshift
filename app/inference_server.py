"""
YOLOX-S Inference Server for MicroShift edge deployment.
Uses the official Megvii YOLOX source from:
  https://github.com/Megvii-BaseDetection/YOLOX
"""
import os
import sys
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

sys.path.insert(0, "/opt/YOLOX")

from yolox.exp import get_exp
from yolox.utils import postprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH     = os.getenv("MODEL_PATH", "/models/yolox_s.pth")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
NMS_THRESHOLD  = float(os.getenv("NMS_THRESHOLD", "0.45"))
INPUT_SIZE     = int(os.getenv("INPUT_SIZE", "640"))
NUM_CLASSES    = int(os.getenv("NUM_CLASSES", "80"))
DEVICE         = "cpu"

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

state = {
    "model":           None,
    "loaded_at":       None,
    "inference_count": 0,
    "total_ms":        0.0,
}


def preprocess_image(img_bgr: np.ndarray):
    """
    Letterbox resize to INPUT_SIZE x INPUT_SIZE and convert to tensor.
    Replicates YOLOX's internal preproc exactly so ratio is always a
    reliable Python float — avoids ValTransform returning None for ratio
    in certain YOLOX versions.
    """
    h, w = img_bgr.shape[:2]
    ratio = min(INPUT_SIZE / h, INPUT_SIZE / w)

    new_h = int(h * ratio)
    new_w = int(w * ratio)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Letterbox canvas filled with 114 (YOLOX default pad value)
    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    # HWC BGR → CHW float32 (YOLOX does NOT normalise to [0,1])
    img = padded[:, :, ::-1].transpose(2, 0, 1)   # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img, dtype=np.float32)

    tensor = torch.from_numpy(img).unsqueeze(0)    # [1, 3, H, W]
    return tensor, float(ratio)


def load_model():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run the model-init Job first."
        )
    logger.info(f"Loading YOLOX-S from {MODEL_PATH} ...")

    exp = get_exp(exp_name="yolox-s")
    exp.num_classes = NUM_CLASSES
    exp.test_conf   = CONF_THRESHOLD
    exp.nmsthre     = NMS_THRESHOLD
    exp.test_size   = (INPUT_SIZE, INPUT_SIZE)

    model = exp.get_model()
    model.eval()

    ckpt    = torch.load(model_path, map_location=DEVICE)
    weights = ckpt.get("model", ckpt)
    model.load_state_dict(weights)

    state["model"]     = model
    state["loaded_at"] = datetime.utcnow().isoformat() + "Z"
    logger.info("✅ YOLOX-S model ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="YOLOX-S Inference Server",
    description="Edge inference for YOLOX-S on MicroShift (COCO 80 classes)",
    version="1.0.0",
    lifespan=lifespan,
)


def run_inference(img_bgr: np.ndarray):
    img_tensor, ratio = preprocess_image(img_bgr)

    with torch.no_grad():
        t0      = time.perf_counter()
        raw_out = state["model"](img_tensor)
        outputs = postprocess(
            raw_out,
            NUM_CLASSES,
            CONF_THRESHOLD,
            NMS_THRESHOLD,
            class_agnostic=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

    state["inference_count"] += 1
    state["total_ms"]        += elapsed_ms

    if outputs[0] is None:
        return [], elapsed_ms

    # outputs[0]: tensor [N, 7] → x1,y1,x2,y2, obj_conf, cls_conf, cls_id
    dets = outputs[0].cpu().numpy()

    detections = []
    for row in dets:
        x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
        obj_conf = float(row[4])
        cls_conf = float(row[5])
        cls_id   = int(row[6])

        # Scale back to original image size
        x1 /= ratio
        y1 /= ratio
        x2 /= ratio
        y2 /= ratio

        detections.append({
            "bbox":       [round(float(x1), 2), round(float(y1), 2),
                           round(float(x2), 2), round(float(y2), 2)],
            "confidence": round(obj_conf * cls_conf, 4),
            "class_id":   cls_id,
            "class_name": COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"cls_{cls_id}",
        })

    return detections, elapsed_ms


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":       "healthy" if state["model"] is not None else "degraded",
        "model_loaded": state["model"] is not None,
        "loaded_at":    state["loaded_at"],
    }


@app.get("/status")
async def status():
    count = state["inference_count"]
    return {
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "model_loaded":     state["model"] is not None,
        "loaded_at":        state["loaded_at"],
        "model_path":       MODEL_PATH,
        "device":           DEVICE,
        "input_size":       INPUT_SIZE,
        "conf_threshold":   CONF_THRESHOLD,
        "nms_threshold":    NMS_THRESHOLD,
        "num_classes":      NUM_CLASSES,
        "total_inferences": count,
        "avg_inference_ms": round(state["total_ms"] / count, 2) if count else 0,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image")

    h, w = img.shape[:2]

    try:
        detections, elapsed_ms = run_inference(img)
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse({
        "filename":       file.filename,
        "image_size":     {"width": w, "height": h},
        "num_detections": len(detections),
        "detections":     detections,
        "inference_ms":   round(elapsed_ms, 2),
        "conf_threshold": CONF_THRESHOLD,
        "nms_threshold":  NMS_THRESHOLD,
    })


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for file in files:
        try:
            contents = await file.read()
            nparr    = np.frombuffer(contents, np.uint8)
            img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                results.append({"filename": file.filename, "error": "invalid image"})
                continue
            dets, ms = run_inference(img)
            results.append({
                "filename":       file.filename,
                "num_detections": len(dets),
                "detections":     dets,
                "inference_ms":   round(ms, 2),
            })
        except Exception as exc:
            results.append({"filename": file.filename, "error": str(exc)})

    return JSONResponse({"results": results, "count": len(results)})


@app.post("/reload-model")
async def reload_model():
    """Hot-reload after delta update writes a new yolox_s.pth to the PVC."""
    try:
        load_model()
        return {"status": "reloaded", "loaded_at": state["loaded_at"]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

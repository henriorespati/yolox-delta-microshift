FROM registry.access.redhat.com/ubi9/python-314-minimal

# ── Notes on this base image ─────────────────────────────────────────────────
# • Package manager : microdnf (not apt-get / dnf)
# • Default user    : 1001 (unprivileged) — switch to root for installs
# • Python          : 3.14 pre-installed at /opt/app-root/
# ─────────────────────────────────────────────────────────────────────────────

USER root

WORKDIR /app

# System packages — restricted to free UBI repos only.
# --disablerepo prevents microdnf hitting entitled RHEL CDN repos (403 on
# unregistered hosts).
# NOTE: gcc/make are NOT needed anymore — we skip YOLOX's C extension
# compilation entirely (fast_cocoeval is COCO evaluation only, not inference).
RUN microdnf install -y \
    --disablerepo='*' \
    --enablerepo='ubi-9-baseos-rpms' \
    --enablerepo='ubi-9-appstream-rpms' \
    git \
    wget \
    tar \
    && microdnf clean all

# Pin setuptools<82 before torch (torch 2.12.x requires setuptools<82)
RUN pip install --no-cache-dir "setuptools<82"

# Install PyTorch CPU — torch 2.6+ ships cp314 wheels on the CPU index
RUN pip install --no-cache-dir \
    "torch>=2.6" \
    "torchvision>=0.21" \
    --index-url https://download.pytorch.org/whl/cpu

# Install YOLOX Python dependencies (mirrors official requirements.txt)
# opencv-python-headless: no libGL/X11 dep — works in UBI9 minimal
# thop/ninja/pycocotools: pulled in by YOLOX imports at runtime
RUN pip install --no-cache-dir \
    numpy \
    "opencv-python-headless>=4.8" \
    loguru \
    tqdm \
    thop \
    tabulate \
    psutil \
    "pycocotools>=2.0.2" \
    "fastapi==0.115.0" \
    "uvicorn[standard]==0.32.0" \
    "python-multipart==0.0.12" \
    packaging

# Clone YOLOX source — NO pip install, NO C extension compilation.
#
# The fast_cocoeval C extension in yolox/layers/cocoeval/ requires Python.h
# (python3.14-devel) which is not in the UBI9 minimal image, and is only
# used for COCO benchmark evaluation — never for inference.
#
# Adding /opt/YOLOX to PYTHONPATH gives full access to all inference APIs:
#   yolox.exp.get_exp(), yolox.utils.postprocess(), ValTransform, etc.
RUN git clone --depth=1 https://github.com/Megvii-BaseDetection/YOLOX.git /opt/YOLOX && \
    rm -rf /opt/YOLOX/.git && \
    # Remove training-only imports not needed for inference.
    # mlflow_logger pulls in packaging, mlflow, and other heavy training deps.
    # LRScheduler pulls in tensorboard. Neither is used during edge inference.
    sed -i '/from .mlflow_logger import MlflowLogger/d' /opt/YOLOX/yolox/utils/__init__.py && \
    sed -i '/from .logger import/d' /opt/YOLOX/yolox/utils/__init__.py

# Clone weight_delta repo
RUN git clone --depth=1 https://github.com/ganeshmurthy/yolo-weight-delta.git /opt/weight-delta && \
    rm -rf /opt/weight-delta/.git

ENV PYTHONPATH="/opt/YOLOX:/opt/weight-delta"

# Copy inference server application
COPY app/ /app/

# Create model directory and fix ownership for non-root user
RUN mkdir -p /models && \
    chown -R 1001:1001 /app /models /opt/YOLOX

EXPOSE 8000

# Switch back to non-root (required for OpenShift/MicroShift SCC)
USER 1001

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD wget -qO- http://localhost:8000/health || exit 1

CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

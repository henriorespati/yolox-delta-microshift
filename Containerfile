FROM registry.access.redhat.com/ubi9/python-314-minimal AS builder

WORKDIR /build

# Copy source repos
COPY YOLOX /build/YOLOX
COPY yolo-weight-delta /build/weight-delta

# ponytail: directories stripped on host before COPY to avoid macOS extended attrs
# ponytail: training imports removed on host via sed before COPY

# ── Final stage ──────────────────────────────────────────────────────────
FROM registry.access.redhat.com/ubi9/python-314-minimal

USER root

WORKDIR /app

# System packages — restricted to free UBI repos only.
RUN microdnf install -y \
    --disablerepo='*' \
    --enablerepo='ubi-9-baseos-rpms' \
    --enablerepo='ubi-9-appstream-rpms' \
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

# Copy pre-cloned YOLOX and weight-delta from builder stage
COPY --from=builder /build/YOLOX /opt/YOLOX
COPY --from=builder /build/weight-delta /opt/weight-delta

ENV PYTHONPATH="/opt/YOLOX:/opt/weight-delta"

# Copy inference server application
COPY app/ /app/

# Create model directory and fix ownership for non-root user
RUN mkdir -p /models && \
    chown -R 1001:1001 /app /models /opt/YOLOX /opt/weight-delta

EXPOSE 8000

# Switch back to non-root (required for OpenShift/MicroShift SCC)
USER 1001

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD wget -qO- http://localhost:8000/health || exit 1

CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

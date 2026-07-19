FROM registry.redhat.io/rhel9/python-314

USER root
WORKDIR /app

RUN dnf config-manager --set-enabled codeready-builder-for-rhel-9-x86_64-rpms && \
    dnf install -y \
        wget \
        gcc \
        pkgconf-pkg-config \
        gobject-introspection-devel \
        cairo-devel \
        cairo-gobject-devel \
        gstreamer1 \
        gstreamer1-plugins-base \
        gstreamer1-plugins-good \
        gstreamer1-plugins-bad-free && \
    dnf clean all && \
    pip install --no-cache-dir "setuptools<82" && \
    pip install --no-cache-dir \
        "torch>=2.6" \
        "torchvision>=0.21" \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir pycairo "pygobject<3.51" && \
    pip install --no-cache-dir \
        numpy \
        "opencv-python-headless>=4.8" \
        loguru \
        thop \
        tabulate \
        psutil \
        "pycocotools>=2.0.2" \
        "fastapi==0.115.0" \
        "uvicorn[standard]==0.32.0" \
        "python-multipart==0.0.12" \
        packaging \
        pyzmq

COPY YOLOX /opt/YOLOX
COPY yolo-weight-delta /opt/weight-delta

ENV PYTHONPATH="/opt/YOLOX:/opt/weight-delta"

# Copy inference server application
COPY app/ /app/

RUN mkdir -p /models && \
    chgrp -R 0 /app /models /opt/YOLOX /opt/weight-delta && \
    chmod -R g=u /app /models /opt/YOLOX /opt/weight-delta

EXPOSE 8000

USER 1001

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD wget -qO- http://localhost:8000/health || exit 1

CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM registry.redhat.io/rhel9/python-314 AS openh264-builder

USER root

RUN dnf config-manager --set-enabled codeready-builder-for-rhel-9-x86_64-rpms && \
    dnf install -y \
        wget \
        bzip2 \
        git \
        meson \
        ninja-build \
        gcc \
        gcc-c++ \
        pkgconf-pkg-config \
        gstreamer1-devel \
        gstreamer1-plugins-base-devel \
        glib2-devel \
        orc-devel && \
    dnf clean all && \
    mkdir -p /build && cd /build && \
    wget -q http://ciscobinary.openh264.org/libopenh264-2.6.0-linux64.8.so.bz2 -O libopenh264.so.bz2 && \
    bunzip2 libopenh264.so.bz2 && \
    echo "1859c0aaf825429cbf36f1f496c5e08c  libopenh264.so" | md5sum -c - && \
    git clone --quiet --depth 1 --branch v2.6.0 https://github.com/cisco/openh264.git openh264-src && \
    mkdir -p /opt/openh264/include/wels /opt/openh264/lib64/pkgconfig && \
    cp openh264-src/codec/api/wels/*.h /opt/openh264/include/wels/ && \
    cp libopenh264.so /opt/openh264/lib64/libopenh264.so.8 && \
    ln -sf libopenh264.so.8 /opt/openh264/lib64/libopenh264.so && \
    printf 'prefix=/opt/openh264\nlibdir=${prefix}/lib64\nincludedir=${prefix}/include\n\nName: openh264\nDescription: OpenH264 codec library (Cisco binary)\nVersion: 2.6.0\nLibs: -L${libdir} -lopenh264\nCflags: -I${includedir}\n' > /opt/openh264/lib64/pkgconfig/openh264.pc && \
    wget -q https://gstreamer.freedesktop.org/src/gst-plugins-bad/gst-plugins-bad-1.22.12.tar.xz && \
    tar xf gst-plugins-bad-1.22.12.tar.xz && \
    cd gst-plugins-bad-1.22.12 && \
    PKG_CONFIG_PATH=/opt/openh264/lib64/pkgconfig meson setup builddir -Dauto_features=disabled -Dopenh264=enabled -Dgpl=disabled --prefix=/usr && \
    ninja -C builddir ext/openh264/libgstopenh264.so

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

COPY app/ /app/
COPY --from=openh264-builder /build/gst-plugins-bad-1.22.12/builddir/ext/openh264/libgstopenh264.so /usr/lib64/gstreamer-1.0/libgstopenh264.so
COPY --from=openh264-builder /build/libopenh264.so /usr/lib64/libopenh264.so.8

RUN mkdir -p /models && \
    chgrp -R 0 /app /models /opt/YOLOX /opt/weight-delta && \
    chmod -R g=u /app /models /opt/YOLOX /opt/weight-delta && \
    ln -sf libopenh264.so.8 /usr/lib64/libopenh264.so && \
    ldconfig

EXPOSE 8000

USER 1001

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD wget -qO- http://localhost:8000/health || exit 1

CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

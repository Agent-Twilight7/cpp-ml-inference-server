# CPU image. Build the model first (python models/export_resnet.py), then:
#   docker build -t inference-server .
#   docker run --rm -p 8080:8080 inference-server
# Extra server flags go after the image name, e.g. `... inference-server --max-batch 8`.

# ---- build stage ----
FROM ubuntu:22.04 AS build
ARG ORT_VERSION=1.30.0
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake ninja-build curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY scripts/fetch_onnxruntime.sh scripts/
RUN ORT_VERSION=${ORT_VERSION} bash scripts/fetch_onnxruntime.sh
COPY CMakeLists.txt .
COPY include include
COPY src src
COPY apps apps
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DINFER_BUILD_TESTS=OFF \
 && cmake --build build --target inference_server loadgen

# ---- runtime stage ----
FROM ubuntu:22.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --create-home app
COPY --from=build /src/third_party/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/
RUN ldconfig
COPY --from=build /src/build/inference_server /src/build/loadgen /usr/local/bin/
COPY models/resnet18.onnx /models/resnet18.onnx
USER app
EXPOSE 8080
# Docker sends SIGTERM on stop; the server drains in-flight requests first.
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=10s --timeout=2s --start-period=30s CMD curl -fs http://localhost:8080/health || exit 1
ENTRYPOINT ["inference_server", "--model", "/models/resnet18.onnx", "--port", "8080"]
CMD ["--workers", "2", "--ort-threads", "4", "--max-batch", "8"]

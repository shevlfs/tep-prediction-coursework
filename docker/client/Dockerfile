FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir \
        rknn-toolkit-lite2 \
        numpy \
        opcua

WORKDIR /app
COPY docker/client/inference_opcua_client.py /app/
COPY rknn/model_fp.rknn /app/model_fp.rknn
COPY onnx/norm_mean.npy /app/
COPY onnx/norm_std.npy /app/

ENTRYPOINT ["python3", "inference_opcua_client.py"]
CMD ["opc.tcp://192.168.88.243:4840"]

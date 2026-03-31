FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates python3 && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    git clone --depth 1 --branch v1.4.6 https://github.com/open62541/open62541.git && \
    cd open62541 && mkdir build && cd build && \
    cmake -DUA_ENABLE_AMALGAMATION=ON -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

COPY opcua/server.c /tmp/build/server.c
RUN cp /tmp/open62541/build/open62541.c /tmp/open62541/build/open62541.h /tmp/build/ && \
    cd /tmp/build && \
    gcc -O2 -o opcua_server server.c open62541.c -I. -lpthread

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends libc6 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /tmp/build/opcua_server /usr/local/bin/opcua_server
COPY data/small_tep/df.csv /data/df.csv

EXPOSE 4840

ENTRYPOINT ["opcua_server"]
CMD ["/data/df.csv"]

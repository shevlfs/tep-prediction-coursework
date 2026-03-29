#define _POSIX_C_SOURCE 200809L

#include "open62541.h"

#include <signal.h>
#include <stdio.h>
#include <unistd.h>

#define ENDPOINT_URL "opc.tcp://192.168.88.243:4840"
#define NAMESPACE_URI "urn:my-csv-server"
#define POLL_INTERVAL_MS 5000

static volatile sig_atomic_t running = 1;

static void stop_handler(int signum) {
    (void)signum;
    running = 0;
}

static UA_Boolean read_double_node(UA_Client *client, UA_UInt16 ns_idx,
                                   const char *node_name, UA_Double *out_value) {
    UA_Variant value;
    UA_Variant_init(&value);

    UA_StatusCode status = UA_Client_readValueAttribute(
        client, UA_NODEID_STRING(ns_idx, (char*)node_name), &value);

    if(status != UA_STATUSCODE_GOOD ||
       !UA_Variant_hasScalarType(&value, &UA_TYPES[UA_TYPES_DOUBLE])) {
        UA_Variant_clear(&value);
        return false;
    }

    *out_value = *(UA_Double*)value.data;
    UA_Variant_clear(&value);
    return true;
}

int main(void) {
    signal(SIGINT, stop_handler);

    UA_Client *client = UA_Client_new();
    UA_ClientConfig_setDefault(UA_Client_getConfig(client));

    UA_StatusCode status = UA_Client_connect(client, ENDPOINT_URL);
    if(status != UA_STATUSCODE_GOOD) {
        fprintf(stderr, "Connection failed: 0x%08x\n", (unsigned)status);
        UA_Client_delete(client);
        return 1;
    }

    UA_UInt16 ns_idx = 0;
    UA_String ns_uri = UA_STRING((char*)NAMESPACE_URI);
    status = UA_Client_NamespaceGetIndex(client, &ns_uri, &ns_idx);
    if(status != UA_STATUSCODE_GOOD) {
        fprintf(stderr, "Namespace lookup failed: 0x%08x\n", (unsigned)status);
        UA_Client_disconnect(client);
        UA_Client_delete(client);
        return 1;
    }

    printf("Connected to %s\n", ENDPOINT_URL);
    printf("Namespace: %s (index %u)\n", NAMESPACE_URI, (unsigned)ns_idx);
    printf("Reading tags... Press Ctrl+C to stop\n");
    fflush(stdout);

    while(running) {
        size_t ok_count = 0;
        UA_Double xmeas1 = 0.0, xmeas2 = 0.0, xmv1 = 0.0;

        for(int i = 1; i <= 41; ++i) {
            char name[16];
            snprintf(name, sizeof(name), "xmeas_%d", i);

            UA_Double value = 0.0;
            if(read_double_node(client, ns_idx, name, &value)) {
                ++ok_count;
                if(i == 1) xmeas1 = value;
                if(i == 2) xmeas2 = value;
            }
        }

        for(int i = 1; i <= 11; ++i) {
            char name[16];
            snprintf(name, sizeof(name), "xmv_%d", i);

            UA_Double value = 0.0;
            if(read_double_node(client, ns_idx, name, &value)) {
                ++ok_count;
                if(i == 1) xmv1 = value;
            }
        }

        printf("ok %zu/52 | xmeas_1=%.5f xmeas_2=%.5f xmv_1=%.5f\n",
               ok_count, xmeas1, xmeas2, xmv1);
        fflush(stdout);

        usleep(POLL_INTERVAL_MS * 1000);
    }

    UA_Client_disconnect(client);
    UA_Client_delete(client);
    return 0;
}

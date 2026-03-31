#define _POSIX_C_SOURCE 200809L

#include "open62541.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>

#define DEFAULT_CSV_PATH "data/df.csv"
#define DEFAULT_ENDPOINT "opc.tcp://0.0.0.0:4840"
#define DEFAULT_NAMESPACE_URI "urn:my-csv-server"
#define DEFAULT_SERVER_NAME "My CSV Server"
#define DEFAULT_ROOT_NAME "root"
#define INTERVAL_MS 5000

typedef struct {
    const char *csv_path;
    FILE *csv;
    char *line;
    size_t line_cap;

    char **columns;
    size_t column_count;
    char **fields;

    size_t *active_indices;
    char **active_names;
    size_t active_count;

    UA_UInt16 ns_idx;
    UA_Boolean loop;
    UA_UInt64 row_index;
    UA_Boolean finished;
} CsvPublisher;

static void trim_eol(char *line) {
    size_t len = strlen(line);
    while(len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
        line[len - 1] = '\0';
        --len;
    }
}

static size_t split_csv_inplace(char *line, char **out_fields, size_t max_fields) {
    if(max_fields == 0)
        return 0;

    size_t count = 0;
    char *cursor = line;
    while(count < max_fields) {
        out_fields[count++] = cursor;
        char *comma = strchr(cursor, ',');
        if(!comma)
            break;
        *comma = '\0';
        cursor = comma + 1;
    }
    return count;
}

static UA_Boolean is_skip_column(const char *column) {
    static const char *skip_cols[] = {"ts", "time", "timestamp", "sample", "run_id"};
    for(size_t i = 0; i < sizeof(skip_cols) / sizeof(skip_cols[0]); ++i) {
        if(!strcasecmp(column, skip_cols[i]))
            return true;
    }
    return false;
}

static UA_StatusCode load_csv_header(CsvPublisher *pub) {
    pub->csv = fopen(pub->csv_path, "r");
    if(!pub->csv) {
        perror("fopen");
        return UA_STATUSCODE_BADNOTFOUND;
    }

    ssize_t read = getline(&pub->line, &pub->line_cap, pub->csv);
    if(read < 0) {
        fprintf(stderr, "CSV file is empty: %s\n", pub->csv_path);
        return UA_STATUSCODE_BADUNEXPECTEDERROR;
    }

    trim_eol(pub->line);
    if(pub->line[0] == '\0') {
        fprintf(stderr, "CSV header is empty: %s\n", pub->csv_path);
        return UA_STATUSCODE_BADUNEXPECTEDERROR;
    }

    size_t estimated = 1;
    for(char *p = pub->line; *p; ++p) {
        if(*p == ',')
            ++estimated;
    }

    char **header_fields = (char**)calloc(estimated, sizeof(char*));
    if(!header_fields)
        return UA_STATUSCODE_BADOUTOFMEMORY;

    pub->column_count = split_csv_inplace(pub->line, header_fields, estimated);
    if(pub->column_count == 0) {
        free(header_fields);
        return UA_STATUSCODE_BADUNEXPECTEDERROR;
    }

    pub->columns = (char**)calloc(pub->column_count, sizeof(char*));
    pub->fields = (char**)calloc(pub->column_count, sizeof(char*));
    pub->active_indices = (size_t*)calloc(pub->column_count, sizeof(size_t));
    pub->active_names = (char**)calloc(pub->column_count, sizeof(char*));

    if(!pub->columns || !pub->fields || !pub->active_indices || !pub->active_names) {
        free(header_fields);
        return UA_STATUSCODE_BADOUTOFMEMORY;
    }

    for(size_t i = 0; i < pub->column_count; ++i) {
        pub->columns[i] = strdup(header_fields[i]);
        if(!pub->columns[i]) {
            free(header_fields);
            return UA_STATUSCODE_BADOUTOFMEMORY;
        }
        if(!is_skip_column(pub->columns[i])) {
            pub->active_indices[pub->active_count] = i;
            pub->active_names[pub->active_count] = pub->columns[i];
            ++pub->active_count;
        }
    }

    free(header_fields);

    if(pub->active_count == 0) {
        fprintf(stderr, "No publishable columns in CSV\n");
        return UA_STATUSCODE_BADUNEXPECTEDERROR;
    }

    return UA_STATUSCODE_GOOD;
}

static UA_Boolean read_next_row(CsvPublisher *pub, size_t *field_count) {
    while(true) {
        ssize_t read = getline(&pub->line, &pub->line_cap, pub->csv);
        if(read < 0) {
            if(!pub->loop)
                return false;

            rewind(pub->csv);
            if(getline(&pub->line, &pub->line_cap, pub->csv) < 0)
                return false;
            continue;
        }

        trim_eol(pub->line);
        if(pub->line[0] == '\0')
            continue;

        *field_count = split_csv_inplace(pub->line, pub->fields, pub->column_count);
        return true;
    }
}

static void publish_next_row(UA_Server *server, void *data) {
    CsvPublisher *pub = (CsvPublisher*)data;
    if(pub->finished)
        return;

    size_t field_count = 0;
    if(!read_next_row(pub, &field_count)) {
        pub->finished = true;
        return;
    }

    for(size_t i = 0; i < pub->active_count; i++) {
        size_t column_index = pub->active_indices[i];
        if(column_index >= field_count)
            continue;

        char *raw = pub->fields[column_index];
        if(!raw || raw[0] == '\0')
            continue;

        errno = 0;
        char *endptr = NULL;
        double value = strtod(raw, &endptr);
        if(endptr == raw || errno == ERANGE || *endptr != '\0')
            continue;

        UA_Variant variant;
        UA_Variant_init(&variant);
        UA_Variant_setScalar(&variant, &value, &UA_TYPES[UA_TYPES_DOUBLE]);

        UA_StatusCode write_status = UA_Server_writeValue(
            server, UA_NODEID_STRING(pub->ns_idx, pub->active_names[i]), variant);

        if(write_status != UA_STATUSCODE_GOOD) {
            UA_LOG_WARNING(UA_Log_Stdout, UA_LOGCATEGORY_SERVER,
                           "Write failed for %s: 0x%08x",
                           pub->active_names[i], (unsigned)write_status);
        }
    }

    ++pub->row_index;
}

static UA_StatusCode create_opcua_nodes(UA_Server *server, CsvPublisher *pub, const char *namespace_uri,
                                        const char *root_name) {
    pub->ns_idx = UA_Server_addNamespace(server, namespace_uri);

    UA_ObjectAttributes root_attr = UA_ObjectAttributes_default;
    root_attr.displayName = UA_LOCALIZEDTEXT("en-US", (char*)root_name);

    UA_StatusCode status = UA_Server_addObjectNode(
        server,
        UA_NODEID_STRING(pub->ns_idx, (char*)root_name),
        UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER),
        UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES),
        UA_QUALIFIEDNAME(pub->ns_idx, (char*)root_name),
        UA_NODEID_NUMERIC(0, UA_NS0ID_BASEOBJECTTYPE),
        root_attr,
        NULL,
        NULL);

    if(status != UA_STATUSCODE_GOOD)
        return status;

    for(size_t i = 0; i < pub->active_count; ++i) {
        UA_VariableAttributes attr = UA_VariableAttributes_default;
        attr.displayName = UA_LOCALIZEDTEXT("en-US", pub->active_names[i]);
        attr.accessLevel = UA_ACCESSLEVELMASK_READ | UA_ACCESSLEVELMASK_WRITE;
        attr.userAccessLevel = UA_ACCESSLEVELMASK_READ;

        UA_Double initial_value = 0.0;
        UA_Variant_setScalar(&attr.value, &initial_value, &UA_TYPES[UA_TYPES_DOUBLE]);

        status = UA_Server_addVariableNode(
            server,
            UA_NODEID_STRING(pub->ns_idx, pub->active_names[i]),
            UA_NODEID_STRING(pub->ns_idx, (char*)root_name),
            UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES),
            UA_QUALIFIEDNAME(pub->ns_idx, pub->active_names[i]),
            UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE),
            attr,
            NULL,
            NULL);

        if(status != UA_STATUSCODE_GOOD)
            return status;
    }

    return UA_STATUSCODE_GOOD;
}

static void cleanup_publisher(CsvPublisher *pub) {
    if(pub->csv)
        fclose(pub->csv);

    free(pub->line);
    free(pub->fields);
    free(pub->active_indices);
    free(pub->active_names);

    if(pub->columns) {
        for(size_t i = 0; i < pub->column_count; i++)
            free(pub->columns[i]);
    }
    free(pub->columns);
}

int main(int argc, char **argv) {
    const char *csv_path = (argc > 1) ? argv[1] : DEFAULT_CSV_PATH;
    const char *endpoint = DEFAULT_ENDPOINT;
    const char *namespace_uri = DEFAULT_NAMESPACE_URI;
    const char *root_name = DEFAULT_ROOT_NAME;

    double interval_ms = INTERVAL_MS;

    UA_Boolean loop = true;

    CsvPublisher pub;
    memset(&pub, 0, sizeof(pub));
    pub.csv_path = csv_path;
    pub.loop = loop;

    UA_StatusCode status = load_csv_header(&pub);
    if(status != UA_STATUSCODE_GOOD) {
        cleanup_publisher(&pub);
        return EXIT_FAILURE;
    }

    UA_UInt16 port = 4840;
    UA_String endpoint_str = UA_STRING((char*)endpoint);
    UA_String host;
    UA_String path;
    UA_String_init(&host);
    UA_String_init(&path);
    if(UA_parseEndpointUrl(&endpoint_str, &host, &port, &path) != UA_STATUSCODE_GOOD) {
        fprintf(stderr, "Invalid OPC_UA_SERVER_ENDPOINT: %s\n", endpoint);
        cleanup_publisher(&pub);
        return EXIT_FAILURE;
    }

    UA_Boolean host_is_any = (host.length == 7 && memcmp(host.data, "0.0.0.0", 7) == 0);
    UA_Boolean override_serverurl = !(host_is_any && port == 4840);

    UA_Server *server = UA_Server_new();
    UA_ServerConfig *config = UA_Server_getConfig(server);

    if(override_serverurl) {
        if(config->serverUrls) {
            UA_Array_delete(config->serverUrls, config->serverUrlsSize, &UA_TYPES[UA_TYPES_STRING]);
            config->serverUrls = NULL;
            config->serverUrlsSize = 0;
        }
        config->serverUrls = (UA_String*)UA_Array_new(1, &UA_TYPES[UA_TYPES_STRING]);
        if(!config->serverUrls) {
            fprintf(stderr, "Failed to allocate server URL array\n");
            UA_Server_delete(server);
            cleanup_publisher(&pub);
            return EXIT_FAILURE;
        }
        config->serverUrls[0] = UA_STRING_ALLOC(endpoint);
        config->serverUrlsSize = 1;
    }

    status = create_opcua_nodes(server, &pub, namespace_uri, root_name);
    if(status != UA_STATUSCODE_GOOD) {
        fprintf(stderr, "Failed to create OPC UA nodes: 0x%08x\n", (unsigned)status);
        UA_Server_delete(server);
        cleanup_publisher(&pub);
        return EXIT_FAILURE;
    }

    publish_next_row(server, &pub);

    UA_UInt64 callback_id = 0;
    status = UA_Server_addRepeatedCallback(server, publish_next_row, &pub,
                                           interval_ms, &callback_id);
    if(status != UA_STATUSCODE_GOOD) {
        fprintf(stderr, "UA_Server_addRepeatedCallback failed: 0x%08x\n", (unsigned)status);
        UA_Server_delete(server);
        cleanup_publisher(&pub);
        return EXIT_FAILURE;
    }

    printf("OPC UA server is running\n");
    printf("Endpoint (configured): %s\n", endpoint);
    printf("Client connect example: opc.tcp://<server-ip>:%u\n", (unsigned)port);
    printf("Namespace URI: %s | ns index: %u\n", namespace_uri, (unsigned)pub.ns_idx);
    printf("CSV: %s\n", csv_path);
    printf("Tags published: %zu | interval: %.3f sec | loop: %s\n",
           pub.active_count, interval_ms / 1000.0, loop ? "true" : "false");
    printf("Press Ctrl+C to stop\n");

    status = UA_Server_runUntilInterrupt(server);

    UA_Server_removeCallback(server, callback_id);
    UA_Server_delete(server);
    cleanup_publisher(&pub);

    return (status == UA_STATUSCODE_GOOD) ? EXIT_SUCCESS : EXIT_FAILURE;
}

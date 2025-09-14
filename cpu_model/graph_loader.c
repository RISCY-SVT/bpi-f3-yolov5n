#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "csinn/csi_nn.h"
#include "shl_utils.h"

// External functions from model.c
void *csinn_(char *params_base);

// Load binary file
static char* get_binary_from_file(const char *filename, int *size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return NULL;
    }
    
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    rewind(fp);
    
    char *buffer = (char*)malloc(file_size);
    if (!buffer) {
        fclose(fp);
        return NULL;
    }
    
    if (fread(buffer, 1, file_size, fp) != file_size) {
        free(buffer);
        fclose(fp);
        return NULL;
    }
    
    fclose(fp);
    if (size) *size = file_size;
    return buffer;
}

// Create graph from params file
void *create_graph(char *params_path) {
    char *params = get_binary_from_file(params_path, NULL);
    if (params == NULL) {
        return NULL;
    }

    char *suffix = params_path + (strlen(params_path) - 7);
    if (strcmp(suffix, ".params") == 0) {
        // create general graph
        return csinn_(params);
    }

    suffix = params_path + (strlen(params_path) - 3);
    if (strcmp(suffix, ".bm") == 0) {
        struct shl_bm_sections *section = (struct shl_bm_sections *)(params + 4128);
        if (section->graph_offset) {
            return csinn_import_binary_model(params);
        } else {
            return csinn_(params + section->params_offset * 4096);
        }
    } else {
        return NULL;
    }
}

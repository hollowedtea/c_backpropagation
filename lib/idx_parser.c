#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    FILE *file;
    uint8_t data_type;
    uint8_t dimension_number;
    uint32_t *dimension_sizes;
} idx_instance;

idx_instance open_idx_instance(const char *filename) {
    idx_instance reader = {};
    
    reader.file = fopen(filename, "rb");
    if (!reader.file) {
        perror("Error opening file! \n");
    }

    fseek(reader.file, 2, SEEK_CUR);
    fread(&reader.data_type, 1, 1, reader.file);
    fread(&reader.dimension_number, 1, 1, reader.file);

    reader.dimension_sizes = (uint32_t *)calloc(0, sizeof(uint32_t) * reader.dimension_number);
    
    uint8_t byte_temp;
    for (int i = 0; i < reader.dimension_number; i++) {
        for (int j = 0; j < 4; j++) {
            fread(&byte_temp, 1, 1, reader.file);
            reader.dimension_sizes[i] += byte_temp << 8*(3 - j);
        }
    }

    return reader;
}

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "matrix.h"

#define IDX_TYPE_UINT8 0x08
#define MNIST_IMAGE_ROWS 28u
#define MNIST_IMAGE_COLUMNS 28u
#define MNIST_CLASS_COUNT 10u

typedef struct {
    FILE *file;
    uint8_t data_type;
    uint8_t dimension_number;
    uint32_t *dimension_sizes;
} idx_instance;

void close_idx_instance(idx_instance *reader);

static int read_exact(FILE *file, void *buffer, size_t size) {
    return file != NULL && fread(buffer, 1, size, file) == size;
}

static int read_be_u32(FILE *file, uint32_t *value) {
    uint8_t bytes[4];

    if (value == NULL || !read_exact(file, bytes, sizeof bytes)) {
        return 0;
    }

    *value = ((uint32_t)bytes[0] << 24) |
             ((uint32_t)bytes[1] << 16) |
             ((uint32_t)bytes[2] << 8)  |
             (uint32_t)bytes[3];

    return 1;
}

// Opens an IDX file, reads its header, and leaves the file positioned at
// the beginning of the data payload. On failure, returns a zeroed instance.
idx_instance open_idx_instance(const char *filename) {
    idx_instance reader = {0};
    uint8_t magic[2];

    if (filename == NULL) {
        fprintf(stderr, "IDX filename is NULL.\n");
        return reader;
    }

    reader.file = fopen(filename, "rb");
    if (reader.file == NULL) {
        perror("Error opening IDX file");
        return reader;
    }

    if (!read_exact(reader.file, magic, sizeof magic) ||
        magic[0] != 0 || magic[1] != 0 ||
        !read_exact(reader.file, &reader.data_type, 1) ||
        !read_exact(reader.file, &reader.dimension_number, 1)) {
        fprintf(stderr, "Invalid or incomplete IDX header: %s\n", filename);
        close_idx_instance(&reader);
        return (idx_instance){0};
    }

    if (reader.dimension_number == 0) {
        fprintf(stderr, "IDX file has zero dimensions: %s\n", filename);
        close_idx_instance(&reader);
        return (idx_instance){0};
    }

    reader.dimension_sizes = calloc(
        reader.dimension_number,
        sizeof *reader.dimension_sizes
    );

    if (reader.dimension_sizes == NULL) {
        perror("Unable to allocate IDX dimensions");
        close_idx_instance(&reader);
        return (idx_instance){0};
    }

    for (size_t i = 0; i < reader.dimension_number; ++i) {
        if (!read_be_u32(reader.file, &reader.dimension_sizes[i])) {
            fprintf(stderr, "Incomplete IDX dimensions: %s\n", filename);
            close_idx_instance(&reader);
            return (idx_instance){0};
        }
    }

    return reader;
}

// Reads raw bytes from the IDX payload.
int read_idx_bytes(idx_instance *reader, void *buffer, size_t byte_count) {
    if (reader == NULL || reader->file == NULL ||
        (buffer == NULL && byte_count != 0)) {
        return 0;
    }

    return read_exact(reader->file, buffer, byte_count);
}

void close_idx_instance(idx_instance *reader) {
    if (reader == NULL) {
        return;
    }

    if (reader->file != NULL) {
        fclose(reader->file);
    }

    free(reader->dimension_sizes);
    *reader = (idx_instance){0};
}

// Frees an array of matrices, including each matrix's content buffer.
void free_matrix_array(matrix *items, size_t count) {
    if (items == NULL) {
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        free(items[i].content);
    }

    free(items);
}

int load_mnist_train(
    const char *images_filename,
    const char *labels_filename,
    matrix **images_out,
    matrix **labels_out,
    size_t *count_out
) {
    idx_instance images = {0};
    idx_instance labels = {0};
    matrix *train_images = NULL;
    matrix *train_labels = NULL;
    uint8_t *pixel_buffer = NULL;
    size_t sample_count = 0;
    size_t image_rows;
    size_t image_columns;
    size_t image_size;

    if (images_out == NULL || labels_out == NULL || count_out == NULL) {
        fprintf(stderr, "MNIST output arguments must not be NULL.\n");
        return 0;
    }

    *images_out = NULL;
    *labels_out = NULL;
    *count_out = 0;

    images = open_idx_instance(images_filename);
    if (images.file == NULL) {
        return 0;
    }

    labels = open_idx_instance(labels_filename);
    if (labels.file == NULL) {
        close_idx_instance(&images);
        return 0;
    }

    // MNIST images: unsigned byte, 3 dimensions: count, rows, columns.
    if (images.data_type != IDX_TYPE_UINT8 ||
        images.dimension_number != 3 ||
        images.dimension_sizes[0] == 0 ||
        images.dimension_sizes[1] != MNIST_IMAGE_ROWS ||
        images.dimension_sizes[2] != MNIST_IMAGE_COLUMNS) {
        fprintf(stderr, "Invalid MNIST image IDX header.\n");
        goto failure;
    }

    // MNIST labels: unsigned byte, 1 dimension: count.
    if (labels.data_type != IDX_TYPE_UINT8 ||
        labels.dimension_number != 1 ||
        labels.dimension_sizes[0] != images.dimension_sizes[0]) {
        fprintf(stderr, "Invalid MNIST label IDX header.\n");
        goto failure;
    }

    sample_count = (size_t)images.dimension_sizes[0];
    image_rows = (size_t)images.dimension_sizes[1];
    image_columns = (size_t)images.dimension_sizes[2];

    if (image_rows > SIZE_MAX / image_columns) {
        fprintf(stderr, "MNIST image dimensions overflow size_t.\n");
        goto failure;
    }

    image_size = image_rows * image_columns;

    if (image_size > SIZE_MAX / sizeof(float) ||
        sample_count > SIZE_MAX / sizeof(matrix)) {
        fprintf(stderr, "MNIST allocation size overflow.\n");
        goto failure;
    }

    train_images = calloc(sample_count, sizeof *train_images);
    train_labels = calloc(sample_count, sizeof *train_labels);
    pixel_buffer = malloc(image_size);

    if (train_images == NULL || train_labels == NULL || pixel_buffer == NULL) {
        perror("Unable to allocate MNIST data");
        goto failure;
    }

    for (size_t i = 0; i < sample_count; ++i) {
        uint8_t label;

        if (!read_idx_bytes(&images, pixel_buffer, image_size)) {
            fprintf(stderr, "Unable to read MNIST image %zu.\n", i);
            goto failure;
        }

        if (!read_idx_bytes(&labels, &label, 1)) {
            fprintf(stderr, "Unable to read MNIST label %zu.\n", i);
            goto failure;
        }

        if (label >= MNIST_CLASS_COUNT) {
            fprintf(stderr, "Invalid MNIST label %u at sample %zu.\n",
                    (unsigned)label, i);
            goto failure;
        }

        train_images[i].rows = image_size;
        train_images[i].columns = 1;
        train_images[i].content = malloc(image_size * sizeof(float));

        train_labels[i].rows = MNIST_CLASS_COUNT;
        train_labels[i].columns = 1;
        train_labels[i].content = calloc(
            MNIST_CLASS_COUNT,
            sizeof(float)
        );

        if (train_images[i].content == NULL ||
            train_labels[i].content == NULL) {
            perror("Unable to allocate MNIST sample");
            goto failure;
        }

        for (size_t j = 0; j < image_size; ++j) {
            train_images[i].content[j] = (float)pixel_buffer[j] / 255.0f;
        }

        train_labels[i].content[label] = 1.0f;
    }

    free(pixel_buffer);
    close_idx_instance(&images);
    close_idx_instance(&labels);

    *images_out = train_images;
    *labels_out = train_labels;
    *count_out = sample_count;

    return 1;

failure:
    free(pixel_buffer);
    free_matrix_array(train_images, sample_count);
    free_matrix_array(train_labels, sample_count);
    close_idx_instance(&images);
    close_idx_instance(&labels);

    return 0;
}

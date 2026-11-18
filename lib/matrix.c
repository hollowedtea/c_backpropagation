#include "matrix.h"
#include "string.h"

#include <stdio.h>

void matrix_init(matrix *m, size_t rows, size_t columns) {
    m->rows = rows; m->columns = columns;

    size_t size = rows * columns * sizeof(float);
    m->content = (float *)malloc(size);

    memset(m->content, 0, size);
}

void matrix_multiply(matrix m, matrix n, matrix *result) {
    if (m.columns != n.rows) {
        printf("Could multiply matrices %zux%zu * %zux%zu", m.rows, m.columns, n.rows, n.columns);
        return;
    }

    matrix_init(result, m.rows, n.columns);

    float sum = .0;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < n.columns; j++){
            for (int k = 0; k < m.columns; k++) {
                sum += (m.content)[i * m.columns + k] * (n.content)[k * n.columns + j];
            }

            result->content[i * n.columns + j] = sum;
            sum = .0;
        }
    }
}

void matrix_add(matrix m, matrix n, matrix *result) {
    if ((m.columns != n.columns) || (m.rows != n.rows)) {
        printf("Could add matrices %zux%zu * %zux%zu", n.rows, n.columns, m.rows, m.columns);
        return;
    }

    result->rows = m.rows; result->columns = m.columns;
    for (int i = 0; i < m.columns * m.rows; i++) {
        result->content[i] = n.content[i] + m.content[i];
    }
}

void matrix_transpose(matrix *m) {
    size_t r = m->rows;
    size_t c = m->columns;

    m->rows = c;
    m->columns = r;

    float *temp_content = (float *)malloc(sizeof(float) * r * c);
    memcpy(temp_content, m->content, sizeof(float) * r * c);

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            m->content[j * r + i] = temp_content[i * c + j];
        }
    }

    free(temp_content);
}

void matrix_copy(matrix m, matrix *c) {
    *c = m;
    c->content = malloc(sizeof(float) * c->rows * c->columns);

    memcpy(c->content, m.content, sizeof(float) * c->rows * c->columns);
}

void matrix_scale(float c, matrix *m) {
    for (int i = 0; i < m->columns * m->rows; i++) {
        m->content[i] *= c;
    }
}

#include "matrix.h"

void matrix_init(matrix *m, size_t rows, size_t columns) {
    m->rows = rows; m->columns = columns;
    m->content = (float *)malloc(sizeof(float) * rows * columns);
}

void matrix_multiply(matrix m, matrix n, matrix *result) {
    if (m.columns != n.rows) { return; }

    matrix_init(result, m.rows, n.columns);

    float sum = .0;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < n.columns; j++){
            for (int k = 0; k < m.columns; k++) {
                sum += (m.content)[i * m.columns + k] * (n.content)[k * n.columns + j];
            }

            result->content[i * m.rows + j] = sum;
            sum = .0;
        }
    }
}

void matrix_add(matrix m, matrix n, matrix *result) {
    if ((m.columns != n.columns) || (m.rows != n.rows)) {
        return;
    }

    result->rows = m.rows; result->columns = m.columns;
    for (int i = 0; i < m.columns * m.rows; i++) {
        result->content[i] = n.content[i] + m.content[i];
    }
}

void matrix_transpose(matrix *m) {
    if (m->rows != m->columns) { return; }
    size_t m_size = m->rows;

    float tmp;
    for (int i = 1; i < m_size; i++) {
        for (int j = 0; j < i; j++) {
            tmp = m->content[j * m_size + i];
            m->content[j * m_size + i] = m->content[i * m_size + j];
            m->content[i * m_size + j] = tmp;
        }
    }
}

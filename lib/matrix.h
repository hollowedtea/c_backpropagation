#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifndef _MATRIX
#define _MATRIX

typedef struct {
    size_t rows, columns;
    float *content;
} matrix;

void matrix_init(matrix *m, size_t rows, size_t columns);
void matrix_multiply(matrix m, matrix n, matrix *result);
void matrix_add(matrix m, matrix n, matrix *result);
void matrix_transpose(matrix *m);

#endif

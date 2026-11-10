#include "perceptron.h"
#include "matrix.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

perceptron perceptron_init(size_t layer_count, size_t *layer_sizes, float (**activation_functions)(float *, size_t, size_t)) {
    perceptron p = {};
    p.layers = layer_count;

    p.matrices = (matrix *)malloc(sizeof(matrix) * (layer_count - 1));
    p.biases = (matrix *)malloc(sizeof(matrix) * (layer_count - 1));
    p.activation_functions = (float (**)(float *, size_t, size_t))malloc(sizeof(float (*)(float *, size_t, size_t)) * (layer_count - 1));

    for (int i = 0; i < layer_count - 1; i++) {
        matrix_init(&(p.matrices[i]), layer_sizes[i], layer_sizes[i + 1]);
        matrix_init(&(p.biases[i]), 1, layer_sizes[i]);

        p.activation_functions[i] = activation_functions[i];
    }

    return p;
}

matrix forward_propagation(perceptron *p, matrix input_layer) {
    matrix m1 = {};
    memcpy(&m1, &input_layer, sizeof(input_layer));

    matrix m2 = {};

    matrix *previous = &m1;
    matrix *next = &m2;

    for (int i = 0; i < p->layers - 1; i++) {
        matrix_multiply(*previous, p->matrices[i], next);
        matrix_add(*next, p->biases[i], next);

        float next_activated[next->rows * next->columns];

        for (int j = 0; j < next->rows * next->columns; j++) {
            next_activated[j] = (*p->activation_functions[i])(next->content, next->rows * next->columns, j);
        }
        for (int j = 0; j < next->rows * next->columns; j++) {
            next->content[j] = next_activated[j];
        }

        matrix *temp = previous;
        previous = next;
        next = temp;
    }

    return *previous;
}

void back_propagation(perceptron *p, matrix train_data) {

}

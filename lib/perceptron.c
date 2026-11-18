#include "perceptron.h"
#include "matrix.h"
#include "cost_functions.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

float min(float a, float b) {
    if (a > b) {
        return b;
    }

    return a;
}

float max(float a, float b) {
    if (a > b) {
        return a;
    }

    return b;
}

perceptron perceptron_init(size_t layer_count, size_t *layer_sizes, activation_function *activation_functions, cost_function cost_func, float weight_range[2]) {
    if (layer_count < 2) {
        printf("Cannot create one with specified layer count\n");
        return (perceptron){};
    }

    perceptron p = {};
    p.layers = layer_count;

    p.matrices = (matrix *)malloc(sizeof(matrix) * (layer_count - 1));
    p.biases = (matrix *)malloc(sizeof(matrix) * (layer_count - 1));
    p.activation_functions = (activation_function *)malloc(sizeof(activation_function) * (layer_count - 1));
    p.cost_func = cost_func;

    // Initialize weights and biases
    for (int i = 0; i < layer_count - 1; i++) {
        matrix_init(&(p.matrices[i]), layer_sizes[i + 1], layer_sizes[i]);
        matrix_init(&(p.biases[i]), 1, layer_sizes[i + 1]);

        p.activation_functions[i] = activation_functions[i];
    }

    // Fill up the data
    float a = min(weight_range[0], weight_range[1]);
    float b = max(weight_range[0], weight_range[1]);
    
    for (int i = 0; i < layer_count - 1; i++) {
        for (int j = 0; j < p.matrices[i].columns * p.matrices[i].rows; j++) {
            p.matrices[i].content[j] = (b - a)*(rand() / (float)RAND_MAX) + a;
        }

        memset(p.biases[i].content, 0, sizeof(int) * layer_sizes[i + 1]);
    }

    return p;
}

matrix forward_propagation(perceptron *p, matrix input_layer) {
    if (!(input_layer.columns == 1 && input_layer.rows == p->matrices[0].columns)) {
        printf("Input layer has non-valid dimensions\n");
        exit(1);
    }

    matrix cur = input_layer;
    int owns_cur = 0;

    for (size_t i = 0; i < p->layers - 1; i++) {
        matrix next = {0};
        matrix_multiply(p->matrices[i], cur, &next);
        if (owns_cur) free(cur.content);

        matrix_add(next, p->biases[i], &next);

        size_t n = next.rows * next.columns;
        float *activated = malloc(sizeof(float) * n);
        for (size_t j = 0; j < n; j++) {
            activated[j] = p->activation_functions[i].function(next.content, n, j);
        }

        memcpy(next.content, activated, sizeof(float) * n);
        free(activated);

        cur = next;
        owns_cur = 1;
    }
    return cur;
}

void batch_fit(perceptron *p, matrix batch, matrix target) {
    if (batch.rows != p->matrices->rows) {
        printf("The data don't fit!\n");
        return;
    }

    // initializing the layer node values matrices
    matrix **preactivations = (matrix **)malloc(sizeof(matrix *) * batch.columns);
    matrix **activations = (matrix **)malloc(sizeof(matrix *) * (batch.columns));

    for (int i = 0; i < batch.columns; i++) {
        // divide train data into individual input layers
        preactivations[i] = (matrix *)malloc(sizeof(matrix) * p->layers);
        activations[i] = (matrix *)malloc(sizeof(matrix) * p->layers);

        matrix_init(&(activations[i][0]), 1, batch.rows);
        matrix_init(&(preactivations[i][0]), 1, batch.rows);

        for (int j = 0; j < batch.rows; j++) {
            activations[i][0].content[j] = batch.content[batch.columns * j + i];
        }

        // forward pass to fill in the node value data
        for (int j = 0; j < p->layers - 1; j++) {
            // computing preactivated layer
            matrix_multiply(p->matrices[j], activations[i][j], &(preactivations[i][j + 1]));
            matrix_add(preactivations[i][j + 1], p->biases[j], &(preactivations[i][j + 1]));

            // computig activated layer
            size_t n = preactivations[i][j + 1].rows * preactivations[i][j + 1].columns;

            matrix_init(&(activations[i][j + 1]), 1, p->matrices[j].columns);


            for (int k = 0; k < n; k++) {
                activations[i][j + 1].content[k] = p->activation_functions[j].function(preactivations[i][j + 1].content, n, k);
            }
        }
    }


    // compute gradients...
}

void back_propagation(perceptron *p, matrix train_data_input, matrix train_data_target, size_t batch_size, size_t epochs) {
    
}









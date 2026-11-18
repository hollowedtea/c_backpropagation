#include "perceptron.h"
#include "matrix.h"
#include "cost_functions.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static float min(float a, float b) {
    if (a > b) {
        return b;
    }

    return a;
}

static float max(float a, float b) {
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
        matrix_init(&(p.biases[i]), layer_sizes[i + 1], 1);

        p.activation_functions[i] = activation_functions[i];
    }

    // Fill up the data
    float a = min(weight_range[0], weight_range[1]);
    float b = max(weight_range[0], weight_range[1]);
    
    for (int i = 0; i < layer_count - 1; i++) {
        for (int j = 0; j < p.matrices[i].columns * p.matrices[i].rows; j++) {
            p.matrices[i].content[j] = (b - a)*(rand() / (float)RAND_MAX) + a;
        }

        memset(p.biases[i].content, 0, sizeof(float) * layer_sizes[i + 1]);
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

static void batch_fit(perceptron *p, matrix *batch, matrix *target, size_t batch_size, float learning_rate) {
    // initializing the layer node values matrices
    matrix **preactivations = (matrix **)malloc(sizeof(matrix *) * batch_size);
    matrix **activations = (matrix **)malloc(sizeof(matrix *) * batch_size);

    for (int i = 0; i < batch_size; i++) {
        // divide train data into individual input layers
        preactivations[i] = (matrix *)malloc(sizeof(matrix) * p->layers);
        activations[i] = (matrix *)malloc(sizeof(matrix) * p->layers);

        matrix_init(&(activations[i][0]), batch[i].rows, 1);
        matrix_init(&(preactivations[i][0]), batch[i].rows, 1);

        for (int j = 0; j < batch[i].rows; j++) {
            activations[i][0].content[j] = batch[i].content[j];
        }

        // forward pass to fill in the node value data
        for (int j = 0; j < p->layers - 1; j++) {
            // computing preactivated layer
            matrix_multiply(p->matrices[j], activations[i][j], &(preactivations[i][j + 1]));
            matrix_add(preactivations[i][j + 1], p->biases[j], &(preactivations[i][j + 1]));

            // computig activated layer
            size_t n = preactivations[i][j + 1].rows * preactivations[i][j + 1].columns;

            matrix_init(&(activations[i][j + 1]), p->matrices[j].rows, 1);


            for (int k = 0; k < n; k++) {
                activations[i][j + 1].content[k] = p->activation_functions[j].function(preactivations[i][j + 1].content, n, k);
            }
        }
    }


    // compute gradients...

    size_t l = p->layers - 1;

    matrix *weight_gradients = (matrix *)malloc(sizeof(matrix) * l);
    matrix *bias_gradients = (matrix *)malloc(sizeof(matrix) * l);

    //memset(weight_gradients, 0, sizeof(matrix) * l);
    //memset(bias_gradients, 0, sizeof(matrix) * l);

    for (int i = 0; i < l; i++) {
        matrix_init(&weight_gradients[i], p->matrices[i].rows, p->matrices[i].columns);
        matrix_init(&bias_gradients[i], p->matrices[i].rows, 1);
    }

    for (int i = 0; i < batch_size; i++) {
        // evil math
        matrix jacobian = p->activation_functions[l - 1].jacobian(preactivations[i][l].content, p->matrices[l - 1].rows);
        matrix cost_gradient = p->cost_func.gradient(activations[i][l].content, target[i].content, p->matrices[l - 1].rows);

        matrix_transpose(&jacobian);

        matrix delta = {0};
        matrix_multiply(jacobian, cost_gradient, &delta);
        // make initial delta flip signs to automatically do it
        // to all of the gradients due to recursion
        matrix_scale(-1, &delta); 

        matrix_add(bias_gradients[l - 1], delta, &bias_gradients[l - 1]);

        matrix a = {0};

        matrix_copy(activations[i][l - 1], &a);
        matrix_transpose(&a);

        matrix weight_grad = {0};

        matrix_multiply(delta, a, &weight_grad);

        matrix_add(weight_gradients[l - 1], weight_grad, &weight_gradients[l - 1]);

        free(jacobian.content);
        free(cost_gradient.content);
        free(a.content);
        free(weight_grad.content);

        for (int j = p->layers - 2; j > 0; j--) {
            // much more of evil math
            matrix delta_prev = {0};

            matrix jacobian = p->activation_functions[j - 1].jacobian(preactivations[i][j].content, p->matrices[j - 1].rows);
            matrix_transpose(&jacobian);

            matrix W = {0};
            matrix_copy(p->matrices[j], &W);
            matrix_transpose(&W);

            matrix delta_intermediate = {0};
            matrix_multiply(W, delta, &delta_intermediate);
            // d^(l-1) = J^T_{a^(l-1)} * (W^l)^T * d^l
            matrix_multiply(jacobian, delta_intermediate, &delta_prev);

            free(delta_intermediate.content);
            free(delta.content);
    
            //matrix_copy(delta_prev, &delta);
            //free(delta_prev.content);

            delta = delta_prev;

            // grad_{b^l}(E) = d^l
            matrix_add(bias_gradients[j - 1], delta, &bias_gradients[j - 1]);

            matrix a = {0};

            matrix_copy(activations[i][j - 1], &a);
            matrix_transpose(&a);

            matrix weight_grad = {0};

            matrix_multiply(delta, a, &weight_grad);

            // grad_{W^l}(E) = d^l * (a^(l-1))^T
            matrix_add(weight_gradients[j - 1], weight_grad, &weight_gradients[j - 1]);

            free(jacobian.content);
            free(a.content);
            free(weight_grad.content);
            free(W.content);
        }

        free(delta.content);
    }

    // Freeing forward pass data
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < p->layers; j++) {
            free(preactivations[i][j].content);
            free(activations[i][j].content);
        }
    }

    for (int i = 0; i < batch_size; i++) {
        free(preactivations[i]);
        free(activations[i]);
    }

    free(preactivations);
    free(activations);

    // Taking mean and applying learning rate
    for (int i = 0; i < l; i++) {
        float nc = learning_rate/((float)batch_size);
        matrix_scale(nc, &weight_gradients[i]);
        matrix_scale(nc, &bias_gradients[i]);
    }

    // Applying gradients
    for (int i = 0; i < l; i++) {
        matrix_add(p->matrices[i], weight_gradients[i], &p->matrices[i]);
        matrix_add(p->biases[i], bias_gradients[i], &p->biases[i]);
    }

    // freeing gradients my beloved
    for (int i = 0; i < l; i++) {
        free(weight_gradients[i].content);
        free(bias_gradients[i].content);
    }

    free(weight_gradients);
    free(bias_gradients);
}

// Layout: every COLUMN of train_data_input / train_data_target is one sample
void back_propagation(perceptron *p, matrix train_data_input, matrix train_data_target,
                      size_t batch_size, float learning_rate, size_t epochs) {
    size_t samples  = train_data_input.columns;
    size_t features = train_data_input.rows;
    size_t outputs  = train_data_target.rows;

    if (samples == 0 || batch_size == 0 || samples != train_data_target.columns ||
        features != p->matrices[0].columns || outputs != p->matrices[p->layers - 2].rows) {
        printf("The data don't fit!\n");
        return;
    }
    if (batch_size > samples) batch_size = samples;

    matrix *batch = (matrix *)malloc(sizeof(matrix) * batch_size);
    matrix *target = (matrix *)malloc(sizeof(matrix) * batch_size);
    size_t *order = (size_t *)malloc(sizeof(size_t) * samples);

    for (size_t k = 0; k < batch_size; k++) {
        matrix_init(&batch[k], features, 1);
        matrix_init(&target[k], outputs, 1);
    }
    for (size_t s = 0; s < samples; s++) order[s] = s;

    for (size_t e = 0; e < epochs; e++) {
        // new sample order every epoch (Fisher-Yates), so batches differ between epochs
        for (size_t s = samples - 1; s > 0; s--) {
            size_t r = (size_t)rand() % (s + 1);
            size_t tmp = order[s]; order[s] = order[r]; order[r] = tmp;
        }

        for (size_t start = 0; start < samples; start += batch_size) {
            // the last batch of an epoch may be smaller
            size_t m = (samples - start < batch_size) ? samples - start : batch_size;

            for (size_t k = 0; k < m; k++) {
                size_t s = order[start + k];
                // a column of a row-major matrix is strided: element (r, s) is at r * samples + s
                for (size_t r = 0; r < features; r++) batch[k].content[r] = train_data_input.content[r * samples + s];
                for (size_t r = 0; r < outputs; r++)  target[k].content[r] = train_data_target.content[r * samples + s];
            }

            printf("epoch %zu/%zu, sample %zu/%zu\r", e, epochs, start, samples);
            fflush(stdout);
            batch_fit(p, batch, target, m, learning_rate);
        }
    }

    printf("\n");

    for (size_t k = 0; k < batch_size; k++) {
        free(batch[k].content);
        free(target[k].content);
    }
    free(batch);
    free(target);
    free(order);
}









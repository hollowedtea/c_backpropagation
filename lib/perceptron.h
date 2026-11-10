#include "matrix.h"
#include <stddef.h>

typedef struct {
    size_t layers;
    matrix *matrices;
    matrix *biases;
    float (**activation_functions)(float *, size_t, size_t);
    float (*cost_function)(float *, float *);

} perceptron;

perceptron perceptron_init(size_t layer_count, size_t *layer_sizes, float (**activation_functions)(float *, size_t, size_t));
matrix forward_propagation(perceptron *p, matrix input_layer);


#include "matrix.h"
#include "activation_functions.h"
#include "cost_functions.h"
#include <stddef.h>

#ifndef _PERCEPTRON
#define _PERCEPTRON

typedef struct {
    size_t layers;
    matrix *matrices;
    matrix *biases;
    activation_function *activation_functions;
    cost_function cost_func;

} perceptron;

perceptron perceptron_init(size_t layer_count, size_t *layer_sizes, activation_function *activation_functions, cost_function cost_func, float range[2]);
matrix forward_propagation(perceptron *p, matrix input_layer);
#endif

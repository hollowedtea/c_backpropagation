#include <stdlib.h>
#include "matrix.h"

#ifndef _ACTIVATION_FUNCTIONS

#define _ACTIVATION_FUNCTIONS

typedef struct {
    float (*function)(float *preprocess_layer, size_t layer_length, size_t current_node);
    matrix (*jacobian)(float *preprocess_layer, size_t layer_length);
} activation_function;

activation_function construct_ReLU();
activation_function construct_softmax();

#endif

